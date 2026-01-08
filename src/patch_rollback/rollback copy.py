import numpy as np
import fastdb4py as fdb
from pathlib import Path

from .crm import Patch
from .model.patch import PatchMeta
from .model.schema import GridSchema
from .preprocessor import preprocess
from .fdb_feature import Ne, Ns, IndexLike

def rollback(
    patch_name: str, schema_name: str,
    epsg: int,
    element_resolution: list[list[float]],
    source_ne: str, source_ns: str, target_patch: str, target_schema: str
):
    # Preprocess ##############################################
    
    ne_fdb_fn = Path.absolute(Path('./fdb/ne.fdb'))
    ns_fdb_fn = Path.absolute(Path('./fdb/ns.fdb'))
    ne_fdb_fn.parent.mkdir(parents=True, exist_ok=True)
    ns_fdb_fn.parent.mkdir(parents=True, exist_ok=True)
    
    target_patch_fn = Path.absolute(Path(target_patch))
    target_schema_fn = Path.absolute(Path(target_schema))
    target_patch_fn.mkdir(parents=True, exist_ok=True)
    target_schema_fn.mkdir(parents=True, exist_ok=True)
    
    # Clean up existing patch topology to ensure fresh start
    if (target_patch_fn / 'patch.topo.parquet').exists():
        (target_patch_fn / 'patch.topo.parquet').unlink()
    
    preprocess(source_ne, source_ns, str(ne_fdb_fn), str(ns_fdb_fn), check=False)
    
    # Helper ##################################################

    import taichi as ti
    def copy_to_taichi(np_array: np.ndarray, dtype: any, shape: any) -> ti.MatrixField | ti.ScalarField:
        """
        Copy a numpy array to a Taichi field.
        This function can facilitate the conversion from column data in FastDB table to Taichi field.
        
        Note: tichi not initialized inside this function, Taichi should be initialized before calling this function.
        
        e.g.:
            column_data = nes.column.x  # numpy array from FastDB
            field = copy_to_taichi(column_data, ti.f32, None)
        """
        if shape is None:
            shape = np_array.shape
        else:
            if np_array.shape != shape:
                np_array = np_array.reshape(shape)
            
        field = ti.field(dtype=dtype, shape=np_array.shape if shape is None else shape)
        field.from_numpy(np_array)
        return field
    
    def encode_index_batch(levels: np.ndarray, global_ids: np.ndarray) -> np.ndarray:
        return (levels.astype(np.uint64) << 32) | global_ids.astype(np.uint64)

    def _decode_index_batch(encoded: np.ndarray) -> tuple[list[int], list[int]]:
        levels = (encoded >> 32).astype(np.uint8)
        global_ids = (encoded & 0xFFFFFFFF).astype(np.uint32)
        return levels.tolist(), global_ids.tolist()
    
    # Main Rollback Process ###################################
    
    ti.init(arch=ti.gpu)
    
    ne_fdb = fdb.ORM.load(ne_fdb_fn, from_file=True)
    ns_fdb = fdb.ORM.load(ns_fdb_fn, from_file=True)
    
    nes = ne_fdb[Ne][Ne]
    nss = ns_fdb[Ns][Ns]
    # sts = ns_fdb[SideTopoInfo][SideTopoInfo]
    
    e_num = len(nes)
    
    levels = ti.field(dtype=ti.u8, shape=e_num)
    global_ids = ti.field(dtype=ti.u32, shape=e_num)
    exs = copy_to_taichi(nes.column.x, ti.f32, None)
    eys = copy_to_taichi(nes.column.y, ti.f32, None)
    sxs = copy_to_taichi(nss.column.x, ti.f32, None)
    sys = copy_to_taichi(nss.column.y, ti.f32, None)
    isl1 = copy_to_taichi(ne_fdb[IndexLike]['isl1'].column.index, ti.i32, [e_num, 10])
    isl3 = copy_to_taichi(ne_fdb[IndexLike]['isl3'].column.index, ti.i32, [e_num, 10])
    
    bbox = ti.field(dtype=ti.f32, shape=4) # xmin, ymin, xmax, ymax
    
    level_num = len(element_resolution)
    resolutions = ti.field(dtype=ti.f32, shape=(len(element_resolution), 2))
    resolutions.from_numpy(np.array(element_resolution, dtype=np.float32))
    
    @ti.kernel
    def get_element_levels_and_bbox():
        bbox[0] = bbox[1] = 99999999.0
        bbox[2] = bbox[3] = 0.0
        
        for ei in range(1, ti.i32(e_num)):
            lsi0 = isl1[ei, 0]
            lsi2 = isl3[ei, 0]
            slh = ti.floor(exs[ei] - sxs[lsi0] + 0.5) * 2.0
            slv = ti.floor(eys[ei] - sys[lsi2] + 0.5) * 2.0
            
            for level_idx in range(ti.i32(level_num)):
                if ti.abs(slh - resolutions[level_idx, 0]) < 1e-6:
                    levels[ei] = ti.u8(level_idx + 1)
                    break
            
            # Update bbox
            xmin = exs[ei] - slh * 0.5
            ymin = eys[ei] - slv * 0.5
            xmax = exs[ei] + slh * 0.5
            ymax = eys[ei] + slv * 0.5
            ti.atomic_min(bbox[0], xmin)
            ti.atomic_min(bbox[1], ymin)
            ti.atomic_max(bbox[2], xmax)
            ti.atomic_max(bbox[3], ymax)
    
    get_element_levels_and_bbox()
    levels_np = levels.to_numpy()[1:]   # skip the first virtual element
    
    # Count elements in each level
    level_counts = {}
    for l in levels_np:
        if l in level_counts:
            level_counts[l] += 1
        else:
            level_counts[l] = 1
    print('Element counts in each level:')
    for l, count in level_counts.items():
        print(f'    Level {l}: {count} elements')

    # Print bounding box
    bbox_np = bbox.to_numpy()
    print(f'Bounding Box: xmin={bbox_np[0]}, ymin={bbox_np[1]}, xmax={bbox_np[2]}, ymax={bbox_np[3]}')
    
    domain_width = bbox_np[2] - bbox_np[0]
    domain_height = bbox_np[3] - bbox_np[1]
    print(f'Domain Size: width={domain_width}, height={domain_height}')
    
    level_infos: list[list[int]] = [[] for _ in range(level_num)]   # level_infos[level] = cell rows and cols of that level
    for level_idx in range(level_num):
        level = level_idx + 1
        cell_size_x = resolutions[level_idx, 0]
        cell_size_y = resolutions[level_idx, 1]
        cell_cols = int(np.ceil(domain_width / cell_size_x))
        cell_rows = int(np.ceil(domain_height / cell_size_y))
        level_infos[level_idx] = [cell_rows, cell_cols]
        print(f'Level {level}: cell size=({cell_size_x}, {cell_size_y}), grid=({cell_rows} rows, {cell_cols} cols)')
    
    level_infos_t = ti.field(dtype=ti.u32, shape=(level_num, 2))
    level_infos_t.from_numpy(np.array(level_infos, dtype=np.uint32).reshape((level_num, 2)))
    
    # Create schema
    schema = GridSchema(
        name=schema_name,
        epsg=epsg,
        alignment_origin=(bbox_np[0], bbox_np[1]),
        grid_info=element_resolution
    )
    with open(target_schema_fn / f'{schema_name}.json', 'w') as f:
        f.write(schema.model_dump_json(indent=4))
    
    # Create patch meta
    patch_meta = PatchMeta(
        name=patch_name,
        bounds=(float(bbox_np[0]), float(bbox_np[1]), float(bbox_np[2]), float(bbox_np[3]))
    )
    with open(target_patch_fn / f'meta.json', 'w') as f:
        f.write(patch_meta.model_dump_json(indent=4))
    
    # Create patch crm
    patch = Patch(
        schema_file_path=str(target_schema_fn / f'{schema_name}.json'),
        grid_patch_path=str(target_patch_fn)
    )

    # Sync level_infos from Patch to ensure consistency in Global ID calculation
    # Patch calculates grid dimensions hierarchically (power of 2 typically), 
    # which might differ from simple ceil(width/res) if domain size isn't perfect multiple.
    for level_idx in range(level_num):
        level = level_idx + 1
        # Patch level_info index is the level itself (0 is virtual root, 1 is Level 1)
        if level < len(patch.level_info):
            p_info = patch.level_info[level]
            # level_infos stores [rows, cols]
            level_infos[level_idx] = [p_info['height'], p_info['width']]
            # print(f'Sync Level {level}: grid=({p_info["height"]} rows, {p_info["width"]} cols)')
    
    level_infos_t.from_numpy(np.array(level_infos, dtype=np.uint32).reshape((level_num, 2)))
    
    @ti.kernel
    def assign_global_ids():
        domain_w = bbox[2] - bbox[0]
        domain_h = bbox[3] - bbox[1]
        
        for ei in range(1, ti.i32(e_num)):
            x = exs[ei]
            y = eys[ei]
            level_idx = levels[ei]
            
            # Use effective cell size (stretched to fill domain) to match Patch logic
            l_rows = level_infos_t[level_idx - 1, 0]
            l_cols = level_infos_t[level_idx - 1, 1]
            
            cell_size_x = domain_w / l_cols
            cell_size_y = domain_h / l_rows
            
            origin_x = bbox[0]
            origin_y = bbox[1]
            
            # Use small epsilon to avoid boundary issues, though floor handles strictly less
            col = ti.u32(ti.floor((x - origin_x) / cell_size_x))
            row = ti.u32(ti.floor((y - origin_y) / cell_size_y))
            
            # Clamp to ensuring valid range (handle precision edge cases at max boundary)
            if col >= l_cols:
                col = l_cols - 1
            if row >= l_rows:
                row = l_rows - 1
                
            global_ids[ei] = row * l_cols + col
    
    assign_global_ids()
    global_ids_np = global_ids.to_numpy()[1:]   # skip the first virtual element
    
    # Collect subdividable gids in each level
    activated_cell_info: list[list[int]] = [[] for _ in range(level_num)]  # activated_cell_info[level] = [gid1, gid2, ...]
    for ei in range(e_num - 1):
        level = levels_np[ei]
        gid = global_ids_np[ei]
        activated_cell_info[level - 1].append(gid)
    
    # Collect subdividable cells' parents from the finest level to the coarsest level
    # To ensure all parent cells are included, we iteratively go through each level
    subdividable_gids_in_levels: list[set[int]] = [set() for _ in range(level_num)]
    current_level = level_num
    for level in range(current_level, 1, -1):   # Skip level 0 (non-existent) and level 1 (no parents)
        cell_num = len(activated_cell_info[level - 1])
        
        # First, try to get parents of already subdividable gids in this level
        _, p_gids = patch.get_parents(
            levels=[level] * len(subdividable_gids_in_levels[level - 1]),
            global_ids=list(subdividable_gids_in_levels[level - 1])
        )
        for pgid in p_gids:
            subdividable_gids_in_levels[level - 2].add(pgid)
        
        # Then, get parents of activated gids in this level
        _, p_gids = patch.get_parents(
            levels= [level] * cell_num, 
            global_ids= activated_cell_info[level - 1]
        )
        for pgid in p_gids:
            subdividable_gids_in_levels[level - 2].add(pgid)
    
    # Subdivide cells in each level
    for level_idx, subdividable_gids in enumerate(subdividable_gids_in_levels):
        level = level_idx + 1
        patch.subdivide_grids(
            levels= [level] * len(subdividable_gids),
            global_ids= list(subdividable_gids)
        )
        
    # Get all activated cells in the patch
    active_grid_indices = patch.get_active_grid_indices()
    
    # Decode levels and global ids from recorded active grid indices
    real_active_indices = encode_index_batch(levels_np, global_ids_np)
    
    # Mark those not in real_active_indices as deleted by numpy set difference
    deleted_indices = np.setdiff1d(active_grid_indices, real_active_indices)
    if len(deleted_indices) > 0:
        print(f'Deleting {len(deleted_indices)} grids that are not in the real active set...')
        d_levels, d_global_ids = _decode_index_batch(deleted_indices)
        patch.delete_grids(
            levels=d_levels,
            global_ids=d_global_ids
        )
    
    # Print final stats
    final_active_indices = patch.get_active_grid_indices()
    print(f'Final active grids in patch: {len(final_active_indices)}')
    
    # Save patch
    patch.save()
    