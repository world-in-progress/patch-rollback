import linecache
import numpy as np
import fastdb4py as fdb
from pathlib import Path
import multiprocessing as mp
from functools import partial

from .feature.fdb_feature import IndexLike, SideTopoInfo, Ne, Ns

def create_ne_fdb_parallel(ne_fn: str, fdb_fn: str):
    """Create NE FDB from NE file in parallel"""
    shared_name = 'shared_ne'
    ne_path = Path(ne_fn)
    if not ne_path.exists():
        raise FileNotFoundError(f'NE file not found: {ne_path}')
    
    # Get element count
    ne_f = open(ne_path, 'r', encoding='utf-8')
    element_count = sum(1 for _ in ne_f) + 1  # including virtual element 0
    ne_f.close()
    
    db = fdb.ORM.truncate([
        fdb.TableDefn(IndexLike, element_count * 10, 'isl1'),
        fdb.TableDefn(IndexLike, element_count * 10, 'isl2'),
        fdb.TableDefn(IndexLike, element_count * 10, 'isl3'),
        fdb.TableDefn(IndexLike, element_count * 10, 'isl4'),
        fdb.TableDefn(Ne, element_count)
    ])
    
    db.share(shared_name, close_after=False)
    
    # Add actual hydro elements in parallel
    batch_size = 50000
    batch_args = [i for i in range(1, element_count, batch_size)]
    batch_func = partial(
        _batch_ne_worker,
        ne_count=element_count,
        fdb_fn=shared_name,
        batch_size=batch_size,
        ne_file=ne_fn
    )
    
    num_processes = min(mp.cpu_count(), len(batch_args))
    with mp.Pool(processes=num_processes) as pool:
        pool.map(batch_func, batch_args)
    
    # Save to file and remove shared database
    fdb_path = Path(fdb_fn)
    fdb_path.parent.mkdir(parents=True, exist_ok=True)
    db.save(str(fdb_path))
    db.unlink()

def create_ns_fdb_parallel(ns_fn: str, fdb_fn: str):
    """Create NS FDB from NS file in parallel"""
    shared_name = 'shared_ns'
    ns_path = Path(ns_fn)
    if not ns_path.exists():
        raise FileNotFoundError(f'NS file not found: {ns_path}')
    
    # Get side count
    ns_f = open(ns_path, 'r')
    side_count = sum(1 for _ in ns_f) + 1  # including virtual side 0
    ns_f.close()
    
    db = fdb.ORM.truncate([
        fdb.TableDefn(Ns, side_count),
        fdb.TableDefn(SideTopoInfo, side_count * 3)
    ])
    
    db.share(shared_name, close_after=False)
    
    # Add actual sides in parallel
    batch_size = 50000
    batch_args = [i for i in range(1, side_count, batch_size)]
    batch_func = partial(
        _batch_ns_worker,
        ns_count=side_count,
        fdb_fn=shared_name,
        batch_size=batch_size,
        ns_file=ns_fn
    )
    
    num_procs = min(mp.cpu_count(), len(batch_args))
    with mp.Pool(processes=num_procs) as pool:
        pool.map(batch_func, batch_args)
        
    # Save to file and remove shared database
    fdb_path = Path(fdb_fn)
    fdb_path.parent.mkdir(parents=True, exist_ok=True)
    db.save(str(fdb_path))
    db.unlink()

def _batch_ne_worker(ne_si: int, ne_count: int, fdb_fn: str, batch_size: int, ne_file: str):
    db = fdb.ORM.load(fdb_fn)
    nes = db[Ne][Ne]
    e_xs = nes.column.x
    e_ys = nes.column.y
    e_zs = nes.column.z
    e_types = nes.column.type
    e_indices = nes.column.index
    e_lcount = nes.column.l_side_num
    e_rcount = nes.column.r_side_num
    e_bcount = nes.column.b_side_num
    e_tcount = nes.column.t_side_num
    
    isl1 = db[IndexLike]['isl1'].column.index
    isl2 = db[IndexLike]['isl2'].column.index
    isl3 = db[IndexLike]['isl3'].column.index
    isl4 = db[IndexLike]['isl4'].column.index
    
    for idx in range(ne_si, min(ne_si + batch_size, ne_count)):
        ne_record = linecache.getline(ne_file, idx)
        data = ne_record.split(',')
        indices_array = np.array([int(v) for v in data[5:-4]], dtype=np.uint32)
        
        # Set hydro element data directly to np arrays
        e_indices[idx] = int(data[0])
        e_xs[idx] = float(data[-4])
        e_ys[idx] = float(data[-3])
        e_zs[idx] = float(data[-2])
        e_types[idx] = int(data[-1])
        e_lcount[idx] = int(data[1])
        e_rcount[idx] = int(data[2])
        e_bcount[idx] = int(data[3])
        e_tcount[idx] = int(data[4])
        
        # Set side indices
        si_offset = idx * 10
        l_count = e_lcount[idx]
        r_count = e_rcount[idx]
        b_count = e_bcount[idx]
        t_count = e_tcount[idx]
        isl1[si_offset:si_offset + l_count] = indices_array[0:l_count]
        isl2[si_offset:si_offset + r_count] = indices_array[l_count:l_count + r_count]
        isl3[si_offset:si_offset + b_count] = indices_array[l_count + r_count:l_count + r_count + b_count]
        isl4[si_offset:si_offset + t_count] = indices_array[l_count + r_count + b_count:l_count + r_count + b_count + t_count]

def _batch_ns_worker(ns_si: int, ns_count: int, fdb_fn: str, batch_size: int, ns_file: str):
    db = fdb.ORM.load(fdb_fn)
    nss = db[Ns][Ns]
    s_indices = nss.column.index
    s_lengths = nss.column.length
    s_xs = nss.column.x
    s_ys = nss.column.y
    s_zs = nss.column.z
    s_attrs = nss.column.attr
    
    sts = db[SideTopoInfo][SideTopoInfo].column.info
    
    for idx in range(ns_si, min(ns_si + batch_size, ns_count)):
        ns_record = linecache.getline(ns_file, idx)
        data = ns_record.split(',')
        topo = np.array([int(v) for v in data[1:6]], dtype=np.uint32)
        sts_s = idx * 3
        if topo[0] == 1: # horizontal (connects bottom/top)
            sts[sts_s:sts_s + 3] = np.array([topo[0], topo[3], topo[4]], dtype=np.uint32)
        else: # vertical (connects left/right)
            sts[sts_s:sts_s + 3] = np.array([topo[0], topo[1], topo[2]], dtype=np.uint32)
        
        # Set side data directly to np arrays
        s_indices[idx] = int(data[0])
        s_lengths[idx] = float(data[6])
        s_xs[idx] = float(data[7])
        s_ys[idx] = float(data[8])
        s_zs[idx] = float(data[9])
        s_attrs[idx] = int(data[10])

def _check_ne_fdb(ne_fn: str, fdb_fn: str):
    db = fdb.ORM.load(fdb_fn, from_file=True)
    nes = db[Ne][Ne]
    isl1 = db[IndexLike]['isl1'].column.index
    isl2 = db[IndexLike]['isl2'].column.index
    isl3 = db[IndexLike]['isl3'].column.index
    isl4 = db[IndexLike]['isl4'].column.index
    
    with open(ne_fn, 'r', encoding='utf-8') as f:
        for line in f:
            record = line.split(',')
            idx = int(record[0])
            left_edge_num = int(record[1])
            right_edge_num = int(record[2])
            bottom_edge_num = int(record[3])
            top_edge_num = int(record[4])
            start = 5
            left_edges = [int(edge_idx) for edge_idx in record[start:start + left_edge_num]]
            start += left_edge_num
            right_edges = [int(edge_idx) for edge_idx in record[start:start + right_edge_num]]
            start += right_edge_num
            bottom_edges = [int(edge_idx) for edge_idx in record[start:start + bottom_edge_num]]
            start += bottom_edge_num
            top_edges = [int(edge_idx) for edge_idx in record[start:start + top_edge_num]]

            ne_element = nes[idx]
            assert ne_element.index == idx, f'Index mismatch: {ne_element.index} != {idx}'
            assert ne_element.l_side_num == left_edge_num, f'Left edge count mismatch at index {idx}'
            assert ne_element.r_side_num == right_edge_num, f'Right edge count mismatch at index {idx}'
            assert ne_element.b_side_num == bottom_edge_num, f'Bottom edge count mismatch at index {idx}'
            assert ne_element.t_side_num == top_edge_num, f'Top edge count mismatch at index {idx}'

            si_offset = idx * 10
            l_edges_db = isl1[si_offset:si_offset + left_edge_num].tolist()
            r_edges_db = isl2[si_offset:si_offset + right_edge_num].tolist()
            b_edges_db = isl3[si_offset:si_offset + bottom_edge_num].tolist()
            t_edges_db = isl4[si_offset:si_offset + top_edge_num].tolist()

            assert l_edges_db == left_edges, f'Left edges mismatch at index {idx}'
            assert r_edges_db == right_edges, f'Right edges mismatch at index {idx}'
            assert b_edges_db == bottom_edges, f'Bottom edges mismatch at index {idx}'
            assert t_edges_db == top_edges, f'Top edges mismatch at index {idx}'

    print('FDB NE data verification passed.')

def _check_ns_fdb(ns_fn: str, fdb_fn: str):
    db = fdb.ORM.load(fdb_fn, from_file=True)
    nss = db[Ns][Ns]
    
    with open(ns_fn, 'r', encoding='utf-8') as f:
        for line in f:
            record = line.split(',')
            idx = int(record[0])
            orient = int(record[1])
            left = int(record[2])
            right = int(record[3])
            bottom = int(record[4])
            top = int(record[5])
            length = float(record[6])
            x = float(record[7])
            y = float(record[8])
            z = float(record[9])
            attr = int(record[10])

            ns_element = nss[idx]
            ns_topo = db[SideTopoInfo][SideTopoInfo].column.info[idx * 3:(idx + 1) * 3]
            assert ns_element.index == idx, f'Index mismatch: {ns_element.index} != {idx}'
            if orient == 1:  # horizontal
                assert ns_topo[0] == orient, f'Orient mismatch at index {idx}'
                assert ns_topo[1] == bottom, f'Bottom mismatch at index {idx}'
                assert ns_topo[2] == top, f'Top mismatch at index {idx}'
            else:  # vertical
                assert ns_topo[0] == orient, f'Orient mismatch at index {idx}'
                assert ns_topo[1] == left, f'Left mismatch at index {idx}'
                assert ns_topo[2] == right, f'Right mismatch at index {idx}'
            assert abs(ns_element.length - length) < 1e-3, f'Length mismatch at index {idx}'
            assert abs(ns_element.x - x) < 0.1, f'X mismatch at index {idx}'
            assert abs(ns_element.y - y) < 0.1, f'Y mismatch at index {idx}'
            assert abs(ns_element.z - z) < 1e-3, f'Z mismatch at index {idx}'
            assert ns_element.attr == attr, f'Attr mismatch at index {idx}'

    print('FDB NS data verification passed.')

def preprocess(resource_ne: str, resource_ns: str, output_ne_fdb: str, output_ns_fdb: str, check: bool = False):
    processes = []
    p1 = mp.Process(target=create_ne_fdb_parallel, args=(resource_ne, output_ne_fdb))
    p2 = mp.Process(target=create_ns_fdb_parallel, args=(resource_ns, output_ns_fdb))
    p1.start()
    p2.start()
    processes.append(p1)
    processes.append(p2)
    
    for p in processes:
        p.join()
    
    if check:
        _check_ne_fdb(resource_ne, output_ne_fdb)
        _check_ns_fdb(resource_ns, output_ns_fdb)