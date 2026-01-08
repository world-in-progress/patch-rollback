import logging
import shutil
from pathlib import Path
from src.patch_rollback import rollback

logging.basicConfig(level=logging.INFO)

resolutions = [
    [64.0, 64.0],   # level 1
    [32.0, 32.0],   # level 2
    [16.0, 16.0],   # level 3
    [8.0, 8.0],     # level 4
    [4.0, 4.0],     # level 5
    [2.0, 2.0],     # level 6
    [1.0, 1.0],     # level 7
]

if __name__ == '__main__':
    output_path = Path('./resource/output/')
    if output_path.exists():
        shutil.rmtree(output_path)
    
    rollback(
        patch_name='example_patch',
        schema_name='example_schema',
        epsg=2326,
        element_resolution=resolutions,
        source_ne='./resource/ne.txt',
        source_ns='./resource/ns.txt',
        target_patch='./resource/output/patch/',
        target_schema='./resource/output/schema/'
    )
    
    # preprocess(
    #     resource_ne='./resource/ne.txt',
    #     resource_ns='./resource/ns.txt',
    #     output_ne_fdb='./fdb/ne.fdb',
    #     output_ns_fdb='./fdb/ns.fdb',
    #     check=False
    # )
    
    # import fastdb4py as fdb
    
    # db = fdb.ORM.load('shared_ne')
    # db.unlink()