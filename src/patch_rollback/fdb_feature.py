import fastdb4py as fdb

class IndexLike(fdb.Feature):
    index: fdb.U32

class U8Value(fdb.Feature):
    value: fdb.U8

class Ne(fdb.Feature):
    index: fdb.U32
    x: fdb.F32
    y: fdb.F32
    z: fdb.F32
    l_side_num: fdb.U32
    r_side_num: fdb.U32
    b_side_num: fdb.U32
    t_side_num: fdb.U32
    type: fdb.U8

class Ns(fdb.Feature):
    index: fdb.U32
    length: fdb.F32
    x: fdb.F32
    y: fdb.F32
    z: fdb.F32
    attr: fdb.U8

class SideTopoInfo(fdb.Feature):
    """
    Used to store side topology information for NsData
    A side has 5 topology info:
    0: orient (1: horizontal, 2: vertical)
    1: left or bottom Ne index (0: no Ne)
    2: right or top Ne index (0: no Ne)
    """
    info: fdb.U32