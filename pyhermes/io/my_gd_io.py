import os

import numpy as np

from pyhermes.utils import func_util
from pyhermes.param.logbase import setup_logger 
from pyhermes.utils.func_util import get_fname_info



def _read_my_gd2_single(filename, ptype=1):
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    header_dtype = np.dtype([
        ("block1", np.int32),
        ("npart", np.uint32, 6),
        ("massarr", np.float64, 6),
        ("time", np.float64),
        ("redshift", np.float64),
        ("flag_sfr", np.int32),
        ("flag_feedback", np.int32),
        ("npartTotal", np.int32, (6,)),
        ("flag_cooling", np.int32),
        ("num_files", np.int32),
        ("BoxSize", np.float64),
        ("Omega0", np.float64),
        ("OmegaLambda", np.float64),
        ("HubbleParam", np.float64),
        ("fill", np.int32, 24),
        ("block2", np.int32),
    ])
    _out = {}
    with open(filename, 'rb') as f:
        # Read header
        _header = np.fromfile(f, dtype=header_dtype, count=1)[0]
        if _header['npart'][0] != 0:
            logger.error("Currently, only DM-only snapshots in Gadget1/2/3/4 legacy-format1 are supported, but it appears your file contains SPH particles")
            func_util.safe_exit(1)
        npart = _header['npart'][ptype]
        # Read pos
        _blksize = np.fromfile(f, dtype=np.int32, count=1)
        _out['pos'] = np.fromfile(f, dtype=np.float32, count=int(npart*3)).reshape(-1,3)
        _blksize = np.fromfile(f, dtype=np.int32, count=1)
        # Read vel
        _blksize = np.fromfile(f, dtype=np.int32, count=1)
        _out['vel'] = np.fromfile(f, dtype=np.float32, count=int(npart*3)).reshape(-1,3)
        _blksize = np.fromfile(f, dtype=np.int32, count=1)
        # Skip IDs
        _blksize = np.fromfile(f, dtype=np.int32, count=1)
        _ = np.fromfile(f, dtype=np.int32, count=int(npart))
        _blksize = np.fromfile(f, dtype=np.int32, count=1)
        # Read/Construct mass
        if _header['massarr'][ptype] != 0:
            masstab = True
            _out['mass'] = _header['massarr'][ptype]
        else:
            masstab = False
            _blksize = np.fromfile(f, dtype=np.int32, count=1)
            _out['mass'] = np.fromfile(f, dtype=np.float32, count=int(npart))
            _blksize = np.fromfile(f, dtype=np.int32, count=1)
    return _out, masstab

def read_my_gd2(file):
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    files = func_util.find_subsplit_files(file)
    out = {
        'pos' : [],
        'vel' : [],
        'mass': [],
    }
    masstab_pre = None
    for file in files:
        _out_part, masstab = _read_my_gd2_single(file)
        out['pos'].append(_out_part['pos'])
        out['vel'].append(_out_part['vel'])
        if masstab_pre is not None:
            if masstab == masstab_pre:
                if not masstab:
                    out['mass'].append(_out_part['mass'])
                    masstab_pre = masstab
                else:
                    out['mass'] = _out_part['mass']
                    masstab_pre = masstab
            else:
                logger.error("Inconsistent masstab across subsnaps. Please make sure you're using data from the same simulation.")
                func_util.safe_exit(1)
        else:
            if not masstab:
                out['mass'].append(_out_part['mass'])
                masstab_pre = masstab
            else:
                out['mass'] = _out_part['mass']
                masstab_pre = masstab
    out['pos']   = np.concatenate(out['pos'], axis=0)
    out['vel']   = np.concatenate(out['vel'], axis=0)
    out['size']  = out['pos'].shape[0]
    if not masstab_pre:
        out['mass'] = np.concatenate(out['mass'], axis=0)
    return out


### For Gadget-fof file(s) ###

def extdict_sfields(header, out_dict, selected_fields):
    for field in selected_fields:
        out_dict[field] = header[field]
    return out_dict

def _read_my_gd2_fof_single(filename):
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    header_dtype = np.dtype([
        ('block1', np.int32),       
        ('Ngroups', np.int64),        
        ('Nsubhalos', np.int64),      
        ('Nids', np.int64),           
        ('TotNgroups', np.int64),     
        ('TotNsubhalos', np.int64),   
        ('TotNids', np.int64),        
        ('num_files', np.int32),      
        ('dummy', np.int32),         
        ('time', np.float64),           
        ('redshift', np.float64),       
        ('BoxSize', np.float64),        
        ('block2', np.int32),        
    ])
    _out = {}
    _sfields = [
        'Ngroups', 
        'Nsubhalos', 
        'Nids', 
        'TotNgroups', 
        'TotNsubhalos', 
        'TotNids', 
        'num_files', 
        'time', 
        'redshift', 
        'BoxSize'
    ]
    with open(filename, 'rb') as f:
        _header = np.fromfile(f, dtype=header_dtype, count=1)
        extdict_sfields(_header, _out, _sfields)
        if _out['Nsubhalos'] != 0:
            logger.error("Currently, the <gadget-fof> reader supports only FoF, i.e., Gadget compiled using the 'FOF' Config flag but without 'SUBFIND'.")
            func_util.safe_exit(1)
        # Skip GroupLen
        _blksize = np.fromfile(f, dtype=np.int32, count=1)
        _ = np.fromfile(f, dtype=np.int32, count=int(_out['Ngroups']))
        _blksize = np.fromfile(f, dtype=np.int32, count=1)
        # Read GroupMass
        _blksize = np.fromfile(f, dtype=np.int32, count=1)
        _mass = np.fromfile(f, dtype=np.float32, count=int(_out['Ngroups']))
        _out['mass'] = _mass
        _blksize = np.fromfile(f, dtype=np.int32, count=1)
        # Read GroupPos
        _blksize = np.fromfile(f, dtype=np.int32, count=1)
        _pos = np.fromfile(f, dtype=np.float32, count=int(_out['Ngroups']*3))
        _out['pos'] = _pos.reshape(-1,3)
        _blksize = np.fromfile(f, dtype=np.int32, count=1)
        # Read GroupVel
        _blksize = np.fromfile(f, dtype=np.int32, count=1)
        _vel = np.fromfile(f, dtype=np.float32, count=int(_out['Ngroups']*3))
        _out['vel'] = _vel.reshape(-1,3)
        _blksize = np.fromfile(f, dtype=np.int32, count=1)
        # Skip GroupLenType
        pass
        # Skip GroupOffsetType
        pass
        # Skip GroupMassType
        pass
        # Skip GroupAscale
        pass
    return _out

def read_my_gd2_fof(file):
    files = func_util.find_subsplit_files(file)
    out = {
        'TotNgroups': 0,
        'TotNsubhalos': 0,
        'TotNids': 0,
        'num_files': 0,
        'time': 0.0,
        'redshift': 0.0,
        'BoxSize': 0.0,
        'mass': [],
        'pos': [],
        'vel': []
    }
    for file in files:
        _out_part = _read_my_gd2_fof_single(file)
        out['TotNgroups'] = _out_part['TotNgroups']
        out['TotNsubhalos'] = _out_part['TotNsubhalos']
        out['TotNids'] = _out_part['TotNids']
        out['num_files'] = _out_part['num_files']
        out['time'] = _out_part['time']
        out['redshift'] = _out_part['redshift']
        out['BoxSize'] = _out_part['BoxSize']
        out['mass'].append(_out_part['mass'])
        out['pos'].append(_out_part['pos'])
        out['vel'].append(_out_part['vel'])
    out['mass']  = np.concatenate(out['mass'], axis=0)
    out['pos']   = np.concatenate(out['pos'], axis=0)
    out['vel']   = np.concatenate(out['vel'], axis=0)
    out['size']  = out['pos'].shape[0]
    return out