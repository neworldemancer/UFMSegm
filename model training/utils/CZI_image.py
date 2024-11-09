"""
   Copyright 2015-2023, University of Bern, Laboratory for High Energy Physics and Theodor Kocher Institute, M. Vladymyrov

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.


Utility for working with multi-times & tiled/multiposition CZI files to extract image frames. wrapper over czifile
""" 

import numpy as np
import czifile
import copy
from tqdm.auto import tqdm

def segment_has_M_dim(s):
    de = s.dimension_entries
    dim_names = [de_i.dimension for de_i in de]
    return 'M' in dim_names
    
def tile_subblocks(img):
    return [s for s in img.subblocks() if segment_has_M_dim(s)]
    
def get_subblock_dim_ofs(s):
    return {d.dimension:d.start for d in s.dimension_entries}

def get_subblocks_dim_ofs(subblocks):
    do = [get_subblock_dim_ofs(s) for s in subblocks]
    k = do[0].keys()
    for do_i in do[1:]:
        assert do_i.keys() == k
    return do

def reduce_dims(dim_ofs, relative_coord=True):
    assert len(dim_ofs)>0
    
    k = dim_ofs[0].keys()
    
    vals = {ki:set() for ki in k}
    
    for do_i in dim_ofs:
        for k, v in do_i.items():
            vals[k].add(v)
            
    #print('vals', vals)
    
    removed_keys = {k for k, v in vals.items() if len(v) == 1}
    removed_keys.update({'X', 'Y'})
    #print('removed_keys', removed_keys)
    
    dim_ofs_upd = copy.deepcopy(dim_ofs)
    for do_i in dim_ofs_upd:
        for k in removed_keys:
            del do_i[k]
            
    coord_ranges = copy.deepcopy(vals)
    for k in removed_keys:
        del coord_ranges[k]
    
    XY = [(do_i['X'], do_i['Y']) for do_i in dim_ofs]
    if relative_coord:
        Xmin = min(vals['X'])
        Ymin = min(vals['Y'])
        XY = [(x-Xmin, y-Ymin) for x,y in XY]
    return dim_ofs_upd, coord_ranges, XY
    
def get_n_tiles(coord_ranges):
    return (max(coord_ranges['M'])+1) if 'M' in coord_ranges else 1
def get_n_channel(coord_ranges):
    return (max(coord_ranges['C'])+1) if 'C' in coord_ranges else 1
def get_n_times(coord_ranges):
    return (max(coord_ranges['T'])+1) if 'T' in coord_ranges else 1
def get_n_z(coord_ranges):
    return (max(coord_ranges['Z'])+1) if 'Z' in coord_ranges else 1
    
def get_data_dict(subblocks, dim_ofs):
    data = {
        tuple(do.values()):s.data().squeeze() for s, do in tqdm(zip(subblocks, dim_ofs))
    }
    return data
    
def get_data_dict(subblocks, dim_ofs):
    data = {
        tuple(do.values()):s.data().squeeze() for s, do in zip(subblocks, dim_ofs)
    }
    return data
    
def get_xy_dict(XY, dim_ofs):
    data = {
        tuple(do.values()):xy for xy, do in zip(XY, dim_ofs)
    }
    return data
    
def get_frame(data_dict, coord_ranges, tile, t, c, z):
    key = []
    
    key_LUT = {'M':tile, 'T':t, 'C':c, 'Z': z}
    for k in coord_ranges:
        key.append(key_LUT.get(k, 0))
        
    key = tuple(key)
    return data_dict.get(key, None)
    
def frame_is_missing(data_dict, coord_ranges, tile, t, c, z):
    key = []
    
    key_LUT = {'M':tile, 'T':t, 'C':c, 'Z': z}
    for k in coord_ranges:
        key.append(key_LUT.get(k, 0))
        
    key = tuple(key)
    return key not in data_dict
    
def get_frame_xy(xy_dict, coord_ranges, tile, t, c, z):
    key = []

    key_LUT = {'M':tile, 'T':t, 'C':c, 'Z': z}
    for k in coord_ranges:
        key.append(key_LUT.get(k, 0))

    key = tuple(key)
    return xy_dict.get(key, None)
    
class CZI_image:
    def __init__(self, file_name):
        with czifile.CziFile(file_name, detectmosaic=False) as img:
            subblocks = tile_subblocks(img)
            self.dim_ofs = get_subblocks_dim_ofs(subblocks)
            
            self.dim_ofs, self.coord_ranges, self.XY = reduce_dims(self.dim_ofs)
            
            self.n_tile = get_n_tiles(self.coord_ranges)
            self.n_c = get_n_channel(self.coord_ranges)
            self.n_t = get_n_times(self.coord_ranges)
            self.n_z = get_n_z(self.coord_ranges)
            
            self.data_dict = get_data_dict(subblocks=subblocks, dim_ofs=self.dim_ofs)
            self.xy_dict = get_xy_dict(XY=self.XY, dim_ofs=self.dim_ofs)
            
            for f in self.data_dict.values():
                self.h, self.w = f.shape
                self.dtype = f.dtype
                break
            
    def get_frame(self, tile=0, t=0, c=0, z=0):
        return get_frame(self.data_dict, self.coord_ranges, tile, t, c, z)
        
    def get_frame_fill_missing(self, tile=0, t=0, c=0, z=0):
        try:
            f = self.get_frame(tile=tile, t=t, c=c, z=z)
        except:
            f = None
        
        if f is None:
            f = np.zeros(shape=(self.h, self.w), dtype=self.dtype)
        
        return f
        
    def frame_is_missing(self, tile=0, t=0, c=0, z=0):
        return frame_is_missing(self.data_dict, self.coord_ranges, tile, t, c, z)
    
    def get_frame_xy(self, tile=0, t=0, c=0, z=0):
        return get_frame_xy(self.xy_dict, self.coord_ranges, tile, t, c, z)
    
