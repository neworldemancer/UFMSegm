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
"""

from scipy.interpolate import interp1d as interp
import numpy as np

from . import imgio as iio
from . import net_utils as nu

tf = None
def _inp_tf():
  global tf
  import tensorflow as _tf
  
  ver = _tf.__version__
  ver_maj = int(ver.split('.')[0])
  if ver_maj == 1:
    tf = _tf
  else:
    tf = _tf.compat.v1
    
def inp_tf():
  if tf is None:
    _inp_tf()
    
class NormHist:
    def __init__(self, ref_raw_tmpl, ref_msk_tmpl=None, idx_min=0, idx_max=5, mask_ofs=0, 
                 dev=0,
                 border=100
                ):
        inp_tf()
        self.dev = dev
        self.graph, self.flt_in, self.flt_out = self.init_conv()
        
        self.raw = None
        self.rawf = None
        self.rawfm = None
        
        if ref_msk_tmpl is None:
            path = ref_raw_tmpl
            
            with open(path+'/start.txt', 'rt') as f:
                idx_min = int(f.readline())
                idx_max = idx_min + 5
                
            ref_raw_tmpl = path+'/raw/%03d.tif'
            ref_msk_tmpl = path+'/bin_mask_bg/%03d.png'
        
        self.ref_stack, self.ref_mask = self.load_stack_mask(ref_raw_tmpl,
                                                             ref_msk_tmpl,
                                                             idx_min, idx_max, mask_ofs)
        self.ref_raw_tmpl = ref_raw_tmpl
        self.ref_msk_tmpl = ref_msk_tmpl
        self.border = border
        self.ref_arr = self.get_float_pix_arr(self.ref_stack, self.ref_mask, own=True)
        self.tgt_col_vals, self.tgt_cdf = self.get_cols_cdf(self.ref_arr)
        self.last_lut = None
    
    def init_conv(self):
        g = tf.Graph()
        with g.as_default():
            f = np.array(
                [
                    [1,2,1],
                    [2,0,2],
                    [1,2,1],
                ])/12
            f = f.reshape(3, 3, 1, 1)
            
            inp = tf.placeholder(tf.float32, shape=(1, None, None, 1), name="Input")
            flt = tf.constant(f, dtype=tf.float32, name="Filter")
            out = tf.nn.conv2d(inp, flt, strides=[1]*4, padding='SAME', name='Out')
        return g, inp, out
    
    def get_stack_delta(self, stack):
        stack0 = stack
        sh = stack0.shape
        if len(sh)==4 and sh[-1] != 1:
            raise ValueError('channel dimension should be 0 or 1')
            
        if len(sh)==3:
            stack0 = stack0.reshape(list(sh)+[1])
            
        stack0 = stack0[:, np.newaxis, ...]
            
        with nu.TFSession(self.graph, devices=[self.dev]) as sess:
            o_stack = []
            for im in stack0:
                o_stack.append(sess.run(self.flt_out, feed_dict={self.flt_in:im}))
                
            o_stack = np.asarray(o_stack)
            o_stack = o_stack[:, 0, ..., 0]
        return o_stack
    
    def load_stack_mask(self, ref_raw_tmpl, ref_msk_tmpl, idx_min, idx_max, mask_ofs):
        raw = iio.read_image_stack(ref_raw_tmpl, idx_max-idx_min, idx_min).astype(np.float32)
        mbg = iio.read_image_stack(ref_msk_tmpl, idx_max-idx_min, idx_min+mask_ofs).astype(np.float32)
        
        if len(raw.shape) == 4:
            raw = raw[..., 0]
        if len(mbg.shape) == 4:
            mbg = mbg[..., 0]
        return raw, mbg
    
    def get_float_pix_arr(self, stack, mask, own=False):
        # remove boudary artifacts
        stack = stack.astype(np.float32)
        
        # filter, get float vals, prevent quantization
        stack_near = self.get_stack_delta(stack)
        #sh = stack_near.shape
        
        stack = stack[:, self.border:-self.border, self.border:-self.border]
        mask = mask[:, self.border:-self.border, self.border:-self.border]
        stack_near = stack_near[:, self.border:-self.border, self.border:-self.border]
        
        r_d = stack_near.flatten()
        raw = stack.flatten()
        msk = mask.flatten()

        u = (r_d>raw).astype(np.int32)
        d = 1-u

        raw_clipped = np.clip(raw, 1, 254)
        r_r = 1/raw_clipped
        r_255r = 1/(255-raw_clipped)
        
        # delta \in [-0.5, 0.5]; -0.5 for 0, 0 for r_d==raw, 0.5 for 255
        delta = (r_d-raw)*(r_r * d + r_255r*u)

        raw_f = raw + delta
        #raw = raw.reshape(sh)
        #r_d = r_d.reshape(sh)
        #raw_f = raw_f.reshape(sh)
        
        raw_bg = raw_f*msk
        
        sh = stack_near.shape
        if own:
            self.raw = raw.reshape(sh)
            self.rawf = raw_f.reshape(sh)
            self.rawfm = raw_bg.reshape(sh)
        
        raw_bg = raw_bg[(raw_bg>0)*(raw_bg<253)]  # mb too much, but why not
        #print(raw_bg.min(), raw_bg.max())
        
        return raw_bg
    
    def get_cols_cdf(self, arr, n=1):
        col_vals = np.linspace(0, 256, 256*n+1)
        hist = np.histogram(arr, col_vals)
        
        pdf = hist[0]
        cdf = np.cumsum(pdf).astype(np.float64)
        cdf /= cdf[-1]
        
        cut_left = max(0, np.sum(cdf == cdf[0])-1)
        cut_right = max(0, np.sum(cdf == cdf[-1])-1)
        
        begin = cut_left
        end = len(cdf) - cut_right
        
        col_vals = col_vals[begin:end]
        cdf = cdf[begin:end]
        
        return col_vals, cdf
    
    def get_lut(self, src_col_vals, src_cdf, scale=1):
        intr_col_src_to_prob = interp(src_col_vals, src_cdf,
                                      kind='linear', fill_value="extrapolate")
        intr_prob_to_col_tgt = interp(self.tgt_cdf, self.tgt_col_vals,
                                      kind='linear', fill_value="extrapolate",
                                      assume_sorted=False)
        
        col_vals_n = np.linspace(0, 255, 256*scale)
        col_vals_lut = intr_prob_to_col_tgt(intr_col_src_to_prob(col_vals_n))
        
        self.last_lut = col_vals_lut
        return col_vals_lut
        
    def correct_stack(self, stack, from_stack_ref, from_mask_res, n=1):
        from_ref_arr = self.get_float_pix_arr(from_stack_ref, from_mask_res)
        src_col_vals, src_cdf = self.get_cols_cdf(from_ref_arr)
        
        self.last_src_col_vals, self.last_src_cdf = src_col_vals, src_cdf 
        
        if n!=1:
            # requires convolution on all dataset and things similar to big part of get_float_pix_arr
            # requires refactoring
            raise ValueError('n!=1 corresponds to floating point pixel colors. Not yet implemented')
            
        lut = self.get_lut(src_col_vals, src_cdf, scale=n)
        
        # stack = self.get_float_stack(stack)
        sh = stack.shape
        stack_flat = stack.flatten()
        
        stack_n_int = np.clip((stack_flat*n).astype(np.int32), 0, 255*n)
        stack_corrected = lut[stack_n_int]
        stack_corrected = stack_corrected.round().clip(0, 255)
        stack_corrected = stack_corrected.reshape(sh).astype(np.uint8)
        
        return stack_corrected      
        
    def correct_stack_float(self, stack, from_stack_ref, from_mask_res, n=1):
        raw_ref = from_stack_ref.flatten()
        msk_ref = from_mask_res.flatten()
        raw_ref_bg = raw_ref*msk_ref
        from_ref_arr = raw_ref_bg[(raw_ref_bg>0.)*(raw_ref_bg<253.)]
        
        src_col_vals, src_cdf = self.get_cols_cdf(from_ref_arr, n)
        
        self.last_src_col_vals, self.last_src_cdf = src_col_vals, src_cdf 
            
        lut = self.get_lut(src_col_vals, src_cdf, scale=n)
        
        # stack = self.get_float_stack(stack)
        sh = stack.shape
        stack_flat = stack.flatten()
        
        stack_n_int = np.clip((stack_flat*n).astype(np.int32), 0, 255*n)
        stack_corrected = lut[stack_n_int]
        stack_corrected = stack_corrected.round().clip(0, 255)
        stack_corrected = stack_corrected.reshape(sh).astype(np.uint8)
        
        return stack_corrected