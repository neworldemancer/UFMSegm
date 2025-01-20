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

# imports

import os
import time
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from time import time as timer
from time import sleep
import shutil

plt.style.use('ggplot')
from threading import Thread

from ipywidgets import Layout, HBox, VBox, Text, Button, Output, HTML, Checkbox
from IPython.display import display

from . import imgio as iio
from . import predictor as pr
from .histnorm import NormHist

import dataclasses
import json
import traceback


# make proc config dataclass
@dataclasses.dataclass
class ProcConfig:
    next_dataset_id: int = 0
    dataset_id_first_run: int = -1
    dataset_id_last_run: int = -1

    tile_overlap_fraction: float = 0.1

    skip_completed: bool = True

    datasets_path: str = os.path.abspath('../../datasets_seg') + os.path.sep
    models_path: str = os.path.abspath('../../trained models') + os.path.sep
    ref_ds_path: str = models_path + 'ref datasets' + os.path.sep
    datasets_path_proc: str = 'D:\\UFMTrack\\datasets_seg\\'  # 'Q:\\' #
    VF_root: str = 'C:\\VivoFollow\\'
    rds_id: int = 7

    fast_model_itr: int = 30001
    main_model_itr: int = 35001

    fast_model_name: str = 'model_BBB_BN_TV_FCN4_HN_CDC_2D_2021.04.17_22-54'  # 2D, 1 tf 2021 best
    main_model_name: str = 'model_BBB_BN_TV_FCNx16_HN_CDC_2021.07.06_16-25'  # 3D, 2021 best

    cuda_dev_ids: list = dataclasses.field(
        default_factory=lambda: [0])  # [0, 1, 2, 3, 4, 5, 6, 7] # list of available GPUs


@dataclasses.dataclass
class PathConfig:
    datasets_path: str = ""
    datasets_path_proc: str = ""
    models_path: str = ""
    ref_ds_path: str = ""
    VF_root: str = ""
    proc_bin_path: str = ""
    ds_info_file: str = ""


# methods for loading and storing the config to file
_cfg_filename = 'proc_cfg.json'


def load_proc_cfg(cfg_path=None):
    cfg_path = cfg_path or os.path.join(os.path.abspath(os.path.curdir), _cfg_filename)
    if os.path.exists(cfg_path):
        with open(cfg_path, 'rt') as f:
            cfg = ProcConfig(**json.load(f))
    else:
        cfg = ProcConfig()
    return cfg


def save_proc_cfg(cfg, cfg_path=None):
    cfg_path = cfg_path or os.path.join(os.path.abspath(os.path.curdir), _cfg_filename)
    with open(cfg_path, 'wt') as f:
        # human-readable, 4 spaces indentation
        json.dump(dataclasses.asdict(cfg), f, indent=4)


def get_tiled_groups(datasets_is_tile,
                     datasets_tile_group_id, datasets_tile_idx,
                     datasets_ids, datasets_hw, merged_ds_start_idx,
                     overlap_frac=0.2
                     ):
    all_tile_dataset_ids = []
    all_non_tile_dataset_ids = []

    tiles_info = {}  # id: [[ny, nx, tile_dx, tile_dy], list_of_tile_dataset_ids],

    curr_group_id = None
    curr_group_ds_idxs = []

    def get_group_size_ofs(group_hw, n_tiles):
        group_hw = np.array(group_hw)
        # print(group_hw)
        mean_hw = group_hw.mean(axis=0)

        deviation_hw = np.abs(mean_hw - group_hw) / mean_hw
        incompatible_hw = deviation_hw > 0.01  # 1% difference is too much

        assert not np.any(incompatible_hw), 'size of time tiles are too different'
        # print(group_hw, mean_hw, incompatible_hw)

        h, w = mean_hw
        overlap = w * overlap_frac  # 10% of width
        tile_dx = int(w - overlap)
        tile_dy = int(h - overlap)

        # simple rules for identifying tile configuration:
        ny_nx = []
        ny_p_nx = []
        for ny in range(1, 1 + int(np.floor(np.sqrt(n_tiles)))):
            nx = n_tiles // ny
            res = n_tiles - ny * nx
            if res == 0:
                ny_nx.append([ny, nx])
                ny_p_nx.append(ny + nx)
        idx = np.argmin(ny_p_nx)
        ny, nx = ny_nx[idx]

        # print(ny_nx)
        # print(ny_nx[idx])

        return [ny, nx, tile_dx, tile_dy]

    def fill_group(curr_group_ds_idxs, datasets_ids, datasets_hw,
                   all_tile_dataset_ids, all_non_tile_dataset_ids,
                   tiles_info, group_idx):

        if len(curr_group_ds_idxs) < 2:
            # cancel group
            all_non_tile_dataset_ids.extend([datasets_ids[idx] for idx in curr_group_ds_idxs])
        else:
            groups_ds_ids = [datasets_ids[idx] for idx in curr_group_ds_idxs]
            print(f'processing group {group_idx}, ds ids: {groups_ds_ids}')
            all_tile_dataset_ids.extend(groups_ds_ids)

            n_tiles = len(groups_ds_ids)

            group_hw = [datasets_hw[idx] for idx in curr_group_ds_idxs]

            size_ofs = get_group_size_ofs(group_hw, n_tiles)
            tiles_info[merged_ds_start_idx + group_idx[0]] = [size_ofs, groups_ds_ids]

            group_idx[0] += 1

    group_idx = [0]  # in a list to be modified inside fill_group function
    for idx, (is_tile, group_id, tile_idx, ds_id) in enumerate(zip(datasets_is_tile,
                                                                   datasets_tile_group_id,
                                                                   datasets_tile_idx,
                                                                   datasets_ids)):
        group_close = (curr_group_id is not None) and (not is_tile or group_id != curr_group_id)

        if group_close:
            fill_group(curr_group_ds_idxs, datasets_ids, datasets_hw,
                       all_tile_dataset_ids, all_non_tile_dataset_ids,
                       tiles_info, group_idx)

            curr_group_id = None
            curr_group_ds_idxs = []

        if not is_tile:
            all_non_tile_dataset_ids.append(ds_id)
        else:
            assert (curr_group_id == group_id or curr_group_id is None)
            curr_group_id = group_id
            assert tile_idx == len(curr_group_ds_idxs)
            curr_group_ds_idxs.append(idx)

    if curr_group_id is not None:
        fill_group(curr_group_ds_idxs, datasets_ids, datasets_hw,
                   all_tile_dataset_ids, all_non_tile_dataset_ids,
                   tiles_info, group_idx)

    return all_non_tile_dataset_ids, all_tile_dataset_ids, tiles_info


class PredictorMT:
    def __init__(self, mod_path, mod_itr,
                 gpu_ids,
                 batch_sz, input_sz,
                 io_map,
                 z_border=2  # one side
                 ):
        self.z_border = z_border
        self.gpu_ids = gpu_ids
        self.n_thr = len(self.gpu_ids)

        self.predictors = [pr.Predictor(mod_path, mod_itr,
                                        device_id=None,
                                        device_ids=gpu_ids,
                                        gpuid=i,
                                        batch_sz=batch_sz, input_sz=input_sz,
                                        in_out_dict=io_map,
                                        ) for i, dev_id in enumerate(gpu_ids)]
        self.res = []

    def normalize_stack(self, stack, norm_stack, normalization_percentile_range):
        return self.predictors[0].normalize_stack(stack, norm_stack, normalization_percentile_range)

    def split_stack(self, stack):
        n_z = len(stack)
        n_z_chunk = (n_z + self.n_thr - 1) // self.n_thr

        begin_end_pairs = [[max(0, i * n_z_chunk - self.z_border),
                            min(n_z, (i + 1) * n_z_chunk + self.z_border)] for i in range(self.n_thr)]

        chunks = [stack[b:e] for b, e in begin_end_pairs]
        return chunks

    def merge_result(self, stacks):
        # self.z_border

        cropped_overlap = [
            s[0 if i == 0 else self.z_border:
              len(s) if i == (self.n_thr - 1) else -self.z_border
            ]
            for i, s in enumerate(stacks)
        ]

        res = np.concatenate(cropped_overlap, axis=0)
        return res

    def _predict_image_stack(self, stack_norm, margin, keep_edge, edge_size, thr_idx):
        pred = self.predictors[thr_idx]
        res = pred.predict_image_stack(stack_norm, margin, keep_edge, edge_size)
        self.res[thr_idx] = res

    def predict_image_stack(self, stack_norm, margin, keep_edge=True, edge_size=0):
        # split stack
        chunks = self.split_stack(stack_norm)

        self.res = [None for i in range(self.n_thr)]

        # make threads
        threads = [Thread(target=self._predict_image_stack,
                          args=(chunks[i], margin, keep_edge, edge_size, i)
                          ) for i in range(self.n_thr)]

        # run all
        for t in threads:
            t.start()

        # wait all
        for t in threads:
            t.join()

        merged = self.merge_result(self.res)
        # merge output
        # processed.shape == (192, 1077, 1405, 3), i.e. plain np tensor

        return merged


class PredictorMT:
    def __init__(self, mod_path, mod_itr,
                 gpu_ids,
                 batch_sz, input_sz,
                 io_map,
                 z_border=2  # one side
                 ):
        self.z_border = z_border
        self.gpu_ids = gpu_ids
        self.n_thr = len(self.gpu_ids)

        self.predictors = [pr.Predictor(mod_path, mod_itr,
                                        device_id=None,
                                        device_ids=gpu_ids,
                                        gpuid=i,
                                        batch_sz=batch_sz, input_sz=input_sz,
                                        in_out_dict=io_map,
                                        ) for i, dev_id in enumerate(gpu_ids)]
        self.res = []

    def normalize_stack(self, stack, norm_stack, normalization_percentile_range):
        return self.predictors[0].normalize_stack(stack, norm_stack, normalization_percentile_range)

    def split_stack(self, stack):
        n_z = len(stack)
        n_z_chunk = (n_z + self.n_thr - 1) // self.n_thr

        begin_end_pairs = [[max(0, i * n_z_chunk - self.z_border),
                            min(n_z, (i + 1) * n_z_chunk + self.z_border)] for i in range(self.n_thr)]

        chunks = [stack[b:e] for b, e in begin_end_pairs]
        return chunks

    def merge_result(self, stacks):
        # self.z_border

        cropped_overlap = [
            s[0 if i == 0 else self.z_border:
              len(s) if i == (self.n_thr - 1) else -self.z_border
            ]
            for i, s in enumerate(stacks)
        ]

        res = np.concatenate(cropped_overlap, axis=0)
        return res

    def _predict_image_stack(self, stack_norm, margin, keep_edge, edge_size, thr_idx):
        pred = self.predictors[thr_idx]
        res = pred.predict_image_stack(stack_norm, margin, keep_edge, edge_size)
        self.res[thr_idx] = res

    def predict_image_stack(self, stack_norm, margin, keep_edge=True, edge_size=0):
        # split stack
        chunks = self.split_stack(stack_norm)

        self.res = [None for _ in range(self.n_thr)]

        # make threads
        threads = [Thread(target=self._predict_image_stack,
                          args=(chunks[i], margin, keep_edge, edge_size, i)
                          ) for i in range(self.n_thr)]

        # run all
        for t in threads:
            t.start()

        # wait all
        for t in threads:
            t.join()

        merged = self.merge_result(self.res)
        # merge output
        # processed.shape == (192, 1077, 1405, 3), i.e. plain np tensor

        return merged


def proc_path(path, pcfg: PathConfig):
    return path.replace(pcfg.datasets_path, pcfg.datasets_path_proc).replace('/', '\\').replace('%', '%%')


def make_genmask_bat(ds_idx, stck_tmpl, num, pcfg: PathConfig):
    # gen background mask generation bat file. Alignment pipeline is used
    prog = pcfg.proc_bin_path + 'TimeFramesAligner_64.exe '
    cfg = f'-cfg:{pcfg.proc_bin_path}TFAligner_bg_mask.cfg '

    mask_tmpl = pcfg.datasets_path + '%03d/' % ds_idx + 'pred_for_algn/' + 'img_%03d.png'
    tgt_dir = pcfg.datasets_path + '%03d/' % ds_idx + 'imgs_aligned_all/'
    s_t = proc_path(stck_tmpl, pcfg)
    m_t = proc_path(mask_tmpl, pcfg)
    tgd = proc_path(tgt_dir, pcfg)

    cmd = '@echo off\n'
    cmd += 'pushd %~dp0\n'
    cmd += prog + cfg
    cmd += '-savedir:"%s" ' % tgd
    cmd += '-n_itr:0 '
    cmd += '-stack:"%s" ' % s_t
    cmd += '-mask:"%s" ' % m_t
    cmd += '-n_frames:%d ' % num

    cmd += '\n'

    cmd += 'popd\n'

    bat_file = pcfg.datasets_path + '%03d/' % ds_idx + 'genmask.bat'
    # print(cmd)
    with open(bat_file, 'wt') as f:
        f.write(cmd)


def gen_collective_genmask(ds_idx_list, pcfg: PathConfig):
    """
    Generated all genmask bat file: executes individual genmask
    returns: full path to created batfile
    """
    # cmd = '@echo off\n'
    cmd = ''

    for ds_idx in ds_idx_list:
        cmd += 'call %03d' % ds_idx + '\\genmask.bat \n'
    bat_file = pcfg.datasets_path + 'genmask_' + str(ds_idx_list) + '.bat'
    with open(bat_file, 'wt') as f:
        f.write(cmd)

    return bat_file


def start_remote_job(pcfg: PathConfig, bat_file, iteration_sleep_time=5):
    if bat_file is None:
        return

    itr = 0
    while os.path.exists(pcfg.datasets_path + 'remote.bat'):
        if itr == 0:
            print('Waiting previous remote job to be done...')
        itr += 1
        sleep(iteration_sleep_time)
    shutil.copy(bat_file, pcfg.datasets_path + 'remote.bat')


def wait_for_file(file_path, iteration_sleep_time=5, end_sleep_time=10):
    while not os.path.exists(file_path):
        time.sleep(iteration_sleep_time)

    time.sleep(end_sleep_time)


def show_mdl_smpl(raw, cdc):
    mdl=len(cdc)//2
    _=iio.draw_samples(
        (
            raw[mdl,...,0],
            cdc[mdl,...,0],
            cdc[mdl,...,1],
            cdc[mdl,...,2]
        ), color_range=(0,256)
    )


def gen_proc_bat(ds_ids_list, pcfg: PathConfig, script_name):
    cmd = '@echo off\n'

    for ds_idx in ds_ids_list:
        cmd += 'call %03d' % ds_idx + f'\\{script_name}.bat \n'

    if len(ds_ids_list) == 0:
        return None
    first_last = str(ds_ids_list[0])
    if len(ds_ids_list) > 1:
        first_last += '-' + str(ds_ids_list[-1])

    bat_file = pcfg.datasets_path + f'{script_name}_[' + first_last + '].bat'
    with open(bat_file, 'wt') as f:
        f.write(cmd)

    return bat_file


def gen_segm_bat(ds_ids_list, pcfg: PathConfig):
    return gen_proc_bat(ds_ids_list, pcfg, script_name='segment')


def read_info_file(ds_inf_file_path):
    try:
        with open(ds_inf_file_path, 'rt') as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []
    ds_inf = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        subs = line.split('-')
        k = int(subs[0].strip())
        v = '-'.join(subs[1:])

        ds_inf[k] = v.strip()
    return ds_inf


def save_ds_inf(ds_inf_file_path, ds_inf, new_ds_list):
    with open(ds_inf_file_path, 'wt') as f:
        for k, v in ds_inf.items():
            f.write(f'{k} - {v}\n')

        next_idx = (max(ds_inf.keys()) + 1) if len(ds_inf) else 0
        for i, title in enumerate(new_ds_list):
            f.write(f'{next_idx + i} - {title}\n')

def run_segmentation(cfg: ProcConfig):
    nb_start_t = timer()
    proc_res = {'status': False, 'last_dataset_idx': -1}

    # all work here
    try:
        # datasets
        datasets_ids = list(range(cfg.dataset_id_first_run,
                                  cfg.dataset_id_last_run + 1))  # range of tiled datasets, expected to be last folders, and 8 tiles
        merged_ds_idx0 = datasets_ids[-1] + 1

        # path from inference node
        pcfg = PathConfig(datasets_path=cfg.datasets_path,
                          datasets_path_proc=cfg.datasets_path_proc,
                          models_path=cfg.models_path,
                          ref_ds_path=cfg.ref_ds_path,
                          VF_root=cfg.VF_root,
                          proc_bin_path=cfg.VF_root + 'bin\\',
                          ds_info_file=cfg.datasets_path+'info.txt'
                          )

        fast_model_itr = cfg.fast_model_itr
        main_model_itr = cfg.main_model_itr

        fast_model_name = cfg.fast_model_name
        main_model_name = cfg.main_model_name

        mod_path_f = pcfg.models_path + '/' + fast_model_name

        mod_path = pcfg.models_path + '/' + main_model_name

        # path from win processing node. Should be consistent with cfg files.
        # Remote processing should be avoided for performance reasons,
        # but if used, should be within closed private network

        dev_ids = cfg.cuda_dev_ids
        # dev_id = dev_ids[0]

        auto_proc = True

        rds_id = cfg.rds_id

        tile_overlap_frac = cfg.tile_overlap_fraction
        skip_completed = cfg.skip_completed

        nums = []
        datasets_names = []
        fluo_present = []

        datasets_tmplts = []
        datasetsf_tmplts = []
        datasets_normed = []
        datasets_hist_normed = []

        block_boundaries = []

        datasets_is_tile = []
        datasets_tile_idx = []
        datasets_tile_group_id = []
        datasets_hw = []  # image hight/width

        tile_sep = '_tile'

        # prepare datasets info for processing
        for ds_idx in datasets_ids:
            # read info file
            info_file_name = pcfg.datasets_path + '%03d/info.txt' % ds_idx
            with open(info_file_name, 'rt') as f:
                ds_name = f.readline()

            # list files in dir
            ds_path = pcfg.datasets_path + '%03d/%s/' % (ds_idx, ds_name)
            all_tif_files = [n for n in os.listdir(ds_path) if ds_name in n]

            all_sfx = sorted([n.replace(ds_name + '_t', '').replace('.tif', '') for n in all_tif_files])
            last_sfx = all_sfx[-1]

            is_tile = tile_sep in ds_name
            if is_tile:
                grpid_idx = ds_name.split(tile_sep)
                assert len(
                    grpid_idx) == 2, f'unexpected dataset name format: "{ds_name}", contains multiple "{tile_sep}"'
                tile_group_id, tile_idx = grpid_idx
                tile_idx = int(tile_idx) - 1
            else:
                tile_group_id, tile_idx = '', -1

            # get # channel, #time points
            has_fluo = 'c' in last_sfx

            if has_fluo:
                n_t_c_s = last_sfx.split('c')
                n_t, n_c = [int(s) for s in n_t_c_s]
            else:
                n_c = 1
                n_t = int(last_sfx)

            # fill tmplts, num, fluo present,
            tmpl_t = '_t%0' + '%d' % len('%d' % n_t) + 'd'
            tmpl_c = 'c%0' + '%d' % len('%d' % n_c if has_fluo else '1') + 'd'

            dataset_tmplt = ds_path + ds_name + tmpl_t + (tmpl_c % 1 if has_fluo else '') + '.tif'
            datasetf_tmplts = [ds_path + ds_name + tmpl_t + (tmpl_c % ch) + '.tif' for ch in
                               range(2, n_c + 1)] if has_fluo else []
            dataset_normed = ds_path + 'normed'

            dataset_hist_normed = pcfg.datasets_path + '%03d/' % ds_idx + 'hist_normed/img_%03d.tif'  # png

            # print(n_t, n_c, has_fluo, tmpl_t, tmpl_c, dataset_tmplt, datasetf_tmplts, dataset_normed)

            stack = iio.read_image_stack(dataset_tmplt, 1, 1)
            hw = stack.shape[1:]

            # use num per dataset
            nums.append(n_t)
            datasets_names.append(ds_name)

            datasets_is_tile.append(is_tile)
            datasets_tile_idx.append(tile_idx)
            datasets_tile_group_id.append(tile_group_id)

            datasets_hw.append(hw)

            fluo_present.append(has_fluo)

            datasets_tmplts.append(dataset_tmplt)
            datasetsf_tmplts.append(datasetf_tmplts)
            datasets_normed.append(dataset_normed)

            datasets_hist_normed.append(dataset_hist_normed)

            block_info_path = os.path.join(pcfg.datasets_path, '%03d' % ds_idx, 'block_info.txt')
            if os.path.exists(block_info_path):
                with open(block_info_path, 'rt') as f:
                    txt = f.readline()
            else:
                txt = ''
            if txt:
                block_boundary = [[int(bi) for bi in b.split(' ')] for b in txt.split('|')]
            else:
                block_boundary = [[0, n_t]]
            block_boundaries.append(block_boundary)

        al_datasets_tmplts = [pcfg.datasets_path + '%03d' % i + '/imgs_aligned_all/raw/%03d.png' for i, n in
                              zip(datasets_ids, datasets_names)]
        al_datasets_normed = [pcfg.datasets_path + '%03d' % i + '/imgs_aligned_all/normed' for i, n in
                              zip(datasets_ids, datasets_names)]

        all_non_tile_dataset_ids, all_tile_dataset_ids, tiles_info = get_tiled_groups(datasets_is_tile,
                                                                                      datasets_tile_group_id,
                                                                                      datasets_tile_idx,
                                                                                      datasets_ids, datasets_hw,
                                                                                      merged_ds_start_idx=merged_ds_idx0,
                                                                                      overlap_frac=tile_overlap_frac
                                                                                      )

        # load models
        # fast model

        io_map = {'in': 'stack:0', 'out': 'ModelOutput:0', 'out_channels': [0, 1, 2]}
        batch_sz = 8

        # print('AFTER EXECUTING THIS FIRST TIME FOR A MODEL (Needs single-threaded version!) - RESTART THE KERNEL')
        print(mod_path_f, fast_model_itr)
        # pred_fast = pr.Predictor(mod_path_f, fast_model_itr,
        #                          device_id=dev_id,
        #                          batch_sz=batch_sz, input_sz=[512,512],
        #                          in_out_dict=io_map)

        pred_fast_mt = PredictorMT(mod_path_f, fast_model_itr,
                                   gpu_ids=dev_ids,
                                   batch_sz=batch_sz, input_sz=[512, 512],
                                   io_map=io_map)

        io_map = {'in': 'stack:0', 'out': 'ModelOutput:0', 'out_channels': [0, 1, 2]}

        batch_sz = 1

        print(mod_path, main_model_itr)
        # pred = pr.Predictor(mod_path, main_model_itr,
        #                     device_id=dev_id,
        #                     batch_sz=batch_sz, input_sz=[512,512],
        #                     in_out_dict=io_map)
        pred_mt = PredictorMT(mod_path, main_model_itr,
                              gpu_ids=dev_ids,
                              batch_sz=batch_sz, input_sz=[512, 512],
                              io_map=io_map)

        for tmpl, nnrm, num in zip(datasets_tmplts, datasets_normed, nums):
            if skip_completed and os.path.exists(nnrm + '.npz'):
                print(f'skipping {tmpl} - {nnrm} already exists')
                continue

            print(tmpl)
            stack = iio.read_image_stack(tmpl, num, 1)

            print('saving...')
            if len(stack.shape) == 4:
                stack = stack[..., 0]

            for idx, im in enumerate(stack):
                iio.save_image(im, tmpl % (idx + 1))

            stack_center = stack[:, 130:-130, 130:-130]

            print('normalizing...')
            stack_norm = pr.Predictor.normalize_stack(stack=stack,
                                                      norm_stack=stack_center,
                                                      normalization_percentile_range=(2.5, 97.5))
            print('saving normalized...')
            np.savez(nnrm, stack_norm)

        t0 = timer()
        for i, (nnrm, num, ds_idx, stck_tmpl) in enumerate(zip(datasets_normed, nums, datasets_ids, datasets_tmplts)):
            test_done_file = pcfg.datasets_path + '%03d/' % ds_idx + 'imgs_aligned_all/bin_mask_bg/%03d.png' % (num - 1)
            if skip_completed and os.path.exists(test_done_file):
                print(f'skipping {nnrm} - {test_done_file} already aligned')
                continue

            # read normalized
            stack_normf = np.load(nnrm + '.npz')
            for stack_norm in stack_normf.values():
                break
            if len(stack_normf) == 0:
                print('no data in', nnrm)
                continue

            # process
            processed = pred_fast_mt.predict_image_stack(stack_norm, margin=4, edge_size=30)

            # visualize
            # show_mdl_smpl(stack_norm, processed)

            print('saving dataset ', ds_idx, end='\r')
            path = pcfg.datasets_path + '%03d/' % ds_idx + '/pred_for_algn/'
            os.makedirs(path, exist_ok=True)
            for i, img in enumerate(processed):
                pimg = Image.fromarray(img[..., 0])
                pimg.save(path + 'img_%03d.png' % i, quality=99)

            # make maskgen bat file and run if needed
            make_genmask_bat(ds_idx, stck_tmpl, num, pcfg)

            if auto_proc:
                bat_file = gen_collective_genmask([ds_idx], pcfg)
                start_remote_job(pcfg, bat_file)

        if not auto_proc:
            for ds_idx, stck_tmpl, num in zip(datasets_ids, datasets_tmplts, nums):
                make_genmask_bat(ds_idx, stck_tmpl, num, pcfg)
            bat_file = gen_collective_genmask(datasets_ids, pcfg)

        t1 = timer()
        print(t1 - t0, 's')

        # wait masks done
        i = len(datasets_tmplts) - 1
        idx = datasets_ids[i]
        test_file = pcfg.datasets_path + '%03d/' % idx + 'imgs_aligned_all/bin_mask_bg/%03d.png' % (nums[i] - 1)

        wait_for_file(test_file)

        nh = NormHist(pcfg.ref_ds_path + r'/%02d' % rds_id, dev=0)

        for i, tmpl, tmpl_hn, boundaries, num in zip(datasets_ids, datasets_tmplts, datasets_hist_normed,
                                                     block_boundaries, nums):
            test_file = tmpl_hn % (num - 1)
            if skip_completed and os.path.exists(test_file):
                print(f'skipping {tmpl} - {test_file} already hist normalized')
                continue

            print(tmpl)
            stack = iio.read_image_stack(tmpl, num, 1)
            if len(stack.shape) == 4:
                stack = stack[..., 0]

            stack_normed_blocks = []

            mask_tmpl = pcfg.datasets_path + '%03d/' % i + 'imgs_aligned_all/bin_mask_bg/%03d.png'

            for idx, block_boundary in enumerate(boundaries):
                bl_begin, bl_end = block_boundary
                stack_src = stack[bl_begin: bl_end]
                stack_src_ref = stack_src[-5:] if idx == 0 else stack_src[:5]
                print('normalizing block', idx, 'range', block_boundary, '...', end='\r')

                mask_start_idx = (bl_end - 5) if idx == 0 else bl_begin
                mask_ref = iio.read_image_stack(mask_tmpl, 5, mask_start_idx)
                if len(mask_ref.shape) == 4:
                    mask_ref = mask_ref[..., 0]

                stack_normed_block = nh.correct_stack(stack_src, stack_src_ref, mask_ref)
                # fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                # ax[0].plot(nh.last_lut)
                # ax[1].plot(nh.last_lut[200:])
                # plt.show()
                stack_normed_blocks.append(stack_normed_block)

            stack_normed_blocks = np.concatenate(stack_normed_blocks, axis=0)

            print('saving...                                     ', end='\r')
            os.makedirs(os.path.dirname(tmpl_hn), exist_ok=True)
            for idx, im in enumerate(stack_normed_blocks):
                iio.save_image(im, tmpl_hn % idx)

        # gen alignment  bat file
        prog = pcfg.proc_bin_path + 'TimeFramesAligner_64.exe '
        cfg = f'-cfg:{pcfg.proc_bin_path}TFAligner.cfg '

        rmdir_statement = lambda path: """IF exist %s (
          rmdir /s /q %s
        )\n""" % (path, path)

        for ds_idx, stck_tmpl, stck_aux, has_aux, num in zip(datasets_ids, datasets_hist_normed, datasetsf_tmplts,
                                                             fluo_present, nums):
            mask_tmpl = pcfg.datasets_path + '%03d/' % ds_idx + 'pred_for_algn/' + 'img_%03d.png'
            tgt_dir = pcfg.datasets_path + '%03d/' % ds_idx + 'imgs_aligned_all/'
            s_t = proc_path(stck_tmpl, pcfg)
            m_t = proc_path(mask_tmpl, pcfg)
            tgd = proc_path(tgt_dir, pcfg)

            cmd = '@echo off\n'
            cmd = 'pushd %~dp0\n'
            cmd += prog + cfg
            cmd += '-savedir:"%s" ' % tgd
            cmd += '-n_itr:3 '
            cmd += '-stack:"%s" ' % s_t
            cmd += '-raw_idx_0:0 '
            cmd += '-mask:"%s" ' % m_t
            cmd += '-n_frames:%d ' % num

            if has_aux:
                for aux_id, aux_tmpl in enumerate(stck_aux):
                    s_t = proc_path(aux_tmpl, pcfg)
                    cmd += '-stack_%d:"%s" ' % (aux_id, s_t)
                    cmd += '-stack_%d_subpixel ' % aux_id
                    cmd += '-stack_%d_start:1 ' % aux_id

            cmd += '\n'

            cmd += rmdir_statement('"%sraw"' % tgd)
            cmd += 'move "%scorrected" "%sraw"\n' % (tgd, tgd)

            if has_aux:
                for aux_id, _ in enumerate(stck_aux):
                    cmd += rmdir_statement('"%sflr%d"' % (tgd, aux_id + 1))
                    cmd += 'move "%scorrected_st_%d" "%sflr%d"\n' % (tgd, aux_id, tgd, aux_id + 1)
            cmd += 'popd\n'

            bat_file = pcfg.datasets_path + '%03d/' % ds_idx + 'align.bat'
            # print(cmd)
            with open(bat_file, 'wt') as f:
                f.write(cmd)

        # gen all alignments bat file
        cmd = '@echo off\n'

        for ds_idx, al_datasets_tmpl, num in zip(datasets_ids, al_datasets_tmplts, nums):
            test_file = al_datasets_tmpl % (num - 1)  # last dataset, last file
            if skip_completed and os.path.exists(test_file):
                print(f'skipping timeframe alignment for {al_datasets_tmpl} - {test_file} already aligned')
                continue
            cmd += 'call %03d' % ds_idx + '\\align.bat \n'

        first_last = str(datasets_ids[0])
        if len(datasets_ids) > 1:
            first_last += '-' + str(datasets_ids[-1])
        bat_file = pcfg.datasets_path + 'align_[' + first_last + '].bat'
        with open(bat_file, 'wt') as f:
            f.write(cmd)

        start_remote_job(pcfg, bat_file)

        # wait alignment done
        i = len(al_datasets_tmplts) - 1
        test_file = al_datasets_tmplts[i] % (nums[i] - 1)  # last dataset, first file
        while not os.path.exists(test_file):
            time.sleep(5)

        time.sleep(10)

        # normalization
        for tmpl, nnrm, num in zip(al_datasets_tmplts, al_datasets_normed, nums):
            if skip_completed and os.path.exists(nnrm + '.npz'):
                print(f'skipping {tmpl} - {nnrm} already aligned')
                continue

            print(tmpl)
            stack = iio.read_image_stack(tmpl, num)
            stack_center = stack[:, 130:-130, 130:-130]
            stack_norm = pr.Predictor.normalize_stack(stack=stack,
                                                      norm_stack=stack_center,
                                                      normalization_percentile_range=(2.5, 97.5)
                                                      )
            np.savez(nnrm, stack_norm)

        # Prediction by aligned
        t0 = timer()
        for i, (nnrm, ds_idx, num) in enumerate(zip(al_datasets_normed, datasets_ids, nums)):
            test_file = pcfg.datasets_path + '%03d/' % ds_idx + 'pred_cdc/cell/img_%03d.png' % (num - 1)
            if skip_completed and os.path.exists(test_file):
                print(f'skipping inference {nnrm} - {test_file} already aligned')
                continue

            # load normalized
            stack_normf = np.load(nnrm + '.npz')
            for stack_norm in stack_normf.values():
                break
            # process
            processed = pred_mt.predict_image_stack(stack_norm, margin=8, edge_size=60)

            # save processed
            print(ds_idx, end='\r')
            pref = os.path.join(pcfg.datasets_path, '%03d' % ds_idx)
            path_cell = os.path.join(pref, 'pred_cdc', 'cell')
            path_diap = os.path.join(pref, 'pred_cdc', 'diap')
            path_cntr = os.path.join(pref, 'pred_cdc', 'cntr')
            path_cntC = os.path.join(pref, 'pred_cdc', 'cntC')

            os.makedirs(path_cell, exist_ok=True)
            os.makedirs(path_diap, exist_ok=True)
            os.makedirs(path_cntr, exist_ok=True)
            os.makedirs(path_cntC, exist_ok=True)

            for i, img in enumerate(processed):
                for t, path in enumerate([path_cell, path_diap, path_cntr]):
                    pimg = Image.fromarray(img[..., t])
                    im_path = os.path.join(path, 'img_%03d.png' % i)
                    pimg.save(im_path, quality=99)

                cell_im = img[..., 0]
                cntr_im = img[..., 2].copy()
                mask_no_cell = cell_im <= 85
                cntr_im[mask_no_cell] = 0

                pimg = Image.fromarray(cntr_im)
                im_path = os.path.join(path_cntC, 'img_%03d.png' % i)
                pimg.save(im_path, quality=99)

            del processed
            del stack_norm

        t1 = timer()
        print(t1 - t0, 's')

        # Tile merging
        merged_tiles_ids = list(tiles_info.keys())
        # gen lists
        for merged_idx, ti in tiles_info.items():
            ny, nx, tile_dx, tile_dy = ti[0]
            tile_ids = ti[1]

            info_path = os.path.join(pcfg.datasets_path, '%03d' % merged_idx, 'tiles_info')
            os.makedirs(info_path, exist_ok=True)

            lists = ['', '', '', '', '', '']

            for idx in tile_ids:
                lists[0] += pcfg.datasets_path_proc + '%03d' % idx + r'\imgs_aligned_all\raw\%03d.png' + '\n'
                lists[1] += pcfg.datasets_path_proc + '%03d' % idx + r'\pred_cdc\cell\img_%03d.png' + '\n'
                lists[2] += pcfg.datasets_path_proc + '%03d' % idx + r'\pred_cdc\diap\img_%03d.png' + '\n'
                lists[3] += pcfg.datasets_path_proc + '%03d' % idx + r'\pred_cdc\cntC\img_%03d.png' + '\n'
                lists[4] += pcfg.datasets_path_proc + '%03d' % idx + r'\imgs_aligned_all\flr1\%03d.png' + '\n'
                lists[5] += pcfg.datasets_path_proc + '%03d' % idx + r'\imgs_aligned_all\flr2\%03d.png' + '\n'

            tmap = '%d %d\n' % (nx, ny)
            idx = -1
            for iy in range(ny):
                for ix in range(nx):
                    idx += 1

                    tmap += '%d %d %d %d %d\n' % (idx, ix, iy, ix * tile_dx, iy * tile_dy)

            for tlist, fname in zip(lists, ['raw.tl', 'cell.tl', 'diap.tl', 'cntc.tl', 'flr1.tl', 'flr2.tl']):
                fpath = os.path.join(info_path, fname)
                with open(fpath, 'wt') as f:
                    f.write(tlist)

            fpath = os.path.join(info_path, 'map.tm')
            with open(fpath, 'wt') as f:
                f.write(tmap)

        # gen merging bat file
        prog_tal = pcfg.proc_bin_path + 'proc_iv.bat'

        ds_id_to_idx = {i: idx for idx, i in enumerate(datasets_ids)}

        for idx in tiles_info:
            cmd = '@echo off\n'
            cmd = 'pushd ' + pcfg.proc_bin_path + '\n'

            tile_idx = ds_id_to_idx[tiles_info[idx][1][0]]
            n_tf = nums[tile_idx]
            cmd += 'call ' + prog_tal + ' %03d' % idx + ' %d' % n_tf + ' %d' % len(datasetsf_tmplts[tile_idx])
            cmd += '\n'
            cmd += 'popd\n'

            bat_file = pcfg.datasets_path + '%03d/' % idx + 'merge.bat'
            with open(bat_file, 'wt') as f:
                f.write(cmd)

        # gen all merging bat file
        cmd = '@echo off\n'

        for ds_idx in tiles_info:
            test_file = pcfg.datasets_path + '%03d/' % ds_idx + 'imgs_aligned_all/raw/%03d.png' % 0  # first file
            if skip_completed and os.path.exists(test_file):
                print(f'skipping merging for {ds_idx} - {test_file} already aligned')
                continue

            cmd += 'call %03d' % ds_idx + '\\merge.bat \n'

        bat_file = pcfg.datasets_path + 'merge_' + str(merged_tiles_ids) + '.bat'
        with open(bat_file, 'wt') as f:
            f.write(cmd)

        if auto_proc:
            start_remote_job(pcfg, bat_file)

        # update datasets info file - add raw for each of the merged datasets id + title
        # the title is formed as f'{tiles_info[first_tile_idx]}-{tiles_info[last_tile_idx]}'
        # read info file into dict
        ds_inf = read_info_file(pcfg.ds_info_file)

        # add new datasets
        new_ds_list = []
        for ds_id, ((ny, nx, tile_dx, tile_dy), list_of_tile_dataset_ids) in tiles_info.items():
            first_tile_idx = min(list_of_tile_dataset_ids)
            last_tile_idx = max(list_of_tile_dataset_ids)
            title = f'{ds_inf[first_tile_idx]}-{ds_inf[last_tile_idx]}'
            new_ds_list.append(title)

        # save new info file
        save_ds_inf(pcfg.ds_info_file, ds_inf, new_ds_list)
        new_ds_inf = read_info_file(pcfg.ds_info_file)
        if max(new_ds_inf.keys()) != max(tiles_info.keys()):
            print(f'warning: potential errirs in the datasets info file: {pcfg.ds_info_file}'
                  f' - last dataset id: {max(new_ds_inf.keys())}'
                  f'tilkes info: {str(tiles_info)}')

        # Cell segmentation
        # gen segmentation bat file
        prog_seg = pcfg.proc_bin_path + 'proc_ds_flr_n.bat'

        for i, idx in enumerate(datasets_ids + merged_tiles_ids):
            test_idx = i if idx not in merged_tiles_ids else ds_id_to_idx[tiles_info[idx][1][0]]
            has_flour = fluo_present[test_idx]

            cmd = '@echo off\n'
            cmd = 'pushd ' + pcfg.proc_bin_path + '\n'
            cmd += 'call ' + prog_seg + ' %03d' % idx + ' %d' % nums[test_idx] + ' %d' % len(datasetsf_tmplts[test_idx])
            cmd += '\n'
            cmd += 'popd\n'

            bat_file = pcfg.datasets_path + '%03d/' % idx + 'segment.bat'
            with open(bat_file, 'wt') as f:
                f.write(cmd)

        # gen all segmentation bat file
        bat_file_all_separate = gen_segm_bat(datasets_ids, pcfg)
        bat_file_all_m = gen_segm_bat(merged_tiles_ids, pcfg)
        bat_file_all_non_m = gen_segm_bat(all_non_tile_dataset_ids, pcfg)

        # process tiles
        # if auto_proc:
        #     start_remote_job(pcfg, bat_file_all_separate)

        # wait merging and segmnet merged
        if auto_proc:
            start_remote_job(pcfg, bat_file_all_non_m)

        # wait merging and segmnet merged
        if auto_proc:
            start_remote_job(pcfg, bat_file_all_m)

        print(f'merged_tiles_ids: {merged_tiles_ids}, all_non_tile_dataset_ids: {all_non_tile_dataset_ids}')

        last_dataset_idx = max(merged_tiles_ids + all_non_tile_dataset_ids)
        tgt_file = pcfg.datasets_path_proc + '%03d' % last_dataset_idx + r'\segmentation\cells\tr_cells_tmp.dat'
        wait_for_file(tgt_file)

        # End time report
    except Exception as e:
        print('Error:', e)
        print(traceback.format_exc(), flush=True)
        return proc_res

    nb_end_t = timer()
    print(f'processing run time: {(nb_end_t - nb_start_t) / 3600:.2f} h')

    # result:
    proc_res['status'] = True
    proc_res['last_dataset_idx'] = last_dataset_idx

    return proc_res


def seg_gui():
    # 0. load config
    proc_cfg = load_proc_cfg()

    # 1.1 "datasets indexes range" - 2 input fields for integers of first, last dataset idx
    first_idx_guess = proc_cfg.next_dataset_id

    ds_inf = read_info_file(proc_cfg.datasets_path + 'info.txt')
    if len(ds_inf):
        last_idx_guess = max(ds_inf.keys())
    else:
        last_idx_guess = first_idx_guess + 7

    dataset_id_first_ti = Text(value=f'{first_idx_guess}', description='Dataset range: first:',
                               layout=Layout(width='300px'))
    dataset_id_last_ti = Text(value=f'{last_idx_guess}', description='last:', layout=Layout(width='300px'))
    datasets_range_b = HBox([dataset_id_first_ti, dataset_id_last_ti])

    # 1.2 overlap percent, default 10%
    overlap_perc_ti = Text(value='10', description='Tiles overlap %:', layout=Layout(width='600px'))

    # 1.3 redo completed processing (currently segmentation is always executed)
    redo_proc_cb = Checkbox(value=False, description='Redo completed processing', layout=Layout(width='600px'))

    # 2. "dataset root path" - input field for string with default value set to `datasets_path`
    dataset_path_ts = Text(value=proc_cfg.datasets_path, description='Dataset root path:', layout=Layout(width='600px'))

    # 3. "dataset proc root path" - input field for string with default value set to `datasets_path_proc`
    dataset_proc_path_ts = Text(value=proc_cfg.datasets_path_proc, description='Dataset proc root path:',
                                layout=Layout(width='600px'))

    # 4. "VovoFollow root path" input field for string with default value set to `VF_root`
    VF_root_ts = Text(value=proc_cfg.VF_root, description='VivoFollow root path:', layout=Layout(width='600px'))

    # 5. "models path" - input field for string with default value set to `models_path`
    models_path_ts = Text(value=proc_cfg.models_path, description='Models path:', layout=Layout(width='600px'))

    # 6. "reference datasets path" - input field for string with default value set to `ref_ds_path`
    ref_ds_path_ts = Text(value=proc_cfg.ref_ds_path, description='Reference datasets path:',
                          layout=Layout(width='600px'))

    # 7. "CUDA processing devices" - input field for list of integers with default value set to `[0]`
    dev_ids_str = ', '.join([str(i) for i in proc_cfg.cuda_dev_ids])
    dev_ids_ts = Text(value=dev_ids_str, description='CUDA processing devices:', layout=Layout(width='600px'))

    # 8. "reference dataset ID" - input field for integer with default value set to `7`
    rds_id_ti = Text(value=f'{proc_cfg.rds_id}', description='Reference dataset ID:', layout=Layout(width='600px'))

    # 9. Button "Run" to start processing. which prints in the output field the values of parameters
    run_button = Button(description='Run', layout=Layout(width='600px'))

    # output
    out = Output()

    def on_run_button_clicked(b, proc_cfg: ProcConfig):
        proc_cfg.dataset_id_first_run = int(dataset_id_first_ti.value)
        proc_cfg.dataset_id_last_run = int(dataset_id_last_ti.value)
        proc_cfg.tile_overlap_fraction = int(overlap_perc_ti.value)/100
        proc_cfg.skip_completed = ~redo_proc_cb.value
        proc_cfg.next_dataset_id = int(dataset_id_last_ti.value) + 1  # tentative - to be updated dependin
        proc_cfg.datasets_path = dataset_path_ts.value
        proc_cfg.models_path = models_path_ts.value
        proc_cfg.ref_ds_path = ref_ds_path_ts.value
        proc_cfg.datasets_path_proc = dataset_proc_path_ts.value
        proc_cfg.VF_root = VF_root_ts.value
        proc_cfg.rds_id = int(rds_id_ti.value)
        proc_cfg.cuda_dev_ids = [int(i) for i in dev_ids_ts.value.replace(' ', '').replace('\t', '').split(',')]

        with out:
            print(dataclasses.asdict(proc_cfg))

            proc_res = run_segmentation(proc_cfg)

            res = proc_res['status']

            if res:
                print('Processing finished successfully')
            else:
                print('Processing failed. Please Contact support')

            last_dataset_idx = proc_res['last_dataset_idx']

            proc_cfg.next_dataset_id = last_dataset_idx + 1
        save_proc_cfg(proc_cfg)

    on_run_button_clicked_p = partial(on_run_button_clicked, proc_cfg=proc_cfg)
    run_button.on_click(on_run_button_clicked_p)

    # display
    controls_b = VBox(
        [
            datasets_range_b,
            overlap_perc_ti,
            redo_proc_cb,
            dataset_path_ts,
            dataset_proc_path_ts,
            VF_root_ts,
            models_path_ts,
            ref_ds_path_ts,
            dev_ids_ts,
            rds_id_ti,
            run_button,
            out])

    display(HTML('''
    <style>
        .widget-label { min-width: 200px !important; }
    </style>'''))

    display(controls_b)


