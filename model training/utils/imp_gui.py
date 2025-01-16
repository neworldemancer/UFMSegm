"""
   Copyright 2015-2024, University of Bern,
    Laboratory for High Energy Physics, Theodor Kocher Institute,
     and Data Science Lab
     M. Vladymyrov

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

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from .CZI_image import CZI_image
from PIL import Image
from . import imgio as iio

import os
import glob

import shutil

from IPython.display import display
from ipywidgets import Layout, HBox, VBox, Text, Button, Output, HTML

from dataclasses import dataclass, asdict
import json
import traceback


def minmax(a):
    return np.min(a), np.max(a)


# 1. CZI_image
def save_as_8bit_tifs(root_dir, ds_name, ci: CZI_image):
    # format : <ds_name>_t0xcxmx.tif

    n_c = ci.n_c
    n_tile = ci.n_tile
    n_t = ci.n_t

    tmpl_t = 't%0' + '%d' % len('%d' % n_t) + 'd'
    tmpl_ch = 'c%0' + '%d' % len('%d' % n_c) + 'd'
    tmpl_tl = 'm%0' + '%d' % len('%d' % n_tile) + 'd'

    path = os.path.join(root_dir, ds_name)
    os.makedirs(path, exist_ok=True)

    for ch in range(n_c):
        arr = [minmax(ci.get_frame(tile=tile, t=t, c=ch)) for t in range(n_t) for tile in range(n_tile)]
        arr = np.array(arr).T
        v_min = np.min(arr[0])
        v_max = np.max(arr[1])
        del arr

        print(ch, v_min, v_max)

        for t in tqdm(range(n_t)):
            for tile in range(n_tile):
                sfx = '_'
                sfx += tmpl_t % (t + 1)

                sfx += tmpl_ch % (ch + 1)

                sfx += tmpl_tl % (tile + 1)

                name = os.path.join(path, ds_name + sfx + '.tif')

                im = ci.get_frame(tile=tile, t=t, c=ch)
                im_normed = (im - v_min) / (v_max - v_min)
                im_8b = (255 * im_normed.clip(0, 1)).astype(np.uint8)

                iio.save_image(im_8b, name)


def dataset_from_czi(root_dir, image_file_name):
    """
    Generates images from tiles
    """
    # read
    print('reading')
    ci = CZI_image(image_file_name)
    ds_name = os.path.splitext(os.path.basename(image_file_name))[0]
    save_as_8bit_tifs(root_dir, ds_name, ci)
    del ci


# 2. Splitting tiles:
def create_ds_dir(idx, ds_name, datasets_path):
    path = os.path.join(datasets_path, '%03d'%idx)
    if os.path.exists(path):
        return False
    else:
        path = os.path.join(path, ds_name)
        os.makedirs(path, exist_ok=False)
        return True


def make_record_info(ds_inf_file_path, idx, ds_name, datasets_path, tile=None):
    ttl = ds_name + f', tile {tile}' if tile is not None else ''

    ds_inf = read_info_file(ds_inf_file_path)
    save_ds_inf(ds_inf_file_path, ds_inf, [ttl])
    ds_inf = read_info_file(ds_inf_file_path)

    assert max(ds_inf.keys()) == idx, f'max({ds_inf.keys()}) ={max(ds_inf.keys())} != {idx}'


def ds_info(ds_path):
    list_files = sorted([n for n in os.listdir(ds_path) if '.tif' in n])

    file_names = [fn.replace('.tif', '') for fn in list_files if ('t' in fn and '_' in fn)]  # only formatted, timelapse

    list_sfx = [fn.split('_')[-1] for fn in file_names]

    name_tmpls = ['_'.join(fn.split('_')[:-1]) for fn in file_names if '_' in fn]

    if len(name_tmpls) == 0:
        raise FileNotFoundError('files with name "<xxx>_<xx>t%d<xx>.tif not found"')

    name_tmpl = name_tmpls[0]

    last_sfx = list_sfx[-1]
    has_ch = 'c' in last_sfx
    has_tl = 'm' in last_sfx
    has_ps = 's' in last_sfx  # positions

    if has_ps:
        last_sfx_trunc_s = last_sfx.replace('s', '')

        n_ps_s, res_sfx = last_sfx_trunc_s.split('t')
        n_ps = int(n_ps_s)

        if has_ch:
            n_t_s, ch_tl_s = res_sfx.split('c')
            n_t = int(n_t_s)

            if has_tl:
                n_ch, n_tl = [int(s) for s in ch_tl_s.split('m')]
            else:
                n_ch = int(ch_tl_s)
                n_tl = 1
        else:
            if has_tl:
                n_t, n_tl = [int(s) for s in res_sfx.split('m')]
            else:
                n_t = int(res_sfx)
                n_tl = 1
            n_ch = 1
    else:
        n_ps = 1
        last_sfx_trunc_t = last_sfx.replace('t', '')
        if has_ch:
            n_t_s, ch_tl_s = last_sfx_trunc_t.split('c')
            n_t = int(n_t_s)

            if has_tl:
                n_ch, n_tl = [int(s) for s in ch_tl_s.split('m')]
            else:
                n_ch = int(ch_tl_s)
                n_tl = 1
        else:
            if has_tl:
                n_t, n_tl = [int(s) for s in last_sfx_trunc_t.split('m')]
            else:
                n_t = int(last_sfx_trunc_t)
                n_tl = 1
            n_ch = 1

    tmpl_ps = 's%0' + '%d' % len('%d' % n_tl) + 'd'
    tmpl_t = 't%0' + '%d' % len('%d' % n_t) + 'd'
    tmpl_ch = 'c%0' + '%d' % len('%d' % n_ch) + 'd'
    tmpl_tl = 'm%0' + '%d' % len('%d' % n_tl) + 'd'

    return {'n_ps': n_ps,
            'n_t': n_t,
            'n_ch': n_ch,
            'n_tl': n_tl,
            'has_ps': has_ps,
            'has_ch': has_ch,
            'has_tl': has_tl,
            'tmpl_ps': tmpl_ps,
            'tmpl_t': tmpl_t,
            'tmpl_ch': tmpl_ch,
            'tmpl_tl': tmpl_tl,
            'name_tmpl': name_tmpl
            }


def get_file_name(path, inf_dict, ps, t, ch, tile):
    sfx = '_'

    if inf_dict['has_ps']:
        sfx += inf_dict['tmpl_ps'] % (ps + 1)
    sfx += inf_dict['tmpl_t'] % (t + 1)
    if inf_dict['has_ch']:
        sfx += inf_dict['tmpl_ch'] % (ch + 1)
    if inf_dict['has_tl']:
        sfx += inf_dict['tmpl_tl'] % (tile + 1)

    name = inf_dict['name_tmpl'] + sfx + '.tif'
    fn = os.path.join(path, name)
    return fn


def create_segmentation_datasets(datasets_path, datasets_names, start_ds_idx, ds_inf_file_path,
                                 time_subsample_min_t=210):  # if n_t >time_subsample_min_t - take every second frame
    for ds_names in datasets_names:
        if isinstance(ds_names, str):
            ds_names = [(0, ds_names, 0)]

        # 1. get info for all
        copy_struct = []  # 1 element per ds: (ds_dir_name, info, n_before, n_after)

        all_nt = []
        for item in ds_names:
            n_copy_before, ds_path, n_copy_after = item
            inf = ds_info(ds_path)
            copy_struct.append((ds_path, inf, n_copy_before, n_copy_after))
            all_nt.append(inf['n_t'])

        subsample_fact = 2 if max(all_nt) > time_subsample_min_t else 1
        # print(subsample_fact, copy_struct, '\n')
        # continue

        # check validity: all mush have same format
        for key in ['n_ps', 'n_ch', 'n_tl', 'has_ps', 'has_ch', 'has_tl']:
            el0 = copy_struct[0][1][key]
            for struct in copy_struct[1:]:
                assert (struct[1][key] == el0)

        n_t_out = 0  # num output timeframes
        for struct in copy_struct:
            n_t_out += struct[2] + struct[3] + (struct[1]['n_t'] // subsample_fact)

        inf0 = copy_struct[0][1]
        n_ch = inf0['n_ch']

        n_ps_i = inf0['n_ps']
        n_tl_i = inf0['n_tl']

        n_tl = n_ps_i * n_tl_i

        has_ch = inf0['has_ch']
        has_tl = inf0['has_ps'] or inf0['has_tl']

        out_ds_name_general = '_'.join([struct[1]['name_tmpl'] for struct in copy_struct])

        tmpl_t = 't%0' + '%d' % len('%d' % n_t_out) + 'd'
        tmpl_ch = 'c%0' + '%d' % len('%d' % n_ch) + 'd'
        tmpl_tl = 'm%0' + '%d' % len('%d' % 1) + 'd'

        oinf = {'n_ps': 1,
                'n_t': n_t_out,
                'n_ch': n_ch,
                'n_tl': 1,
                'has_ps': False,
                'has_ch': has_ch,
                'has_tl': False,
                'tmpl_t': tmpl_t,
                'tmpl_ch': tmpl_ch,
                'tmpl_tl': tmpl_tl,
                'name_tmpl': out_ds_name_general
                }

        # create dirs and fill info file
        tile_to_idx = {}
        tile_ds_name = {}

        creation_ok = True
        for tl in range(n_tl):
            idx = start_ds_idx + tl
            tile_to_idx[tl] = idx

            out_ds_name = out_ds_name_general + ('_tile%d' % (tl + 1) if has_tl else '')
            tile_ds_name[tl] = out_ds_name

            if not create_ds_dir(idx, out_ds_name, datasets_path):
                print('dataset with idx', idx, 'already exists. please check manually. Aborting.')
                creation_ok = False
                break
        if not creation_ok:
            break

        start_ds_idx += n_tl

        for tl in range(n_tl):
            idx = tile_to_idx[tl]
            out_ds_name = tile_ds_name[tl]
            make_record_info(ds_inf_file_path, idx, out_ds_name, datasets_path, (tl + 1) if has_tl else None)

            ods_path = os.path.join(datasets_path, '%03d' % idx)
            with open(os.path.join(ods_path, 'info.txt'), 'wt') as f:
                f.write(out_ds_name)

        for ps_i in range(n_ps_i):
            for tl_i in range(n_tl_i):
                tl = ps_i * n_tl_i + tl_i

                idx = tile_to_idx[tl]
                out_ds_name = tile_ds_name[tl]
                oinf['name_tmpl'] = out_ds_name

                ods_path = os.path.join(datasets_path, '%03d' % idx, out_ds_name)

                block_boundaries = []
                for ch in range(n_ch):
                    t_o = 0
                    for struct in copy_struct:
                        in_path, inf, copy_before, copy_after = struct
                        n_t_i = inf['n_t']
                        for i in range(copy_before):
                            t_i = 0
                            i_file = get_file_name(in_path, inf, ps_i, t_i, ch, tl_i)
                            o_file = get_file_name(ods_path, oinf, -1, t_o, ch, 0)
                            shutil.copy(i_file, o_file)
                            t_o += 1

                        for t_i in range(n_t_i // subsample_fact):
                            i_file = get_file_name(in_path, inf, ps_i, t_i * subsample_fact, ch, tl_i)
                            o_file = get_file_name(ods_path, oinf, -1, t_o, ch, 0)
                            shutil.move(i_file, o_file)
                            t_o += 1

                        for i in range(copy_after):
                            i_file = get_file_name(ods_path, oinf, ps_i, t_o - 1, ch, tl_i)
                            o_file = get_file_name(ods_path, oinf, -1, t_o, ch, 0)
                            shutil.copy(i_file, o_file)
                            t_o += 1

                        if ch == 0:
                            begin = 0 if len(block_boundaries) == 0 else block_boundaries[-1][1]
                            end = t_o
                            block_boundaries.append([begin, end])

                block_info_path = os.path.join(datasets_path, '%03d' % idx, 'block_info.txt')
                with open(block_info_path, 'wt') as f:
                    txt = '|'.join([' '.join([str(bi) for bi in b]) for b in block_boundaries])
                    f.write(txt)

        for item in ds_names:
            n_copy_before, ds_path, n_copy_after = item
            # shutil.rmtree(ds_path)

    return start_ds_idx


# 3. Ipywidgets interface

# Files path input string "CZI Datasets path"
# Directory path input string "Segmentation datasets path"
# File path with datasets IDs list - text human-readable file with
#   "dataset id - names" pairs like " 85 - Untreated_2024.06.18"
# Button "Process"
# Output - text box with the progress

@dataclass
class DataImportConfig:
    raw_ds_path: str = os.path.abspath('../../datasets_raw')
    seg_ds_path: str = os.path.abspath('../../datasets_seg')
    ds_inf_path: str = os.path.abspath('../../datasets_seg/info.txt')


# methods for loading and storing the config to file
_cfg_filename = 'import_cfg.json'


def load_import_cfg(cfg_path=None):
    cfg_path = cfg_path or os.path.join(os.path.abspath(os.path.curdir), _cfg_filename)
    if os.path.exists(cfg_path):
        with open(cfg_path, 'rt') as f:
            cfg = DataImportConfig(**json.load(f))
    else:
        cfg = DataImportConfig()
    return cfg


def save_import_cfg(cfg, cfg_path=None):
    cfg_path = cfg_path or os.path.join(os.path.abspath(os.path.curdir), _cfg_filename)
    with open(cfg_path, 'wt') as f:
        # human-readable, 4 spaces indentation
        json.dump(asdict(cfg), f, indent=4)


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


def save_with_widgets():
    """
    Saving widgets state for history hack, reused from
    https://stackoverflow.com/questions/59123005/how-to-save-state-of-ipython-widgets-in-jupyter-notebook-using-python-code
    """
    code = '<script>Jupyter.menubar.actions._actions["widgets:save-with-widgets"].handler()</script>'
    display(HTML(code))


def import_gui():
    # 0. Load config
    import_cfg = load_import_cfg()

    # 1. Files path input string "CZI Datasets path"
    raw_ds_path = Text(value=import_cfg.raw_ds_path, description='CZI Datasets path:', disabled=False,
                       layout=Layout(width='600px'))

    # 2. Directory path input string "Segmentation datasets path"
    seg_ds_path = Text(value=import_cfg.seg_ds_path, description='Segmentation datasets path:', disabled=False,
                       layout=Layout(width='600px'))

    # 3. File path with datasets IDs list - text human readable file with dataset id - names pairs like " 85 - Untreated_2024.06.18"
    ds_inf_path = Text(value=import_cfg.ds_inf_path, description='Datasets info path:', disabled=False,
                       layout=Layout(width='600px'))

    # 4. Button "Process"
    process_btn = Button(description='Process', layout=Layout(width='600px'))

    # 5. Output - text box with the progress
    out = Output()

    def on_process_click(b):
        with out:
            try:
                print('Processing...')
                # save config
                import_cfg.raw_ds_path = raw_ds_path.value
                import_cfg.seg_ds_path = seg_ds_path.value
                import_cfg.ds_inf_path = ds_inf_path.value

                save_import_cfg(import_cfg)
                # print('Saved cfg')

                ds_inf = read_info_file(import_cfg.ds_inf_path)

                # print('Read info')
                start_ds_idx = (max(ds_inf.keys()) + 1) if len(ds_inf) else 0
                imported_datasets_range_first = start_ds_idx
                # process
                all_ds = list(glob.glob(import_cfg.raw_ds_path + '\\**\\' + '*.czi', recursive=True))
                for name in all_ds:
                    print(name)
                for fname in all_ds:
                    dataset_from_czi(import_cfg.raw_ds_path, fname)

                datasets_names = []
                path = import_cfg.raw_ds_path
                path = os.path.abspath(path)
                for p2 in sorted(os.listdir(path)):
                    path3 = os.path.join(path, p2)
                    if not os.path.isdir(path3):
                        continue
                    datasets_names.append([[0, path3, 0]])

                print(f'datasets_names = {datasets_names}')
                start_ds_idx = create_segmentation_datasets(import_cfg.seg_ds_path,
                                                            datasets_names,
                                                            start_ds_idx,
                                                            import_cfg.ds_inf_path
                                                            )

                imported_datasets_range_last = start_ds_idx - 1
                print(f'Dataset range: first {imported_datasets_range_first}')
                print(f'Dataset range: last {imported_datasets_range_last}')

                print('Done')
                save_with_widgets()

            except Exception as e:
                print('Error:', e)
                trace_str = traceback.format_exc()
                print(trace_str, flush=True)

    process_btn.on_click(on_process_click)

    controls_b = VBox([
        raw_ds_path,
        seg_ds_path,
        ds_inf_path,
        process_btn,
        out
    ])

    display(HTML('''
    <style>
        .widget-label { min-width: 200px !important; }
    </style>'''))

    display(controls_b)
