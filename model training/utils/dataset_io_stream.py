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

import time

import numpy as np
import tensorflow as tf

from . import imgio as iio
import pickle


class dataset_io_stream(object):
    """Data reading and preparation class. Reads data from image files or TFRecord
    performs preprocessing in queues and produces crops for training:

    Attributes:
    """

    def __init__(self, name, inputs_file_names=None, outputs_file_names=None, pkl_file_name=None,
                 normalize_inputs_outputs=None,  # ((input0_false, input1_true), (output0_false))
                 normalization_percentile_range=None,
                 binary_inputs_outputs=None,
                 stack_depths=(1,),
                 tile_size=512, tile_depth=35, tile_depth_overlap=4,
                 strm_stack_shape=(5, 256, 256), strm_z_stride=1,
                 image_crop_regions=(((0, -1), (0, -1), (0, -1)),),
                 pre_crop_xy=(((20, -20), (20, -20)),),
                 dtype=np.float32, preproc_devs=('/gpu:0',), graph=None,
                 ds_name=None, shuffle_output=True,
                 minibatch_size=0, queue_len=0, num_proc_threads=8, augment=True, streaming=True):
        """
        Args:
            strm_stack_shape: shape of inputs for training.
            image_crop_regions: tuple describing image cropping range, (x_min,x_max), (y_min,y_max), (z_min,z_max)
            pre_crop_xy: crop frame to apply to loaded image inputs before fine selection, normalization etc
            dtype: type of all the tensors
            preproc_devs: list of devices that will be running preprocessing
            graph: graph to which all elements will be added. If `None` new one will be created
            ds_name: if not none image data will be saved in the
                TFRecord format (regular operation impossible, just use to convert data)
            shuffle_output: if set to True, additional shuffling will
                be applied to each batch. otherwise - just obtained by random order of cropping etc
            num_proc_threads: number of threads used for distorted image generation
            
            inputs_file_names (list(str,)): dirs with image inputs or TFRecords file names

            augment (bool): apply augmentation
        
        Returns:
            dataset_io_stream object, with TF objects in namespace `name`.
            
        Note:
            Two operation modes are available:
                1. reads data and outputs from `inputs_file_names` and `outputs_file_names` images.
                    Then creates tiles of size `tile_size`*`tile_size`. Then save to `save_tf_name`.
                    In this case no actual preprocessing is done. Use to prepare dataset only.
                    All data is assumed GS and if it's not - then the first channel is used.
                2. reads data and outputs from a single `inputs_file_names` (should have a ".tfr" extension)
                    and sets up all the preprocessing queues. (This is normal operation).
                    Example:
                    
                        drpp = dataset_io_stream(name='preproc', inputs_file_names=('/dev/all_b_3.tfr',),
                          tile_depth = 42, preproc_devs=('/gpu:0','/gpu:1'))

                        with tf.Session(graph=drpp.GetGraph()) as sess:
                            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                            sess.run(init_op)
                            
                            coord = drpp.StartQueues(sess)
                            nIt = 301
                            for batch_index in range(nIt):
                                # Slow:
                                #st_d,lbl_d = drpp.GetSingleTrainingItem()
                                #st, lbl = sess.run([st_d,lbl_d])
                                
                                # Normal:
                                dbs, dbl = drpp.GetTrainingBatch()
                                st, lbl = sess.run([dbs, dbl])
                            
                            coord.request_stop()    
                            drpp.StopQueues()
                            sess.close()
                    
            If `shuffle_output` is set to True (default, much faster), the batch is generated as a whole.
                Data can be obtained then as `st_d,lbl_d = drpp_obj.GetTrainingBatch()`.
                
            If `shuffle_output` is set to False, the batch is generated piece by piece. Data can be obtained then as
                `st_d,lbl_d = drpp_obj.GetSingleTrainingItem()`.
                
            
        """
        self.tile_size = tile_size
        self.strm_stack_shape = strm_stack_shape
        self.strm_z_stride = strm_z_stride
        self.image_crop_regions = image_crop_regions
        self.augment = augment

        n_inp = len(inputs_file_names) if inputs_file_names is not None else 1
        p_crop_xy = [pre_crop_xy[0]] * n_inp if len(pre_crop_xy) == 1 else pre_crop_xy
        self.pre_crop_xy = p_crop_xy

        self.stack_depths = stack_depths
        self.tile_depth = tile_depth
        self.tile_depth_overlap = tile_depth_overlap

        self.normalize_inputs_outputs = normalize_inputs_outputs
        self.normalization_percentile_range = normalization_percentile_range
        self.binary_inputs_outputs = binary_inputs_outputs

        self.file_name = ''

        self.dtype = dtype
        if dtype != np.float32 and dtype != np.float16:
            raise ValueError('type %s not supported' % (str(dtype)))

        self.tf_dtype = tf.float32 if dtype == np.float32 else tf.float16
        self.n_run_dev = len(preproc_devs)
        self.preproc_devs = preproc_devs

        self.graph = tf.Graph() if graph is None else graph

        self.shuffle_output = shuffle_output
        self.batch_size = 0
        self.minibatch_size = minibatch_size
        self.queue_len = queue_len

        self.num_proc_threads = num_proc_threads

        self.filename_queue = None
        self.tfr_images = self.tfr_labels = None
        self.inputs = []  # self.inputs[sample_idx][input_type_idx]
        self.outputs = []

        self.inputs_mean_std = []
        self.outputs_mean_std = []
        self.input_tiles_mean_std = []
        self.output_tiles_mean_std = []

        self.inputs_tiles_b = None
        self.outputs_tiles_b = None
        self.inputs_tiles = []
        self.outputs_tiles = []

        self.n_tiles = 0
        self.n_inputs = 0
        self.n_outputs = 0

        self.init_operations = []
        self.minibatch = None

        self.rnd_crop_ofs_it = None
        self.rnd_dist_prs_it = None

        self.tst_img = None

        self.last_mb_idx = 0

        with self.graph.as_default():
            with tf.variable_scope(name):
                self.read_data(inputs_file_names, outputs_file_names, pkl_file_name)
                if ds_name is not None:
                    self.save_dataset(ds_name)

                if pkl_file_name is not None and streaming:
                    self.init_streaming()

    def get_graph(self):
        return self.graph

    def read_data(self, stacks_file_names, labels_file_names, pkl_file_name):
        if stacks_file_names is not None:
            self.read_data_image(stacks_file_names, labels_file_names)
            self.make_tiles()
        else:
            self.read_dataset(pkl_file_name)

    def read_data_image(self, inputs_file_names, outputs_file_names):
        """Reads data and outputs from *inputs_file_names* and *outputs_file_names*.
            Args:
                inputs_file_names (list(list(string))): list of images dataset pieces,
                    n_input template strings per each piece, for each input.
                outputs_file_names (list(list(string))): list of images dataset pieces,
                    n_input template strings per each piece, for each output.
        """

        io_files = zip(inputs_file_names, outputs_file_names)

        inputs = []
        outputs = []

        clr_ln = '\033[1K'

        for idx, ins_outs in enumerate(io_files):
            cr = self.image_crop_regions[idx]
            start_idx = cr[2][0]
            num_img = (self.stack_depths[idx] - start_idx) if (cr[2][1] == -1) else (cr[2][1] - start_idx)
            print(clr_ln + 'reading stack %d...' % idx, end='\r')

            if num_img == 0:
                continue

            self.inputs_mean_std.append([])
            self.outputs_mean_std.append([])
            inputs.append([])
            outputs.append([])

            if idx == 0:
                self.n_inputs = len(ins_outs[0])
                self.n_outputs = len(ins_outs[1])

            for in_out in [0, 1]:
                for inp, norm, is_bin in zip(ins_outs[in_out],
                                             self.normalize_inputs_outputs[in_out],
                                             self.binary_inputs_outputs[in_out]):
                    stack = iio.read_image_stack(inp, num_img, start_idx=start_idx)
                    pre_crop = self.pre_crop_xy[idx]

                    stack = stack[:, pre_crop[1][0]:pre_crop[1][1], pre_crop[0][0]:pre_crop[0][1]]

                    if len(stack.shape) == 4:  # gs
                        if stack.shape[3] == 4:
                            stack = (stack[:, :, :, 3] / 255 * stack[:, :, :, 0]).astype(stack.dtype)
                        else:
                            stack = stack[:, :, :, 0]

                    if is_bin:
                        stack = (stack > 64).astype(stack.dtype)  # everything > 64 will be 1, else 0
                        m, s = 0, 1
                    else:
                        if norm:
                            if self.normalization_percentile_range is None:
                                norm_stack = stack
                            else:
                                norm_stack = stack.flatten()
                                p1, p2 = self.normalization_percentile_range
                                r1, r2 = np.percentile(norm_stack, p1), np.percentile(norm_stack, p2)
                                x_idx = (norm_stack > r1) * (norm_stack < r2)
                                norm_stack = norm_stack[x_idx]

                            m, s = np.mean(norm_stack), np.std(norm_stack)
                            print('avg:', m, 'std:', s)
                        else:
                            m, s = 0, 255

                    m_std = [self.inputs_mean_std, self.outputs_mean_std][in_out]
                    m_std[idx].append([m, s])

                    stack = stack[:,
                                  cr[1][0]:cr[1][1],
                                  cr[0][0]:cr[0][1],
                                  ...]

                    stacks = [inputs, outputs][in_out]
                    stacks[idx].append(stack)

        self.inputs = inputs
        self.outputs = outputs

    def make_tiles(self):
        """
        Creates tiles for all inputs read from images. Stack sizes doesn't have to be same
        """
        inp_tiles_b = []
        out_tiles_b = []
        self.input_tiles_mean_std = []
        self.output_tiles_mean_std = []

        print('preparing tiles...')
        for sample_idx, inp_out in enumerate(zip(self.inputs, self.outputs)):
            stk0 = inp_out[0][0]  # first input

            if stk0.size == 0:
                continue

            sh = stk0.shape
            w = sh[2]
            h = sh[1]
            d = sh[0]

            tile_ofs = self.tile_size // 2
            tile_ofs_z = (self.tile_depth - self.tile_depth_overlap)
            n_tile_x = (w + tile_ofs - 1 - tile_ofs) // tile_ofs  # - tile_ofs coz actual tiles are 2*tile_ofs
            n_tile_y = (h + tile_ofs - 1 - tile_ofs) // tile_ofs
            n_tile_z = (d + tile_ofs_z - 1 - self.tile_depth_overlap) // tile_ofs_z

            # print (n_tile_x, n_tile_y, n_tile_z)

            for z_id in range(n_tile_z):
                z_ofs = min(z_id * tile_ofs_z, d - self.tile_depth)
                for y_id in range(n_tile_y):
                    y_ofs = min(y_id * tile_ofs, h - self.tile_size)
                    for x_id in range(n_tile_x):
                        x_ofs = min(x_id * tile_ofs, w - self.tile_size)
                        # print(x_ofs, y_ofs, z_ofs)

                        inp_tiles_b.append([])
                        out_tiles_b.append([])
                        self.input_tiles_mean_std.append([])
                        self.output_tiles_mean_std.append([])

                        io_tiles = [inp_tiles_b[-1], out_tiles_b[-1]]
                        io_mean_std = [self.input_tiles_mean_std[-1], self.output_tiles_mean_std[-1]]
                        sample_io_mean_std = [self.inputs_mean_std[sample_idx], self.outputs_mean_std[sample_idx]]

                        for io_idx in [0, 1]:
                            for ch_idx, stack in enumerate(inp_out[io_idx]):  # loop all types of inputs/outputs
                                tile = stack[z_ofs:z_ofs + self.tile_depth,
                                             y_ofs:y_ofs + self.tile_size,
                                             x_ofs:x_ofs + self.tile_size]

                                io_tiles[io_idx].append(tile)
                                sample_m_s = sample_io_mean_std[io_idx][ch_idx]
                                io_mean_std[io_idx].append(sample_m_s)

        self.inputs_tiles_b = np.asarray(inp_tiles_b, dtype=np.uint8)
        self.outputs_tiles_b = np.asarray(out_tiles_b, dtype=np.uint8)
        self.input_tiles_mean_std = np.asarray(self.input_tiles_mean_std, dtype=np.float32)
        self.output_tiles_mean_std = np.asarray(self.output_tiles_mean_std, dtype=np.float32)

        self.n_tiles = self.inputs_tiles_b.shape[0]

        if self.shuffle_output:
            shuffled_idx = np.random.permutation(self.n_tiles)
            self.inputs_tiles_b = self.inputs_tiles_b[shuffled_idx]
            self.outputs_tiles_b = self.outputs_tiles_b[shuffled_idx]
            self.input_tiles_mean_std = self.input_tiles_mean_std[shuffled_idx]
            self.output_tiles_mean_std = self.output_tiles_mean_std[shuffled_idx]

    def save_dataset(self, f_name):
        clr_ln = '\033[1K'
        # self.n_tiles=1
        self.file_name = f_name + "_%d_st.pkl" % self.n_tiles
        print('Saving dataset %s' % self.file_name)

        ds_dict = {'n_in': self.n_inputs,
                   'n_out': self.n_outputs,
                   'ins': self.inputs_tiles_b,
                   'outs': self.outputs_tiles_b,
                   'ins_norm': self.input_tiles_mean_std,
                   'outs_norm': self.output_tiles_mean_std,
                   'ins_is_mask': self.binary_inputs_outputs[0],
                   'outs_is_mask': self.binary_inputs_outputs[1]
                   }

        with open(self.file_name, 'wb') as f:
            pickle.dump(ds_dict, f, pickle.HIGHEST_PROTOCOL)

        print(clr_ln + 'Saving dataset done!')

    def read_dataset(self, pkl_file_name):
        with open(pkl_file_name, 'rb') as f:
            ds_dict = pickle.load(f)

        self.n_inputs = ds_dict['n_in']
        self.n_outputs = ds_dict['n_out']
        self.inputs_tiles_b = ds_dict['ins']
        self.outputs_tiles_b = ds_dict['outs']
        self.input_tiles_mean_std = ds_dict['ins_norm']
        self.output_tiles_mean_std = ds_dict['outs_norm']
        self.binary_inputs_outputs = []
        self.binary_inputs_outputs.append(ds_dict['ins_is_mask'])
        self.binary_inputs_outputs.append(ds_dict['outs_is_mask'])

        self.inputs_tiles = []
        self.outputs_tiles = []

        inps_outs_raw = [self.inputs_tiles_b, self.outputs_tiles_b]
        inps_outs = [self.inputs_tiles, self.outputs_tiles]
        n_inp_out = [self.n_inputs, self.n_outputs]
        inps_outs_norm = [self.input_tiles_mean_std, self.output_tiles_mean_std]

        for io_idx in [0, 1]:
            n = n_inp_out[io_idx]
            tiles_r = inps_outs_raw[io_idx]
            tiles = inps_outs[io_idx]
            masks = self.binary_inputs_outputs[io_idx]
            norms = inps_outs_norm[io_idx]

            for inp_idx in range(n):
                inp_i = tiles_r[:, inp_idx]
                if masks[inp_idx]:
                    inp_i = inp_i.astype(np.int32)
                else:
                    m, s = norms[:, inp_idx, 0], norms[:, inp_idx, 1]
                    m, s = m.reshape((-1, 1, 1, 1)), s.reshape((-1, 1, 1, 1))
                    inp_i = (inp_i.astype(np.float32) - m)/s

                tiles.append(inp_i)

        sh = self.inputs_tiles[0].shape
        self.batch_size = sh[0]
        self.tile_depth = sh[1]
        self.tile_size = sh[2]

        return

    def get_minibatch(self):
        mb_sz = self.minibatch_size

        n_samples = self.batch_size
        ofs_range = [self.tile_depth - (self.strm_stack_shape[0] - 1) * self.strm_z_stride,  # - 1 + 1
                     self.tile_size - self.strm_stack_shape[1] + 1,
                     self.tile_size - self.strm_stack_shape[2] + 1
                     ]

        mb_inps = []
        mb_outs = []

        ofss = []
        for i in range(mb_sz):
            sample = np.random.choice(n_samples) if self.shuffle_output else self.last_mb_idx
            self.last_mb_idx += 1
            self.last_mb_idx = self.last_mb_idx % self.batch_size

            ofss.append([sample,
                         np.random.choice(ofs_range[0]),
                         np.random.choice(ofs_range[1]),
                         np.random.choice(ofs_range[2])])

        for io_idx in [0, 1]:
            mb_ios = [mb_inps, mb_outs]
            tiles = [self.inputs_tiles, self.outputs_tiles]

            for inp in tiles[io_idx]:
                mb = []
                for ofs in ofss:
                    inp_crop = inp[ofs[0],
                                   ofs[1]:
                                   ofs[1] + (self.strm_stack_shape[0] - 1) * self.strm_z_stride + 1:
                                   self.strm_z_stride,
                                   ofs[2]:ofs[2] + self.strm_stack_shape[1],
                                   ofs[3]:ofs[3] + self.strm_stack_shape[2],
                                   ]
                    mb.append(inp_crop)

                mb = np.stack(mb, axis=0)
                mb_ios[io_idx].append(mb)

        mb = mb_inps + mb_outs
        return mb

    def init_streaming(self):
        return self.init_streaming_aug() if self.augment else self.init_streaming_no_aug()

    def init_streaming_aug_bk(self):
        with self.graph.as_default():
            with tf.variable_scope('raw_queue'):
                all_io_phs = []
                is_binary = self.binary_inputs_outputs[0] + self.binary_inputs_outputs[1]

                train_dataset_init_feed_dict = {}

                with tf.device('/cpu:0'):
                    for tiles in self.inputs_tiles + self.outputs_tiles:
                        tiles_ph = tf.placeholder(dtype=tiles.dtype, shape=tiles.shape)
                        all_io_phs.append(tiles_ph)
                        train_dataset_init_feed_dict[tiles_ph] = tiles

                all_io_phs = tuple(all_io_phs)

                train_dataset = tf.data.Dataset.from_tensor_slices(all_io_phs).repeat()
                if self.shuffle_output:
                    train_dataset = train_dataset.shuffle(max(self.batch_size, self.queue_len*self.minibatch_size))

            # random transformation params
            dummy = np.zeros(shape=10)
            dd = tf.data.Dataset.from_tensor_slices(dummy)

            with tf.variable_scope('offset_queue'):
                # offset
                def rnd_ofs_map(_):
                    max_tile_ofs = np.asarray(
                                                [self.tile_depth - (self.strm_stack_shape[0] - 1)
                                                    * self.strm_z_stride,  # - 1 + 1
                                                 self.tile_size - self.strm_stack_shape[1] + 1,
                                                 self.tile_size - self.strm_stack_shape[2] + 1
                                                ]
                                             )  # self.strm_stack_shape

                    # np.asarray([self.tile_depth, self.tile_size, self.tile_size]) \
                    #                                    - np.asarray(self.strm_stack_shape) + 1

                    rnd_ofs_x = tf.random_uniform([1], minval=0, maxval=max_tile_ofs[2], dtype=tf.int32, name="ofs_X")
                    rnd_ofs_y = tf.random_uniform([1], minval=0, maxval=max_tile_ofs[1], dtype=tf.int32, name="ofs_Y")
                    rnd_ofs_z = tf.random_uniform([1], minval=0, maxval=max_tile_ofs[0], dtype=tf.int32, name="ofs_Z")
                    rnd_flp_x = tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32, name="flp_X")
                    rnd_flp_y = tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32, name="flp_Y")

                    rnd_ofs = tf.concat([rnd_ofs_x, rnd_ofs_y, rnd_ofs_z, rnd_flp_x, rnd_flp_y], axis=0, name='ofs_flp')
                    return rnd_ofs

                ofs_dataset = dd.repeat().map(rnd_ofs_map, num_parallel_calls=1).prefetch(5)
                ofs_iterator = tf.data.Iterator.from_structure(output_types=ofs_dataset.output_types,
                                                               output_shapes=ofs_dataset.output_shapes)
                ofs_iterator_init = ofs_iterator.make_initializer(ofs_dataset)
                self.rnd_crop_ofs_it = ofs_iterator
                self.init_operations.append((ofs_iterator_init, None))

            with tf.variable_scope('distortion_queue'):
                # distortion
                def rnd_dist_map(_):
                    rnd_rot = tf.random_uniform([1], minval=0, maxval=2 * np.pi, dtype=tf.float32, name="rot")
                    rnd_a0 = tf.random_uniform([1], minval=0.8, maxval=1.2, dtype=tf.float32, name="A0")
                    rnd_a1 = tf.random_uniform([1], minval=-0.2, maxval=0.2, dtype=tf.float32, name="A1")
                    rnd_b0 = tf.random_uniform([1], minval=-0.2, maxval=0.2, dtype=tf.float32, name="B0")
                    rnd_b1 = tf.random_uniform([1], minval=0.8, maxval=1.2, dtype=tf.float32, name="B1")
                    rnd_c1 = tf.random_uniform([1], minval=-0.0001, maxval=0.0001, dtype=tf.float32, name="C1")
                    rnd_c2 = tf.random_uniform([1], minval=-0.0001, maxval=0.0001, dtype=tf.float32, name="C2")
                    rnd_dist = tf.concat([rnd_rot, rnd_a0, rnd_a1, rnd_b0, rnd_b1, rnd_c1, rnd_c2], axis=0, name='dist')
                    return rnd_dist

                dist_dataset = dd.repeat().map(rnd_dist_map, num_parallel_calls=1).prefetch(5)

                dist_iterator = tf.data.Iterator.from_structure(output_types=dist_dataset.output_types,
                                                                output_shapes=dist_dataset.output_shapes)
                dist_iterator_init = dist_iterator.make_initializer(dist_dataset)
                self.rnd_dist_prs_it = dist_iterator
                self.init_operations.append((dist_iterator_init, None))

                self.tst_img = tf.ones(name='test_img', dtype=tf.float32, shape=(1, self.tile_size, self.tile_size))

            with tf.variable_scope('transformed_queue'):
                # get suitable transformation
                def preproc(*ims):
                    with tf.device(np.random.choice(self.preproc_devs)):
                        ofs_pars = self.get_offset_and_distortion(20, 3)
                        dist_pars = ofs_pars[3]
                        crop_pars = ofs_pars[4]
                        # itr_ofs = ofs_pars[1]
                        # itr_dist = ofs_pars[0]

                        ims = list(ims)
                        ims_dist = self.get_distorted_ims(ims, is_binary, dist_pars, crop_pars, self.strm_z_stride)

                        # print(ofs_pars)
                        return tuple(ims_dist)

                train_dataset = train_dataset.map(preproc, num_parallel_calls=min(self.queue_len,
                                                                                  self.num_proc_threads))
                train_dataset = train_dataset.batch(self.minibatch_size)
                if self.queue_len > 0:
                    # train_dataset = train_dataset.prefetch(self.queue_len)
                    train_dataset.apply(tf.data.experimental.prefetch_to_device(self.preproc_devs[0]))

                mb_iterator = tf.data.Iterator.from_structure(output_types=train_dataset.output_types,
                                                              output_shapes=train_dataset.output_shapes)
                mb_iterator_init_op = mb_iterator.make_initializer(train_dataset)
                self.init_operations.append((mb_iterator_init_op, train_dataset_init_feed_dict))

                self.minibatch = mb_iterator.get_next()

    def init_streaming_aug(self):
        with self.graph.as_default():
            with tf.variable_scope('raw_queue'):
                all_io_phs = []
                is_binary = self.binary_inputs_outputs[0] + self.binary_inputs_outputs[1]

                train_dataset_init_feed_dict = {}

                with tf.device('/cpu:0'):
                    for tiles in self.inputs_tiles + self.outputs_tiles:
                        tiles_ph = tf.placeholder(dtype=tiles.dtype, shape=tiles.shape)
                        all_io_phs.append(tiles_ph)
                        train_dataset_init_feed_dict[tiles_ph] = tiles

                all_io_phs = tuple(all_io_phs)

                train_dataset = tf.data.Dataset.from_tensor_slices(all_io_phs).repeat()
                if self.shuffle_output:
                    train_dataset = train_dataset.shuffle(max(self.batch_size, self.queue_len*self.minibatch_size))

            # random transformation params
            dummy = np.zeros(shape=10)
            dd = tf.data.Dataset.from_tensor_slices(dummy)

            with tf.variable_scope('offset_queue'):
                # offset
                def rnd_ofs_map(_):
                    max_tile_ofs = np.asarray(
                                                [self.tile_depth - (self.strm_stack_shape[0] - 1)
                                                 * self.strm_z_stride,  # - 1 + 1
                                                 self.tile_size - self.strm_stack_shape[1] + 1,
                                                 self.tile_size - self.strm_stack_shape[2] + 1]
                                             )

                    # np.asarray([self.tile_depth, self.tile_size, self.tile_size]) \
                    #                                    - np.asarray(self.strm_stack_shape) + 1

                    rnd_ofs_x = tf.random_uniform([1], minval=0, maxval=max_tile_ofs[2], dtype=tf.int32, name="ofs_X")
                    rnd_ofs_y = tf.random_uniform([1], minval=0, maxval=max_tile_ofs[1], dtype=tf.int32, name="ofs_Y")
                    rnd_ofs_z = tf.random_uniform([1], minval=0, maxval=max_tile_ofs[0], dtype=tf.int32, name="ofs_Z")
                    rnd_flp_x = tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32, name="flp_X")
                    rnd_flp_y = tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32, name="flp_Y")

                    rnd_ofs = tf.concat([rnd_ofs_x, rnd_ofs_y, rnd_ofs_z, rnd_flp_x, rnd_flp_y], axis=0, name='ofs_flp')
                    return rnd_ofs

                ofs_dataset = dd.repeat().map(rnd_ofs_map, num_parallel_calls=1).prefetch(5)
                ofs_iterator = tf.data.Iterator.from_structure(output_types=ofs_dataset.output_types,
                                                               output_shapes=ofs_dataset.output_shapes)
                ofs_iterator_init = ofs_iterator.make_initializer(ofs_dataset)
                self.rnd_crop_ofs_it = ofs_iterator
                self.init_operations.append((ofs_iterator_init, None))

            with tf.variable_scope('distortion_queue'):
                # distortion
                def rnd_dist_map(_):
                    rnd_rot = tf.random_uniform([1], minval=0, maxval=2 * np.pi, dtype=tf.float32, name="rot")
                    rnd_a0 = tf.random_uniform([1], minval=0.8, maxval=1.2, dtype=tf.float32, name="A0")
                    rnd_a1 = tf.random_uniform([1], minval=-0.2, maxval=0.2, dtype=tf.float32, name="A1")
                    rnd_b0 = tf.random_uniform([1], minval=-0.2, maxval=0.2, dtype=tf.float32, name="B0")
                    rnd_b1 = tf.random_uniform([1], minval=0.8, maxval=1.2, dtype=tf.float32, name="B1")
                    rnd_c1 = tf.random_uniform([1], minval=-0.0001, maxval=0.0001, dtype=tf.float32, name="C1")
                    rnd_c2 = tf.random_uniform([1], minval=-0.0001, maxval=0.0001, dtype=tf.float32, name="C2")
                    rnd_dist = tf.concat([rnd_rot, rnd_a0, rnd_a1, rnd_b0, rnd_b1, rnd_c1, rnd_c2], axis=0, name='dist')
                    return rnd_dist

                dist_dataset = dd.repeat().map(rnd_dist_map, num_parallel_calls=1).prefetch(5)

                dist_iterator = tf.data.Iterator.from_structure(output_types=dist_dataset.output_types,
                                                                output_shapes=dist_dataset.output_shapes)
                dist_iterator_init = dist_iterator.make_initializer(dist_dataset)
                self.rnd_dist_prs_it = dist_iterator
                self.init_operations.append((dist_iterator_init, None))

                with tf.device(np.random.choice(self.preproc_devs)):
                    self.tst_img = tf.ones(name='test_img', dtype=tf.float32, shape=(1, self.tile_size, self.tile_size))

            with tf.variable_scope('transformed_queue'):
                # get suitable transformation
                def preproc(ims):
                    # print(ims)
                    ofs_pars = self.get_offset_and_distortion(20, 3)
                    dist_pars = ofs_pars[3]
                    crop_pars = ofs_pars[4]
                    # itr_ofs = ofs_pars[1]
                    # itr_dist = ofs_pars[0]

                    ims = list(ims)
                    ims_dist = self.get_distorted_ims(ims, is_binary, dist_pars, crop_pars, self.strm_z_stride)

                    # print(ofs_pars)
                    return tuple(ims_dist)

                train_dataset = train_dataset.batch(1)
                if self.queue_len > 0:
                    # train_dataset = train_dataset.prefetch(self.queue_len)
                    train_dataset.apply(tf.data.experimental.prefetch_to_device(self.preproc_devs[0]))

                mb_iterator = tf.data.Iterator.from_structure(output_types=train_dataset.output_types,
                                                              output_shapes=train_dataset.output_shapes)
                mb_iterator_init_op = mb_iterator.make_initializer(train_dataset)
                self.init_operations.append((mb_iterator_init_op, train_dataset_init_feed_dict))

                with tf.device(np.random.choice(self.preproc_devs)):
                    # mb = mb_iterator.get_next()
                    # print(mb)
                    mb_grouped = [[i[0] for i in mb_iterator.get_next()] for _ in range(self.minibatch_size)]
                    # print(mb_grouped)

                    inp = [preproc(i) for i in mb_grouped]
                    # print(inp)

                    inp_reorder = [[i[k] for i in inp] for k in range(len(is_binary))]

                    dist = [tf.stack(inp_i, axis=0) for inp_i in inp_reorder]

                    out_sh = [-1] + list(self.strm_stack_shape)
                    self.minibatch = tuple([tf.reshape(dist_img, shape=out_sh) for dist_img in dist])

                    smpl_idx = self.tile_depth//2
                    sample_stack = tf.reshape(mb_grouped[0][0][smpl_idx], [-1, 512, 512, 1])
                    sample_lbl = tf.reshape(tf.cast(mb_grouped[0][1][smpl_idx] * 255, tf.uint8), [-1, 512, 512, 1])

                    b_smpl_idx = self.strm_stack_shape[0]//2
                    dist_stk_sample = tf.reshape(self.minibatch[0][0, b_smpl_idx], [-1, 256, 256, 1])
                    dist_lbl_sample = tf.reshape(tf.cast(self.minibatch[1][0, b_smpl_idx] * 255, tf.uint8), [-1, 256, 256, 1])

                tf.summary.image('stack_input', sample_stack, 2, collections=['sample_single'])
                tf.summary.image('label_input', sample_lbl, 2, collections=['sample_single'])

                tf.summary.image('stack_dist', dist_stk_sample, 2, collections=['sample_single'])
                tf.summary.image('label_dist', dist_lbl_sample, 2, collections=['sample_single'])

    def init_streaming_no_aug(self):
        with self.graph.as_default():
            with tf.variable_scope('raw_queue'):
                all_io_phs = []
                # is_binary = self.binary_inputs_outputs[0] + self.binary_inputs_outputs[1]

                train_dataset_init_feed_dict = {}

                with tf.device('/cpu:0'):
                    for tiles in self.inputs_tiles + self.outputs_tiles:
                        tiles_ph = tf.placeholder(dtype=tiles.dtype, shape=tiles.shape)
                        all_io_phs.append(tiles_ph)
                        train_dataset_init_feed_dict[tiles_ph] = tiles

                all_io_phs = tuple(all_io_phs)

                train_dataset = tf.data.Dataset.from_tensor_slices(all_io_phs).repeat()
                if self.shuffle_output:
                    train_dataset = train_dataset.shuffle(max(self.batch_size, self.queue_len*self.minibatch_size))

            # random transformation params
            dummy = np.zeros(shape=10)
            dd = tf.data.Dataset.from_tensor_slices(dummy)

            max_tile_ofs = np.asarray([self.tile_depth, self.tile_size, self.tile_size]) \
                           - np.asarray(self.strm_stack_shape) + 1

            max_ofs = max_tile_ofs.max()
            need_crop = max_ofs > 0

            with tf.variable_scope('offset_queue'):
                # offset
                def rnd_ofs_map(_):
                    rnd_ofs_x = tf.random_uniform([1], minval=0, maxval=max_tile_ofs[2], dtype=tf.int32, name="ofs_X")
                    rnd_ofs_y = tf.random_uniform([1], minval=0, maxval=max_tile_ofs[1], dtype=tf.int32, name="ofs_Y")
                    rnd_ofs_z = tf.random_uniform([1], minval=0, maxval=max_tile_ofs[0], dtype=tf.int32, name="ofs_Z")
                    rnd_flp_x = tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32, name="flp_X")
                    rnd_flp_y = tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32, name="flp_Y")

                    rnd_ofs = tf.concat([rnd_ofs_x, rnd_ofs_y, rnd_ofs_z, rnd_flp_x, rnd_flp_y], axis=0, name='ofs_flp')
                    return rnd_ofs

                ofs_dataset = dd.repeat().map(rnd_ofs_map, num_parallel_calls=1).prefetch(5)
                ofs_iterator = tf.data.Iterator.from_structure(output_types=ofs_dataset.output_types,
                                                               output_shapes=ofs_dataset.output_shapes)
                ofs_iterator_init = ofs_iterator.make_initializer(ofs_dataset)
                self.rnd_crop_ofs_it = ofs_iterator
                self.init_operations.append((ofs_iterator_init, None))

            with tf.variable_scope('transformed_queue'):
                # get suitable transformation
                def preproc(*ims):
                    if not need_crop:
                        return ims

                    ofs = ofs_iterator.get_next()
                    ofs = tf.identity(ofs)

                    get_crop_res = lambda of, img: tf.strided_slice(img,
                                                                    begin=[of[2], of[1], of[0]],
                                                                    end=[of[2] + (self.strm_stack_shape[0] - 1)
                                                                         * self.strm_z_stride + 1,
                                                                         of[1] + self.strm_stack_shape[1],
                                                                         of[0] + self.strm_stack_shape[2]],
                                                                    strides=[self.strm_z_stride, 1, 1])

                    ims_crops = [get_crop_res(ofs, im) for im in ims]
                    return ims_crops

                train_dataset = train_dataset.map(preproc, num_parallel_calls=min(self.queue_len,
                                                                                  self.num_proc_threads))
                train_dataset = train_dataset.batch(self.minibatch_size)
                if self.queue_len > 0:
                    # train_dataset = train_dataset.prefetch(self.queue_len)
                    train_dataset.apply(tf.data.experimental.prefetch_to_device(self.preproc_devs[0]))

                mb_iterator = tf.data.Iterator.from_structure(output_types=train_dataset.output_types,
                                                              output_shapes=train_dataset.output_shapes)
                mb_iterator_init_op = mb_iterator.make_initializer(train_dataset)
                self.init_operations.append((mb_iterator_init_op, train_dataset_init_feed_dict))

                self.minibatch = mb_iterator.get_next()

    def get_offset(self, dist_img, max_iter=100):
        with tf.variable_scope('get_offset'):
            i0 = tf.constant(0, dtype=tf.int32, name='itr')
            m0 = tf.zeros([1, self.strm_stack_shape[1], self.strm_stack_shape[2]], dtype=self.dtype,
                          name='transformed_tst_img')
            o0 = tf.zeros([5, ], dtype=tf.int32, name='ofs_params')

            c = lambda i, m, o: tf.logical_and(tf.less(tf.reduce_mean(m), 0.985), tf.less(i, max_iter))

            get_crop = lambda of, img: tf.slice(img, [0, of[1], of[0]],
                                                [1, self.strm_stack_shape[1], self.strm_stack_shape[2]])

            # noinspection PyUnusedLocal
            def b(i, m, o):
                """
                Loop iteration

                Args:
                    i: iteration count
                    m: prev image
                    o: prev offset
                """
                ofs_deq = self.rnd_crop_ofs_it.get_next()
                # ofs_deq = tf.Print(ofs_deq, data=[rndGenQueueC.size()], message="rndGenQueueC: left ")
                # ofs_deq = tf.Print(ofs_deq, data=[ofs_deq], message="off")
                of = tf.identity(ofs_deq)
                # of = tf.Print(of, data=[of, distImg.get_shape()], message='GetCrop')
                return [i + 1, get_crop(of, dist_img), of]

            res = tf.while_loop(
                c, b, loop_vars=[i0, m0, o0],
                shape_invariants=[i0.get_shape(), m0.shape, o0.shape], name='offset_getter_loop')
            return res

    # noinspection PyMethodMayBeStatic
    def get_distorted(self, dist_p, img, interpolation='BILINEAR'):
        with tf.variable_scope('get_distorted'):
            # print(img)
            im_tr = tf.transpose(img, [1, 2, 0])
            rot_trans_tr = tf.contrib.image.transform(im_tr,
                                                      [dist_p[1], dist_p[2], 0, dist_p[3], dist_p[4], 0, dist_p[5],
                                                       dist_p[6]], interpolation=interpolation)
            rot_image_tr = tf.contrib.image.rotate(rot_trans_tr, dist_p[0], interpolation=interpolation)
            rot_image = tf.transpose(rot_image_tr, [2, 0, 1])
            return rot_image

    def get_offset_and_distortion(self, max_iter_ofs=100, max_iter_dist=10):
        with tf.variable_scope('get_offset_and_dist'):
            i0 = tf.constant(0, dtype=tf.int32, name='dist_itr')
            f0 = tf.constant(max_iter_ofs, dtype=tf.int32, name='ofs_itr')
            m0 = tf.zeros([1, self.strm_stack_shape[1], self.strm_stack_shape[2]], dtype=self.dtype,
                          name='transformed_tst_img')
            d0 = tf.zeros([7, ], name='dist_pars')
            o0 = tf.zeros([5, ], dtype=tf.int32, name='ofs_pars')
            c = lambda i, f, m, d, o: tf.logical_and(tf.equal(f, max_iter_ofs), tf.less(i, max_iter_dist))

            # noinspection PyUnusedLocal
            def b(i, f, m, d, o):
                ofs = self.rnd_dist_prs_it.get_next()
                # ofs = tf.Print(ofs, data=[rndGenQueue.size()], message="rndGenQueue: left ")
                ofs = tf.identity(ofs)
                dist_img = self.get_distorted(ofs, self.tst_img)
                itr_img_ofs = self.get_offset(dist_img, max_iter_ofs)
                return [i + 1, itr_img_ofs[0], itr_img_ofs[1], ofs, itr_img_ofs[2]]

            res = tf.while_loop(
                c, b, loop_vars=[i0, f0, m0, d0, o0],
                shape_invariants=[i0.get_shape(), f0.get_shape(), m0.shape, d0.shape, o0.shape],
                name='offset_dist_getter_loop')
            return res

    def get_distorted_ims(self, ims, is_binary, dist_param, crop_param, z_stride):

        ims_dist = []

        str_x = crop_param[3] * 2 - 1
        str_y = crop_param[4] * 2 - 1

        get_crop_res_xy = lambda of, img: tf.slice(img, begin=[0, of[1], of[0]], size=[-1,
                                                                                       self.strm_stack_shape[1],
                                                                                       self.strm_stack_shape[2]])
        get_crop_res_z = lambda of, img: tf.strided_slice(img,
                                                          begin=[of[2]],
                                                          end=[of[2] + (self.strm_stack_shape[0] - 1) *
                                                               z_stride + 1],
                                                          strides=[z_stride])


        # get_crop_res_z = lambda of, img: tf.slice(img, begin=[of[2], 0, 0], size=[self.strm_stack_shape[0], -1, -1])
        out_sh = [-1] + list(self.strm_stack_shape)

        for im, is_bin in zip(ims, is_binary):

            if is_bin:
                im = im * 2

            cropped_im_z = get_crop_res_z(crop_param, im)
            trans_im = self.get_distorted(dist_param, cropped_im_z)

            #with tf.device('/cpu:0'):
            #    crop_param = tf.Print(crop_param, data=[crop_param,
            #                                            tf.shape(im),
            #                                            tf.shape(cropped_im_z),
            #                                            tf.shape(trans_im)], message='get_crop_res_xy')

            cropped_im = get_crop_res_xy(crop_param, trans_im)

            flipped_im = cropped_im[:, ::str_y, ::str_x]

            if is_bin:
                distorted_img = tf.clip_by_value(flipped_im, clip_value_min=0, clip_value_max=1)
            else:
                distorted_img = tf.image.random_brightness(image=flipped_im, max_delta=0.4)
                distorted_img = tf.image.random_contrast(distorted_img, lower=0.5, upper=1.5)

            distorted_img = tf.reshape(distorted_img, shape=out_sh)
            ims_dist.append(distorted_img)

        return ims_dist

    def start_queues(self, session, run_options=None, run_metadata=None):
        for op, dct in self.init_operations:
            session.run(op, options=run_options,
                        run_metadata=run_metadata, feed_dict=dct)
            time.sleep(2)

    def stop_queues(self):
        pass

    # def GetSingleTrainingItem(self):
    #    return self.img_dist_q.dequeue()

    def get_minibatch_stream(self):
        return self.minibatch
