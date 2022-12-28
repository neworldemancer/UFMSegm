import tensorflow as tf
import pickle
import numpy as np
from . import net_utils as nu
import os


class Predictor(object):
    """
    class to make constant model and perform predictions.
    constant models are cached in files
    """

    def __init__(self, model_dir, checkpoint_idx, device_id, batch_sz, in_out_dict=None, input_sz=None,
                 device_ids=None, gpuid=0):

        self.model_dir = model_dir
        self.checkpoint_idx = checkpoint_idx
        self.batch_sz = batch_sz

        self.model_path, self.const_model_path, \
            self.pred_model_path, self.rec_field_path = self.get_paths()

        self.deice_id = device_id
        self.device_ids = device_ids
        self.gpuid = gpuid
        self.input_sz = input_sz

        self.pred_gr_def = None
        self.const_gr_def = None
        self.model_info_dict = {}

        self.in_out_dict = in_out_dict
        geom_dict_loaded = self.load_geometry_dict()
        if self.in_out_dict is None:
            if geom_dict_loaded:
                self.in_out_dict = self.model_info_dict['const_in_out_dict']
            else:
                raise ValueError('The model_info_dict.pkl should be present if in_out_dict is not provided')

        self.load_pred_model()

    def get_paths(self):
        model_path = os.path.join(self.model_dir, 'model-%d' % self.checkpoint_idx)
        const_model_path = os.path.join(self.model_dir, 'cmodel-%d.pb' % self.checkpoint_idx)
        model_rec_fld_path = os.path.join(self.model_dir, 'model_info_dict.pkl')
        pred_model_path = os.path.join(self.model_dir, 'pred_b%d-%d.pb' %
                                       (self.batch_sz, self.checkpoint_idx))

        return model_path, const_model_path, pred_model_path, model_rec_fld_path

    def load_geometry_dict(self):
        if not os.path.exists(self.rec_field_path):
            return False
        with open(self.rec_field_path, 'rb') as f:
            self.model_info_dict = pickle.load(f)

        return True

    def save_geometry_dict(self):
        with open(self.rec_field_path, 'wb') as f:
            pickle.dump(self.model_info_dict, f, protocol=0)

    def load_pred_model(self):
        if not os.path.exists(self.pred_model_path):
            self.create_pred_model()

        with tf.gfile.GFile(self.pred_model_path, mode='rb') as f:
            self.pred_gr_def = tf.GraphDef()
            str_gd = f.read()
            self.pred_gr_def.ParseFromString(str_gd)

        self.load_geometry_dict()
        pass

    def load_const_model(self):
        if not os.path.exists(self.const_model_path):
            self.create_constant_model()

        with tf.gfile.GFile(self.const_model_path, mode='rb') as f:
            self.const_gr_def = tf.GraphDef()
            str_gd = f.read()
            self.const_gr_def.ParseFromString(str_gd)

        pass

    def create_pred_model(self):
        self.load_const_model()
        _ = dir(tf.contrib)

        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(self.const_gr_def, name='')

            const_gr_def_no_shape = graph.as_graph_def(add_shapes=False)

            x = graph.get_tensor_by_name(self.in_out_dict['in'])
            #phase = tf.constant(True)
            sh = x.get_shape().as_list()
            sh[0] = self.batch_sz
            sh_new = []

            make_2d = False
            if self.input_sz is not None:
                if len(sh) == 5:  # BDHWC
                    sh[2:4] = self.input_sz
                    sh_new = sh.copy()
                elif len(sh) == 4:  # BHWD  - in 2D conv models D==C
                    sh[1:3] = self.input_sz
                    sh_new = [self.batch_sz] + sh[-1:] + self.input_sz + [1]
                    make_2d = True

        graph = tf.Graph()
        with graph.as_default():  # new graph loads constant model and variable input
            x = tf.placeholder(dtype=tf.float32,
                               shape=sh_new,
                               name='pred_x')

            if make_2d:
                x_int = tf.transpose(x, perm=[0, 2, 3, 1, 4], name='pred_x_2D_intermediate')
                x = tf.reshape(x_int, shape=sh, name='pred_x_2D')

            out_tensors = self.in_out_dict['out'] if isinstance(self.in_out_dict['out'], list) \
                else [self.in_out_dict['out']]
            y0 = tf.import_graph_def(const_gr_def_no_shape, input_map={self.in_out_dict['in']: x},
                                     return_elements=out_tensors,
                                     name='')

            if isinstance(self.in_out_dict['out'], list):
                all_y = [y if i == -1 else y[..., i]
                         for idxes, y in zip(self.in_out_dict['out_channels'], y0) for i in idxes]
            else:
                all_y = [y0[0][..., i] for i in self.in_out_dict['out_channels']]

            all_t_stack = tf.stack(all_y, axis=-1)
            all_t_stack_lim = 255 * all_t_stack
            all_t_stack_lim = tf.clip_by_value(all_t_stack_lim, 0., 255.)
            tf.cast(x=all_t_stack_lim, dtype=tf.uint8, name='pred_y')

        gr_def = graph.as_graph_def(add_shapes=True)

        with tf.gfile.GFile(self.pred_model_path, mode='wb') as f:
            f.write(gr_def.SerializeToString())

        # path, name = os.path.split(self.pred_model_path)
        # tf.train.write_graph(gr_def, path, name)

        n_outs = [len(idxes) if isinstance(idxes, list) else 1 for idxes in self.in_out_dict['out_channels']]
        n_out = np.sum(n_outs)
        self.model_info_dict['pred_in_out_dict'] = {'in': 'pred_x:0',
                                                    'out': 'pred_y:0',
                                                    'n_out': n_out}

        self.save_geometry_dict()

    def create_constant_model(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(self.deice_id) if not self.device_ids else ','.join([str(idx) for idx in self.device_ids])

        g = tf.Graph()
        with tf.Session(graph=g, config=config) as sess:
            sess.run(tf.global_variables_initializer())

            _ = dir(tf.contrib)  # required for importing operations from contrib
            saver = tf.train.import_meta_graph(self.model_path + '.meta', clear_devices=True)

            saver.restore(sess, self.model_path)

            output_tensor_names = self.in_out_dict['out'] if isinstance(self.in_out_dict['out'], list) \
                else [self.in_out_dict['out']]

            output_op_names = [name.split(':')[0] for name in output_tensor_names]

            gd = g.as_graph_def(add_shapes=True)

            # fix batch norm nodes
            for node in gd.node:
                if node.op == 'RefSwitch':
                    node.op = 'Switch'
                    for index in range(len(node.input)):
                        if 'moving_' in node.input[index]:
                            node.input[index] = node.input[index] + '/read'
                elif node.op == 'AssignSub':
                    node.op = 'Sub'
                    if 'use_locking' in node.attr: del node.attr['use_locking']

            gd_c = tf.graph_util.convert_variables_to_constants(sess,
                                                                gd,
                                                                output_node_names=output_op_names)

            with tf.gfile.GFile(self.const_model_path, mode='wb') as f:
                f.write(gd_c.SerializeToString())

            # path, name = os.path.split(self.const_model_path)
            # tf.train.write_graph(gd_c, path, name)

            self.model_info_dict['const_in_out_dict'] = {'in': self.in_out_dict['in'],
                                                         'out': self.in_out_dict['out']}

            if 'receptive_field' not in self.model_info_dict:
                print('Obtainig receptive field')
                out_tensor = self.in_out_dict['out'][0] if isinstance(self.in_out_dict['out'], list) \
                             else self.in_out_dict['out']
                # ToDo: out_tensor = self.in_out_dict['out'][-1] if isinstance(self.in_out_dict['out'], list) \
                #             else self.in_out_dict['out']

                rf = nu.get_receptive_field_size(self.model_path,
                                                 self.in_out_dict['in'],
                                                 out_tensor,
                                                 self.deice_id)
        
                self.model_info_dict['receptive_field'] = {'rhl': rf[0],
                                                           'rhr': rf[1],
                                                           'rvu': rf[2],
                                                           'rvd': rf[3]}

        pass

    def predict_image_stack(self, stack, margin=8, keep_edge=True, edge_size=0):
        _ = dir(tf.contrib)

        gt = tf.Graph()
        with gt.as_default():  # new graph loads constant model and variable input
            if self.gpuid:
                with tf.device('/gpu:%d'%self.gpuid):
                    tf.import_graph_def(self.pred_gr_def, name='')
                    x = gt.get_tensor_by_name(self.model_info_dict['pred_in_out_dict']['in'])
                    y = gt.get_tensor_by_name(self.model_info_dict['pred_in_out_dict']['out'])
            else:
                tf.import_graph_def(self.pred_gr_def, name='')
                x = gt.get_tensor_by_name(self.model_info_dict['pred_in_out_dict']['in'])
                y = gt.get_tensor_by_name(self.model_info_dict['pred_in_out_dict']['out'])

        t = None
        for op in gt.get_operations():
            if 'phase_train' in op.name:
                t = gt.get_tensor_by_name(op.name + ':0')

        sh_in = x.get_shape().as_list()
        n_b = sh_in[0]

        rf = self.model_info_dict['receptive_field']
        # crop_range = [rf['rvu']+margin, -rf['rvd']-margin, rf['rhl']+margin, -rf['rhr']-margin]

        pred_map = self.generate_map(stack, sh_in[1:4], margin, keep_edge)

        out_shape = list(stack.shape[:-1]) + [self.model_info_dict['pred_in_out_dict']['n_out']]
        res = np.zeros(out_shape, dtype=np.uint8)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(self.deice_id) if not self.device_ids else ','.join([str(idx) for idx in self.device_ids])

        with tf.Session(graph=gt, config=config) as sess:
            n_items = len(pred_map)
            n_it = (n_items + n_b - 1) // n_b

            print('total iterations:', n_it)

            for itr in range(n_it):

                idx_start = min(n_items - n_b, n_b * itr)
                batch_map = pred_map[idx_start:idx_start + n_b]

                batch_list = [stack[i[0][0]:i[0][3],
                                    i[0][1]:i[0][4],
                                    i[0][2]:i[0][5]] for i in batch_map]
                batch = np.stack(batch_list)

                print(itr, end='\r')

                fd = {x: batch}
                if t is not None:
                    fd[t] = False

                batch_res = sess.run(y, feed_dict=fd)

                len_sh = len(batch_res.shape)
                assert(4 <= len_sh <= 5)
                is_2d = len_sh == 4

                for i, item_res in enumerate(batch_res):
                    dst_coord = batch_map[i][1]
                    crop_range = batch_map[i][2]
                    # print(batch_map[i][0], '-> (', crop_range, ')', dst_coord)

                    res[dst_coord[0]:dst_coord[3],
                        dst_coord[1]:dst_coord[4],
                        dst_coord[2]:dst_coord[5]] = item_res[...,
                                                              crop_range[0]:crop_range[1],
                                                              crop_range[2]:crop_range[3],
                                                              :
                                                             ]
        if keep_edge and edge_size > 0:
            res[:, 0:edge_size, :] = 0
            res[:, -edge_size:, :] = 0
            res[:, :, 0:edge_size] = 0
            res[:, :, -edge_size:] = 0

        return res

    def generate_map(self, stack, input_shape, margin, keep_edge):
        rf = self.model_info_dict['receptive_field']
        rxm = rf['rhl'] + margin
        rxp = rf['rhr'] + margin
        rym = rf['rvu'] + margin
        ryp = rf['rvd'] + margin



        in_sz_z, in_sz_y, in_sz_x = input_shape
        overlap_z = in_sz_z - 1
        rzm = overlap_z // 2
        rzp = overlap_z - rzm

        overlap_x = rxm + rxp
        overlap_y = rym + ryp
        overlap_z = rzm + rzp

        ofs_x = in_sz_x - overlap_x
        ofs_y = in_sz_y - overlap_y
        ofs_z = in_sz_z - overlap_z

        sz_z, sz_y, sz_x, _ = stack.shape

        n_it_x = (sz_x - overlap_x + ofs_x - 1) // ofs_x
        n_it_y = (sz_y - overlap_y + ofs_y - 1) // ofs_y
        n_it_z = (sz_z - overlap_z + ofs_z - 1) // ofs_z

        pred_map = []

        for z in range(n_it_z):
            z_min = z * ofs_z
            z_max = min(sz_z, z_min + in_sz_z)
            z_min = z_max - in_sz_z

            for y in range(n_it_y):
                y_min = y * ofs_y
                y_max = min(sz_y, y_min + in_sz_y)
                y_min = y_max - in_sz_y

                rym_i = 0 if keep_edge and y==0        else rym
                ryp_i = 0 if keep_edge and y==n_it_y-1 else ryp

                for x in range(n_it_x):
                    x_min = x * ofs_x
                    x_max = min(sz_x, x_min + in_sz_x)
                    x_min = x_max - in_sz_x

                    rxm_i = 0 if keep_edge and x == 0          else rxm
                    rxp_i = 0 if keep_edge and x == n_it_x - 1 else rxp

                    input_range = [z_min, y_min, x_min, z_max, y_max, x_max]
                    output_range = [z_min + rzm, y_min + rym_i, x_min + rxm_i,
                                    z_max - rzp, y_max - ryp_i, x_max - rxp_i]

                    crop_range = [rym_i, in_sz_y-ryp_i, rxm_i, in_sz_x-rxp_i]

                    pred_map.append([input_range, output_range, crop_range])

        return pred_map

    @staticmethod
    def normalize_stack(stack, norm_stack=None, bad_model=False, normalization_percentile_range=None):
        """
        Args:
            stack (np.ndarray): DHW stack, single channel
            norm_stack (np.ndarray): : DHW stack, single channel, to be used for normalization
            bad_model (bool): if true, normalize by mean and avg of (stack,stack,stack,255)

        Returns:
            stack_n (np.ndarray): DHWC normalized stack, C==1
        """

        n_stack = stack if norm_stack is None else norm_stack

        if bad_model:
            bad_stack = np.stack((n_stack, n_stack, n_stack, np.ones_like(n_stack)*255))
            mean, std = bad_stack.mean(), bad_stack.std()
        else:
            if normalization_percentile_range is None:
                nrm_stack = n_stack
            else:
                nrm_stack = n_stack.flatten()
                p1, p2 = normalization_percentile_range
                r1, r2 = np.percentile(nrm_stack, p1), np.percentile(nrm_stack, p2)
                x_idx = (nrm_stack > r1) * (nrm_stack < r2)
                nrm_stack = nrm_stack[x_idx]

            mean, std = nrm_stack.mean(), nrm_stack.std()

        stack_n = (stack.astype(np.float32) - mean) / std
        stack_n = stack_n[..., np.newaxis]
        return stack_n
