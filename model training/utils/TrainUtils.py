import os
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.client import timeline

from utils import imgio as iio
import requests


class Trainer(object):
    """
    Helper class for training cycle, load, store, freeze, etc.
    Not implemented yet =P
    """

    def __init__(self, graph_info_dict, model_name, models_save_dir, dev_id, dev_ids=None):
        """
        Constructor.
        
        Args: 
            graph_info_dict (dict): Dictionary defining training elements: 'graph'->tf.Graph,
                                        'in' -> tf.Tensor (input), 
                                        'out' -> tf.Tensor (output), 'lbl' -> tf.Tensor (correct output),
                                        'opt'->tf.Operation (optimizer call), 'loss'->tf.Tensor,
                                        'lr'->nu.LearningRate, 'mb_gen_tra'->function (generates minibatch, (data,lbl) )
                                        'init'->function to run before training with sess as param
                                        'fin'->function to run after training with sess as param
                                        'feed_dict_tra' -> flag whether to use the feed_dict during training
                                        'feed_dict_val' -> flag whether to use the feed_dict during validation
                                                            and sampling
                                        'train_phase_ph' -> placeholder for training/run phase
                                        'num_val_minibatches' -> number of minibatches to run validation on
            model_name (str): model name: all files will be named accordingly
            models_save_dir (str): root directory where models are saved
            dev_id (int): CUDA device id to run on
            dev_ids (int, int,...): list of CUDA device ids to run on. If set, dev_id should be None
        """

        self.g = graph_info_dict['graph']
        self.f_init = graph_info_dict['init']
        self.f_fin = graph_info_dict['fin']

        self.feed_dict_tensors = graph_info_dict['feed_dict_tensors']
        self.opt = graph_info_dict['opt']

        self.loss = graph_info_dict['loss']
        self.lr = graph_info_dict['lr']

        self.mb_gen_tra = graph_info_dict['mb_gen_tra']
        self.mb_gen_val = graph_info_dict['mb_gen_val']
        self.num_val_minibatches = graph_info_dict['num_val_minibatches']
        self.use_feed_dict_tra = graph_info_dict['feed_dict_tra']
        self.use_feed_dict_val = graph_info_dict['feed_dict_val']
        self.train_phase_ph = graph_info_dict['train_phase_ph']

        self.itr_callback = graph_info_dict.get('itr_callback', None)

        self.run_mode = graph_info_dict.get('run_mode')                # </ optional
        self.n_run_modes = graph_info_dict.get('n_run_modes', 1)
        self.run_modes_ratio = graph_info_dict.get('run_modes_ratio', [1])
        if not (isinstance(self.opt, list) or isinstance(self.opt, tuple)):
            self.opt = [self.opt] * self.n_run_modes
        if not (isinstance(self.loss, list) or isinstance(self.loss, tuple)):
            self.loss = [self.loss] * self.n_run_modes

        self.draw_states = graph_info_dict['draw_states']
        self.draw_states_ttl = graph_info_dict['draw_states_ttl']
        if len(self.draw_states) > 0 and not isinstance(self.draw_states[0], list):
            self.draw_states = [self.draw_states]*self.n_run_modes
        if len(self.draw_states_ttl) > 0 and not isinstance(self.draw_states_ttl[0], list):
            self.draw_states_ttl = [self.draw_states_ttl]*self.n_run_modes

        self.model_name = model_name
        self.models_save_dir = models_save_dir

        self.dev_id = dev_id
        self.dev_ids = dev_ids

        self.tra_loss_hist = [[] for _ in range(self.n_run_modes)]
        self.val_loss_hist = [[] for _ in range(self.n_run_modes)]
        self.val_hist_time = []

        self.has_notified = False

        with self.g.as_default():
            if len([op for op in self.g.get_operations() if 'losses_list' in op.name]):
                self.losses_ph = self.g.get_tensor_by_name('losses_list:0')
                self.mean_loss = self.g.get_tensor_by_name('losses_mean:0')
            else:
                self.losses_ph = tf.placeholder(dtype=tf.float32, shape=(None,), name='losses_list')
                self.mean_loss = tf.reduce_mean(self.losses_ph, axis=0, name='losses_mean')

                with tf.device('/cpu:0'):
                    tf.summary.scalar('Loss', self.mean_loss, collections=['train_validation'])

    def get_timed_model_dir(self):
        t = time.localtime()
        time_str = '%d.%02d.%02d_%02d-%02d' % (
            t.tm_year, t.tm_mon, t.tm_mday,
            t.tm_hour, t.tm_min)

        md = 'model_%s_%s' % (
            self.model_name,
            time_str)
        return md, time_str

    def get_timed_model_checkpoint(self, time_str, checkpoint_idx):
        cpf = 'model_%s_%s/model-%d' % (
            self.model_name,
            time_str,
            checkpoint_idx)
        return cpf

    def get_checkpoint_file(self, start_params_dict):
        root = start_params_dict['root'] if 'root' in start_params_dict else None
        root = root or self.models_save_dir
        file_name = self.get_timed_model_checkpoint(start_params_dict['time'], start_params_dict['idx'])
        file_name = os.path.join(root, file_name)
        return file_name

    def get_validation_loss(self, sess, iter_idx, val_sum, val_sum_writer):
        for r_mode in range(self.n_run_modes):
            val_losses_mb = []

            if self.use_feed_dict_val:
                print('feed dict for validation')
            for i in range(self.num_val_minibatches):

                dct = {self.train_phase_ph: False}
                if self.use_feed_dict_val:
                    sl = self.mb_gen_val() if self.n_run_modes == 1 else self.mb_gen_val(r_mode)
                    for tensor, value in zip(self.feed_dict_tensors, sl):
                        dct[tensor] = value

                if self.run_mode is not None:
                    dct[self.run_mode] = r_mode

                val_loss = sess.run(self.loss[r_mode], feed_dict=dct)
                val_losses_mb.append(val_loss)

            if val_sum_writer is not None:
                # print('save_sum, val', val_losses_mb)
                summ = sess.run(val_sum, feed_dict={self.losses_ph: np.asarray(val_losses_mb, dtype=np.float32)})
                val_sum_writer.add_summary(summ, iter_idx)

            self.val_loss_hist[r_mode].append(np.mean(np.asarray(val_losses_mb)))
        self.val_hist_time.append(iter_idx)

    def notify_loss(self):
        if not self.has_notified:
            requests.post('https://hooks.slack.com/services/X/Y/Z',
                          data={'payload': """{
                                      "channel": "#bbb",
                                      "username": "DeepBot",
                                      "text": "loss indicates failure in model %s", 
                                      "icon_emoji": ":shit:"}""" % self.model_name
                                }
                          )
            self.has_notified = True

    def notify_done(self):
        requests.post('https://hooks.slack.com/services/X/Y/Z',
                      data={'payload': """{
                              "channel": "#bbb",
                              "username": "DeepBot",
                              "text": "Training model %s is done!", 
                              "icon_emoji": ":sunglasses:"}""" % self.model_name
                            }
                      )

    def train(self, n_iter,
              learning_rate,
              learning_rate_factors={0: 0.5, 200: 1},
              draw_rate=((10, 100), (100, 500), (250, 2000), (1000, 5000), (5000, 50000), (10000, 5000000)),
              start_params_dict=None,
              save=True,
              save_summary=True,
              notify_max_loss=0.1):
        """
        executes training loop.
        
        Args:
            n_iter (int): number of iterations of minibatches (not epochs!)
            learning_rate (float): what to say
            save (bool): flag whether to save checkpoints
            learning_rate_factors (dict): lr factors. key: at which iteration to apply, value: which factor
                                          e.g, if lr=0.1, l_r_f={10: 0.5, 200: 2}, then:
                                              iteration   0->9  : lr = 0.1
                                              iteration  10->199: lr = 0.05
                                              iteration 200->inf: lr = 0.2
            draw_rate (tuple): pairs, how often [0] to draw plots and store a checkpoint, until which [1] iteration.
                                          e.g., ((10, 100), (100, 5000), (5000, 50000)) means:
                                              iteration    0->100  : every 10 iterations
                                              iteration  100->5000 : every 100 iterations
                                              iteration 5000->inf  : every 5000 iterations
            start_params_dict (dict): if not `None` initialize trainable variables from checkpoint. 
                                          'time'-> time string, e.g. '2018.03.19_16-28'
                                          'idx' -> checkpoint index, e.g. 50000
                                          'root' -> root dir of models, optional
            save_summary (bool): save TensorBoard summary
            notify_max_loss (float): if after 1000 steps the minimum loss over last 200 steps
                                           exceeds this value a notification is sent
        """

        self.has_notified = False

        model_dir, time_str = self.get_timed_model_dir()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(self.dev_id) if self.dev_id else str(self.dev_ids)[1:-1]

        with tf.Session(graph=self.g, config=config) as sess:
            draw_rate_idx = 0
            prev_rate_last_train_idx = 0
            self.tra_loss_hist = [[] for _ in range(self.n_run_modes)]
            self.val_loss_hist = [[] for _ in range(self.n_run_modes)]
            self.val_hist_time = []

            model_path = os.path.join(self.models_save_dir, model_dir)
            if save or save_summary:
                os.makedirs(model_path, exist_ok=True)

            if save_summary:
                merged_summary_train = tf.summary.merge_all()
                merged_summary_sample = tf.summary.merge_all('sample')
                merged_summary_sample_0 = tf.summary.merge_all('sample_single')
                # merged_summary_valid = tf.summary.merge_all('validation')
                merged_summary_tra_val = tf.summary.merge_all('train_validation')

                sum_writer = tf.summary.FileWriter(os.path.join(model_path, 'summary/tra'), sess.graph)
                sum_writer_val = tf.summary.FileWriter(os.path.join(model_path, 'summary/val'), sess.graph)
            else:
                merged_summary_train = merged_summary_sample = \
                    merged_summary_sample_0 = merged_summary_tra_val = None
                sum_writer = sum_writer_val = None

            saver = tf.train.Saver(max_to_keep=2000)

            if start_params_dict is not None:
                sess.run(tf.local_variables_initializer())
                file_name = self.get_checkpoint_file(start_params_dict)
                saver.restore(sess, file_name)
            else:
                init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                sess.run(init_op)

            if self.f_init:
                self.f_init(sess)

            self.lr.set_value(sess, learning_rate)

            if save:
                print('checkpoints will be saved to:', model_path)
                saver.export_meta_graph(os.path.join(model_path, 'model.meta'))

            t0 = time.time()
            last_time_idx = -1
            last_val_idx = -1

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            for it_i in range(n_iter):  # iterate training
                if self.itr_callback:
                    self.itr_callback(itr=it_i, sess=sess)

                if it_i in learning_rate_factors:  # update lr if needed
                    lrf = learning_rate * learning_rate_factors[it_i]
                    print('Setting lr=%f' % lrf)
                    self.lr.set_value(sess, lrf)

                tr_losses = []
                for r_mode in range(self.n_run_modes):
                    dct = {}
                    if self.train_phase_ph is not None:
                        dct[self.train_phase_ph] = True

                    if self.use_feed_dict_tra:
                        st_lbl_mb = self.mb_gen_tra() if self.n_run_modes == 1 else self.mb_gen_tra(r_mode)
                        for tensor, value in zip(self.feed_dict_tensors, st_lbl_mb):
                            dct[tensor] = value
                    if self.run_mode is not None:
                        dct[self.run_mode] = r_mode

                    tra_loss = 0
                    for _ in range(self.run_modes_ratio[r_mode]):
                        _, t_loss = sess.run([self.opt[r_mode], self.loss[r_mode]],
                                             feed_dict=dct,
                                             run_metadata=run_metadata if it_i == 0 else None,
                                             options=run_options if it_i == 0 else None)

                        tra_loss += t_loss

                    tra_loss /= self.run_modes_ratio[r_mode]
                    tr_losses.append(tra_loss)

                    self.tra_loss_hist[r_mode].append(tra_loss)

                if save_summary:
                    tr_losses = np.asarray(tr_losses, dtype=np.float32)
                    # print('save_sum, tra', tr_losses)

                    f_dict = {self.losses_ph: tr_losses}
                    if self.train_phase_ph is not None:
                        f_dict[self.train_phase_ph] = True

                    if merged_summary_train is not None:
                        summary_tr = sess.run(merged_summary_train, feed_dict=f_dict)
                        sum_writer.add_summary(summary_tr, it_i)

                    if merged_summary_tra_val is not None:
                        summary_tv = sess.run(merged_summary_tra_val, feed_dict=f_dict)
                        sum_writer.add_summary(summary_tv, it_i)

                if it_i == 0 and save:  # save timeline
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format(
                        show_dataflow=True, show_memory=True)
                    with open(os.path.join(model_path, 'timeline.json'), 'w') as f:
                        f.write(ctf)

                if it_i > 1000 and np.min(np.asarray(self.tra_loss_hist[-200:])) > notify_max_loss:
                    self.notify_loss()

                rate_par = draw_rate[draw_rate_idx] if draw_rate_idx < len(draw_rate) else draw_rate[-1]
                rate = rate_par[0]
                draw_it = it_i - prev_rate_last_train_idx

                if draw_it % rate == 0 or it_i == n_iter - 1:
                    t1 = time.time()
                    print('time per it: %f' % ((t1 - t0) / (it_i - last_time_idx)))
                    t0 = t1
                    last_time_idx = it_i

                    if last_val_idx + 100 <= it_i or it_i == 0:
                        self.get_validation_loss(sess, it_i, merged_summary_tra_val,
                                                 sum_writer_val if save_summary else None)
                        last_val_idx = it_i

                    self.draw_val_sample(sess)

                    for r_mode in range(self.n_run_modes):
                        dct = {}
                        if self.train_phase_ph is not None:
                            dct[self.train_phase_ph] = False
                        if self.run_mode is not None:
                            dct[self.run_mode] = r_mode

                        if save_summary and merged_summary_sample is not None:
                            summary = sess.run(merged_summary_sample, feed_dict=dct)
                            sum_writer.add_summary(summary, it_i)

                        if it_i == 0 and (save_summary and merged_summary_sample_0 is not None):
                            summary = sess.run(merged_summary_sample_0, feed_dict=dct)
                            sum_writer.add_summary(summary, it_i)


                    if save:
                        saver.save(sess,
                                   os.path.join(model_path, 'model'),
                                   global_step=it_i + 1)  # +1 : num accomplished steps, not id

                elif last_val_idx + 5000 <= it_i:
                    self.get_validation_loss(sess, it_i, merged_summary_tra_val,
                                             sum_writer_val if save_summary else None)
                    last_val_idx = it_i

                if draw_rate_idx < len(draw_rate) - 1 and it_i >= rate_par[1]:
                    draw_rate_idx = draw_rate_idx + 1
                    prev_rate_last_train_idx = it_i
                if save and it_i == n_iter - 1:
                    saver.save(sess,
                               os.path.join(model_path, 'model'),
                               global_step=it_i + 1)

            if self.f_fin:
                self.f_fin(sess)

        self.notify_done()
        return time_str

    def draw_val_sample(self, sess):
        for r_mode in range(self.n_run_modes):
            dct = {}
            if self.train_phase_ph is not None:
                dct[self.train_phase_ph] = False
            if self.run_mode is not None:
                dct[self.run_mode] = r_mode

            if self.use_feed_dict_val:
                st_lbl_mb = self.mb_gen_val() if self.n_run_modes == 1 else self.mb_gen_val(r_mode)
                for tensor, value in zip(self.feed_dict_tensors, st_lbl_mb):
                    dct[tensor] = value

            # x_montage = iio.montage(st_tr[0, 2:3, :, :, 0], save_to=None)
            # y_montage = iio.montage(st_pred[0, :, :, :, 1], save_to=None)
            # lbl_montage = iio.montage(lbl_tr[0, :, :, :], save_to=None)

            draw_states_res = sess.run(self.draw_states, feed_dict=dct)

            slots, _ = iio.draw_samples(draw_states_res[r_mode],
                                        self.draw_states_ttl[r_mode] + ['training history' +
                                                                        ' %d' % r_mode if self.n_run_modes > 1
                                                                        else ''
                                                                       ],
                                        height=5, num_extra_slots=1)
            slots[0].semilogy(self.tra_loss_hist[r_mode], 'r')
            slots[0].semilogy(self.val_hist_time, self.val_loss_hist[r_mode], 'b')
            slots[0].legend(['train', 'valid'], loc='best')
            plt.show()

    def get_session(self, start_params_dict=None):
        """
        Creates session on graph and loads params if needed.

        Args:
            start_params_dict (dict): if not `None` initialize trainable variables from checkpoint. 
                                          'time'-> time string, e.g. '2018.03.19_16-28'
                                          'idx' -> checkpoint index, e.g. 50000
                                          'root' -> root dir of models, optional
        """

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(self.dev_id) if self.dev_id else str(self.dev_ids)[1:-1]

        sess = tf.Session(graph=self.g, config=config)
        with sess.as_default():
            with self.g.as_default():
                saver = tf.train.Saver(max_to_keep=2000)

                if start_params_dict is not None:
                    file_name = self.get_checkpoint_file(start_params_dict)
                    saver.restore(sess, file_name)
                else:
                    sess.run(tf.global_variables_initializer())
        return sess

    def plot_validation(self, t_str, n_iter, start=0):
        """
        displays validation samples of the history of training

        Args:
            t_str (string): time string for dataset ID
            n_iter (int): total number of training steps
            start (int): start plotting from iteration start
        """

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(self.dev_id) if self.dev_id else str(self.dev_ids)[1:-1]

        sess = tf.Session(graph=self.g, config=config)

        with sess.as_default():
            with self.g.as_default():
                draw_rate = ((10, 100), (100, 500), (250, 2000), (1000, 5000), (5000, 50000), (10000, 5000000))

                saver = tf.train.Saver(max_to_keep=2000)

                draw_rate_idx = 0
                prev_rate_last_train_idx = 0
                for it_i in range(n_iter):
                    rate_par = draw_rate[draw_rate_idx] if draw_rate_idx < len(draw_rate) else draw_rate[-1]
                    rate = rate_par[0]
                    draw_it = it_i - prev_rate_last_train_idx
                    # print(rate, rate_par)

                    if (draw_it % rate == 0 or it_i == n_iter - 1) and it_i >= start:
                        start_params_dict = {'time': t_str, 'idx': it_i+1}
                        file_name = self.get_checkpoint_file(start_params_dict)
                        if not os.path.exists(file_name + '.index'):
                            print("checkpoint %s doesn't exist" % file_name)
                            break
                        saver.restore(sess, file_name)
                        self.draw_val_sample(sess)

                    if draw_rate_idx < len(draw_rate) - 1 and it_i >= rate_par[1]:
                        draw_rate_idx = draw_rate_idx + 1
                        prev_rate_last_train_idx = it_i
                    if it_i == n_iter - 1:
                        start_params_dict = {'time': t_str, 'idx': it_i+1}
                        file_name = self.get_checkpoint_file(start_params_dict)
                        if not os.path.exists(file_name + '.index'):
                            print("checkpoint %s doesn't exist" % file_name)
                            break
                        saver.restore(sess, file_name)
                        self.draw_val_sample(sess)

    def get_IoU_validation(self, t_str, n_iter, iou_tensors, start=0):
        """
        evaluates IoU displays over all validation set

        Args:
            t_str (string): time string for dataset ID
            n_iter (int): total number of training steps
            iou_tensors(list): list of tensor values to be evaluated, per run mode
            start (int): start plotting from iteration start
        """

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(self.dev_id) if self.dev_id else str(self.dev_ids)[1:-1]

        sess = tf.Session(graph=self.g, config=config)

        iou = {}

        with sess.as_default():
            with self.g.as_default():
                draw_rate = ((10, 100), (100, 500), (250, 2000), (1000, 5000), (5000, 50000), (10000, 5000000))

                saver = tf.train.Saver(max_to_keep=2000)

                draw_rate_idx = 0
                prev_rate_last_train_idx = 0
                for it_i in range(n_iter):
                    rate_par = draw_rate[draw_rate_idx] if draw_rate_idx < len(draw_rate) else draw_rate[-1]
                    rate = rate_par[0]
                    draw_it = it_i - prev_rate_last_train_idx
                    # print(rate, rate_par)

                    if (draw_it % rate == 0 or it_i == n_iter - 1) and it_i >= start:
                        start_params_dict = {'time': t_str, 'idx': it_i+1}
                        file_name = self.get_checkpoint_file(start_params_dict)
                        if not os.path.exists(file_name + '.index'):
                            print("checkpoint %s doesn't exist" % file_name)
                            break
                        saver.restore(sess, file_name)
                        iou_arr = self.get_iou_arr(sess, iou_tensors)
                        iou[it_i] = iou_arr

                    if draw_rate_idx < len(draw_rate) - 1 and it_i >= rate_par[1]:
                        draw_rate_idx = draw_rate_idx + 1
                        prev_rate_last_train_idx = it_i
                    if it_i == n_iter - 1:
                        start_params_dict = {'time': t_str, 'idx': it_i+1}
                        file_name = self.get_checkpoint_file(start_params_dict)
                        if not os.path.exists(file_name + '.index'):
                            print("checkpoint %s doesn't exist" % file_name)
                            break
                        saver.restore(sess, file_name)
                        iou_arr = self.get_iou_arr(sess, iou_tensors)
                        iou[it_i] = iou_arr
        return iou

    def get_iou_arr(self, sess, iou_tensors):
        arr = []
        for r_mode in range(self.n_run_modes):
            dct = {}
            if self.train_phase_ph is not None:
                dct[self.train_phase_ph] = False
            if self.run_mode is not None:
                dct[self.run_mode] = r_mode


            arr_rm = []
            for itr in range(self.num_val_minibatches):
                if self.use_feed_dict_val:
                    st_lbl_mb = self.mb_gen_val() if self.n_run_modes == 1 else self.mb_gen_val(r_mode)
                    for tensor, value in zip(self.feed_dict_tensors, st_lbl_mb):
                        dct[tensor] = value

                iou = sess.run(iou_tensors[r_mode], feed_dict=dct)
                arr_rm.append(iou)

        arr.append(arr_rm)
        return arr

