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

from random import randint

import numpy as np
import os
import tensorflow as tf


def fully_connected_layer(x,
                          n_output,
                          name=None,
                          activation=None,
                          batch_norm_train_phase=None,
                          dropout_keep_prob=None,
                          bn_order='CBR',
                          brn_rd_max=None,
                          spectral_norm=False,
                          reuse=None):
    """Fully connected layer.

    Args:
        x : tf.Tensor
            Input tensor to connect
        n_output : int
            Number of output neurons
        name : None, optional
            TF Scope to apply
        activation : None, optional
            Non-linear activation function
        batch_norm_train_phase (tf.Variable): if not none, batch_norm is applied,
                and this boolean variable controls phase
        bn_order (string): 'BCR' - BatchNorm-Conv-ReLU; 'CBR' - Conv-BatchNorm-ReLU, default
        brn_rd_max(tuple of Tensor/floats): max values of r/d to be used for batch renorm
        spectral_norm (bool): flag whether to apply spectral normalization to weights

    Returns:
        h, W : tf.Tensor, tf.Tensor
            Output of the fully connected layer and the weight matrix
    """
    if len(x.get_shape()) != 2:
        x = tf.layers.flatten(x)

    n_input = x.get_shape().as_list()[1]

    with tf.variable_scope(name or "fc"):
        if bn_order == 'BCR':
            if batch_norm_train_phase is not None:
                if brn_rd_max:
                    x = batch_renorm(x, batch_norm_train_phase, name='bn',
                                     reuse=reuse, r_max=brn_rd_max[0], d_max=brn_rd_max[1])
                else:
                    x = batch_norm(x, batch_norm_train_phase, name='bn',
                                   reuse=reuse)

        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            w = tf.get_variable(
                name='W',
                shape=[n_input, n_output],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

            b = tf.get_variable(
                name='b',
                shape=[n_output],
                dtype=tf.float32,
                initializer=tf.constant_initializer(0.01))

        w_n = spectral_normalized(w, reuse=reuse) if spectral_norm else w
        h = tf.nn.bias_add(
            name='h',
            value=tf.matmul(x, w_n),
            bias=b)
        
        if bn_order == 'CBR':
            if batch_norm_train_phase is not None:
                if brn_rd_max:
                    h = batch_renorm(h, batch_norm_train_phase, name='bn',
                                     reuse=reuse, r_max=brn_rd_max[0], d_max=brn_rd_max[1])
                else:
                    h = batch_norm(h, batch_norm_train_phase, name='bn',
                                   reuse=reuse)

        if dropout_keep_prob is not None:
            h_do = tf.nn.dropout(h, dropout_keep_prob, name='dropout')
            if batch_norm_train_phase is not None:
                h = tf.cond(batch_norm_train_phase,
                               lambda : h_do,
                               lambda: h
                            )
            else:
                h = h_do

        if activation:
            h = activation(h)

        return h, w
    

def conv_3D(x, n_output_ch,
            k_w=3, k_h=3, k_d=3,
            s_x=1, s_y=1, s_z=1,
            activation=tf.nn.relu,
            batch_norm_train_phase=None,
            padding='VALID', name='conv3d', reuse=None,
            w_default=None,
            b_default=None,
            dropout_keep_prob=None,
            brn_rd_max=None,
            spectral_norm=False,
            bn_order='CBR',
            initializer=tf.contrib.layers.xavier_initializer
            ):
    """
    Helper for creating a 3d convolution operation.

    Args:
        x (tf.Tensor): Input tensor to convolve.
        n_output_ch (int): Number of filters.
        k_w (int): Kernel width
        k_h (int): Kernel height
        k_d (int): Kernel depth
        s_x (int): Width stride
        s_y (int): Height stride
        s_z (int): Depth stride
        activation (tf.Function): activation function to apply to the convolved data
        batch_norm_train_phase (tf.Variable): if not none, batch_norm is applied,
            and this boolean variable controls phase
        padding (str): Padding type: 'SAME' or 'VALID'
        name (str): Variable scope
        reuse (tf.Flag): Flag whether to use existing variable. Can be False(None), True, or tf.AUTO_REUSE
        w_default (tf.Tensor, optional): Convolution filter to be applied, instead of new variable.
        b_default (tf.Tensor, optional): Bias to be applied, instead of new variable.
        dropout_keep_prob (tf.Tensor): if not `None` dropout is applied before BN
        brn_rd_max(tuple of Tensor/floats): max values of r/d to be used for batch renorm
        spectral_norm (bool): flag whether to apply spectral normalization to weights
        bn_order (string): 'BCR' - BatchNorm-Conv-ReLU; 'CBR' - Conv-BatchNorm-ReLU, default
        initializer (function): initializer for kernel weights

    Returns:
        op (tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor): Output of activation, convolution, weights, bias
    """
    with tf.variable_scope(name or 'conv3d'):
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            w = w_default or tf.get_variable(name='W',
                                             shape=[k_d, k_h, k_w, x.get_shape()[-1], n_output_ch],
                                             initializer=initializer()
                                             )

        if bn_order == 'BCR':
            if batch_norm_train_phase is not None:
                if brn_rd_max:
                    x = batch_renorm(x, batch_norm_train_phase, name='bn',
                                     reuse=reuse, r_max=brn_rd_max[0], d_max=brn_rd_max[1])
                else:
                    x = batch_norm(x, batch_norm_train_phase, name='bn',
                                   reuse=reuse)

        w_n = spectral_normalized(w, reuse=reuse) if spectral_norm else w
        wx = tf.nn.conv3d(name='conv',
                          input=x, filter=w_n,
                          strides=[1, s_z, s_y, s_x, 1],
                          padding=padding
                          )

        b = b_default

        if batch_norm_train_phase is None or bn_order == 'BCR':
            with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                b = b or tf.get_variable(name='b',
                                         shape=[n_output_ch],
                                         initializer=tf.constant_initializer(value=0.01)
                                         )
            h = tf.nn.bias_add(name='h',
                               value=wx,
                               bias=b
                               )
        else:
            h = wx

        if bn_order == 'CBR':
            if batch_norm_train_phase is not None:
                if brn_rd_max:
                    h = batch_renorm(h, batch_norm_train_phase, name='bn',
                                     reuse=reuse, r_max=brn_rd_max[0], d_max=brn_rd_max[1])
                else:
                    h = batch_norm(h, batch_norm_train_phase, name='bn',
                                   reuse=reuse)

        if dropout_keep_prob is not None:
            h_do = tf.nn.dropout(h, dropout_keep_prob, name='dropout')
            if batch_norm_train_phase is not None:
                h = tf.cond(batch_norm_train_phase,
                               lambda : h_do,
                               lambda: h
                            )
            else:
                h = h_do
        if activation is not None:
            x = activation(h, name=activation.__name__)
        else:
            x = h

    return x, h, w, b


def max_pool_3d(x,
                k_w=2, k_h=2, k_d=2,
                s_x=2, s_y=2, s_z=2,
                padding='VALID', name='mp3d', reuse=None
                ):
    """
    Helper for creating a 3d max pooling operation.

    Args:
        x (tf.Tensor): Input tensor to convolve.
        k_w (int): Kernel width
        k_h (int): Kernel height
        k_d (int): Kernel depth
        s_x (int): Width stride
        s_y (int): Height stride
        s_z (int): Depth stride
        padding (str): Padding type: 'SAME' or 'VALID'
        name (str): Variable scope
        reuse (tf.Flag): Flag whether to use existing variable. Can be False(None), True, or tf.AUTO_REUSE

    Returns:
        op (tf.Tensor): Output of MP
    """
    with tf.name_scope(name or 'mp3d'):
        xmp = tf.nn.max_pool3d(name='MP',
                               input=x,
                               ksize=[1, k_d, k_h, k_w, 1],
                               strides=[1, s_z, s_y, s_x, 1],
                               padding=padding
                               )
    return xmp


def avg_pool_3d(x,
                k_w=2, k_h=2, k_d=2,
                s_x=2, s_y=2, s_z=2,
                padding='VALID', name='ap3d', reuse=None
                ):
    """
    Helper for creating a 3d max pooling operation.

    Args:
        x (tf.Tensor): Input tensor to convolve.
        k_w (int): Kernel width
        k_h (int): Kernel height
        k_d (int): Kernel depth
        s_x (int): Width stride
        s_y (int): Height stride
        s_z (int): Depth stride
        padding (str): Padding type: 'SAME' or 'VALID'
        name (str): Variable scope
        reuse (tf.Flag): Flag whether to use existing variable. Can be False(None), True, or tf.AUTO_REUSE

    Returns:
        op (tf.Tensor): Output of MP
    """
    with tf.name_scope(name or 'ap3d'):
        xmp = tf.nn.avg_pool3d(name='AP',
                               input=x,
                               ksize=[1, k_d, k_h, k_w, 1],
                               strides=[1, s_z, s_y, s_x, 1],
                               padding=padding
                               )
    return xmp


def conv_2D(x, n_output_ch,
            k_w=3, k_h=3,
            s_x=1, s_y=1,
            dilation=1,
            activation=tf.nn.relu,
            batch_norm_train_phase=None,
            padding='VALID', name='conv2d', reuse=None,
            w_default=None,
            b_default=None,
            dropout_keep_prob=None,
            brn_rd_max=None,
            spectral_norm=False,
            bn_order='CBR',
            initializer=tf.contrib.layers.xavier_initializer
            ):
    """
    Helper for creating a 3d convolution operation.

    Args:
        x (tf.Tensor): Input tensor to convolve.
        n_output_ch (int): Number of filters.
        k_w (int): Kernel width
        k_h (int): Kernel height
        s_x (int): Width stride
        s_y (int): Height stride
        dilation (int): dilation factor
        activation (tf.Function): activation function to apply to the convolved data
        batch_norm_train_phase (tf.Variable): if not none, batch_norm is applied,
            and this boolean variable controls phase
        padding (str): Padding type: 'SAME' or 'VALID'
        name (str): Variable scope
        reuse (tf.Flag): Flag whether to use existing variable. Can be False(None), True, or tf.AUTO_REUSE
        w_default (tf.Tensor, optional): Convolution filter to be applied, instead of new variable.
        b_default (tf.Tensor, optional): Bias to be applied, instead of new variable.
        dropout_keep_prob (tf.Tensor): if not `None` dropout is applied before BN
        brn_rd_max(tuple of Tensor/floats): max values of r/d to be used for batch renorm
        spectral_norm (bool): flag whether to apply spectral normalization to weights
        bn_order (string): 'BCR' - BatchNorm-Conv-ReLU; 'CBR' - Conv-BatchNorm-ReLU, default
        initializer (function): initializer for kernel weights

    Returns:
        op (tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor): Output of activation, convolution, weights, bias
    """
    with tf.variable_scope(name or 'conv2d'):
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            w = w_default or tf.get_variable(name='W',
                                             shape=[k_h, k_w, x.get_shape()[-1], n_output_ch],
                                             initializer=initializer()
                                             )

        # if dilation != 1:
        #    print("WARNING: this net_utils is adapted to a BAD TF version: wrong dilation order. \
        #    won't work soon (tf. 1.10)")

        MIN_TF_VERSION_MAJOR = 1
        MIN_TF_VERSION_MINOR = 10
        tf_ver = [int(s) for s in tf.__version__.split('.')]
        is_old_dilation = tf_ver[0] < MIN_TF_VERSION_MAJOR or \
                          (tf_ver[0] == MIN_TF_VERSION_MAJOR and tf_ver[1] < MIN_TF_VERSION_MINOR)

        if is_old_dilation and dilation != 1:
            print("WARNING: tf version is %s, a BAD TF version: wrong dilation order, corrected" % tf.__version__)

        dilation_arr = [1, 1, dilation, dilation] if is_old_dilation else [1, dilation, dilation, 1]

        if bn_order == 'BCR':
            if batch_norm_train_phase is not None:
                if brn_rd_max:
                    x = batch_renorm(x, batch_norm_train_phase, name='bn',
                                     reuse=reuse, r_max=brn_rd_max[0], d_max=brn_rd_max[1])
                else:
                    x = batch_norm(x, batch_norm_train_phase, name='bn',
                                   reuse=reuse)

        w_n = spectral_normalized(w, reuse=reuse) if spectral_norm else w
        wx = tf.nn.conv2d(name='conv',
                          input=x, filter=w_n,
                          strides=[1, s_y, s_x, 1],
                          dilations=dilation_arr,
                          data_format='NHWC',
                          padding=padding
                          )

        b = b_default

        if batch_norm_train_phase is None or bn_order == 'BCR':
            with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                b = b or tf.get_variable(name='b',
                                         shape=[n_output_ch],
                                         initializer=tf.constant_initializer(value=0.01)
                                         )
            h = tf.nn.bias_add(name='h',
                               value=wx,
                               bias=b
                               )
        else:
            h = wx

        if bn_order == 'CBR':
            if batch_norm_train_phase is not None:
                if brn_rd_max:
                    h = batch_renorm(h, batch_norm_train_phase, name='bn',
                                     reuse=reuse, r_max=brn_rd_max[0], d_max=brn_rd_max[1])
                else:
                    h = batch_norm(h, batch_norm_train_phase, name='bn',
                                   reuse=reuse)

        if dropout_keep_prob is not None:
            h_do = tf.nn.dropout(h, dropout_keep_prob, name='dropout')
            if batch_norm_train_phase is not None:
                h = tf.cond(batch_norm_train_phase,
                               lambda : h_do,
                               lambda: h
                            )
            else:
                h = h_do
        if activation is not None:
            x = activation(h, name=activation.__name__)
        else:
            x = h

    return x, h, w, b


def max_pool_2d(x,
                k_w=2, k_h=2,
                s_x=2, s_y=2,
                padding='VALID', name='mp2d', reuse=None
                ):
    """
    Helper for creating a 3d max pooling operation.

    Args:
        x (tf.Tensor): Input tensor to convolve.
        k_w (int): Kernel width
        k_h (int): Kernel height
        s_x (int): Width stride
        s_y (int): Height stride
        padding (str): Padding type: 'SAME' or 'VALID'
        name (str): Variable scope
        reuse (tf.Flag): Flag whether to use existing variable. Can be False(None), True, or tf.AUTO_REUSE

    Returns:
        op (tf.Tensor): Output of MP
    """
    with tf.name_scope(name or 'mp2d'):
        xmp = tf.nn.max_pool(name='MP',
                             value=x,
                             ksize=[1, k_h, k_w, 1],
                             strides=[1, s_y, s_x, 1],
                             padding=padding
                             )
    return xmp


def avg_pool_2d(x,
                k_w=2, k_h=2,
                s_x=2, s_y=2,
                padding='VALID', name='ap2d', reuse=None
                ):
    """
    Helper for creating a 3d max pooling operation.

    Args:
        x (tf.Tensor): Input tensor to convolve.
        k_w (int): Kernel width
        k_h (int): Kernel height
        s_x (int): Width stride
        s_y (int): Height stride
        padding (str): Padding type: 'SAME' or 'VALID'
        name (str): Variable scope
        reuse (tf.Flag): Flag whether to use existing variable. Can be False(None), True, or tf.AUTO_REUSE

    Returns:
        op (tf.Tensor): Output of MP
    """
    with tf.name_scope(name or 'ap2d'):
        xmp = tf.nn.avg_pool(name='AP',
                             value=x,
                             ksize=[1, k_h, k_w, 1],
                             strides=[1, s_y, s_x, 1],
                             padding=padding
                             )
    return xmp


def get_linear_upsampling_vector(stride):
    """
    Helper for creating a linear upsampling filter.
    
    Args:
        stride (int): upsampling ratio. Can be either 2 or 1.
    
    Returns:
        arr (np.array): upsampling array
    """
    return np.array((0.5, 1., 0.5)) if stride == 2 else np.array((1,))


def get_linear_upsampling_kernel(s_x=2, s_y=2, s_z=1):
    """
    Helper for creating a three-linear filter in 3D.

    Parameters
    ----------
    s_x (int): Width stride
    s_y (int): Height stride
    s_z (int): Depth stride

    Returns
    -------
    arr (np.ndarray): convolution matrix for the transpose 3D convolution
    """
    n_x = (s_x - 1) * 2 + 1
    n_y = (s_y - 1) * 2 + 1
    n_z = (s_z - 1) * 2 + 1
    vx = get_linear_upsampling_vector(s_x).reshape((1, 1, n_x))
    vy = get_linear_upsampling_vector(s_y).reshape((1, n_y, 1))
    vz = get_linear_upsampling_vector(s_z).reshape((1, n_z, 1))
    m_xy = np.matmul(vy, vx)
    m_xyt = m_xy.transpose((1, 0, 2))
    m_xyzt = np.matmul(vz, m_xyt)
    m_xyz = m_xyzt.transpose((1, 0, 2))
    return m_xyz


def upscale_3D(x,
               n_output_ch,
               s_x=2,
               s_y=2,
               s_z=1,
               activation=None,
               trainable=True,
               name='tconv3d',
               padding='VALID',
               batch_norm_train_phase=None,
               bn_order='CBR',
               brn_rd_max=None,
               spectral_norm=False,
               reuse=None):
    """
    Helper for creating a 3D upsampling operation of factor 2.
    Kernel size is obtained from strides;

    Args:
        x (tf.Tensor): Input tensor to convolve.
        n_output_ch (int): Number of filters.
        s_x (int): Width stride
        s_y (int): Height stride
        s_z (int): depth stride
        activation (tf.Function): activation function to be applied
        trainable (bool): Flag whether upsampling filters should be adjusted
        padding (str): Padding type: 'SAME' or 'VALID'
        name (str): Variable scope
        batch_norm_train_phase (tf.Variable): if not none, batch_norm is applied,
            and this boolean variable controls phase
        bn_order (string): 'BCR' - BatchNorm-Conv-ReLU; 'CBR' - Conv-BatchNorm-ReLU, default
        brn_rd_max(tuple of Tensor/floats): max values of r/d to be used for batch renorm
        spectral_norm (bool): flag whether to apply spectral normalization to weights
        reuse (tf.Flag): Flag whether to use existing variable. Can be False(None), True, or tf.AUTO_REUSE

    Returns:
        op (tf.Tensor, tf.Tensor, tf.Tensor): Output of activation, Output of convolution, weights
    """

    wi = get_linear_upsampling_kernel(s_x, s_y, s_z)
    k_w = (s_x - 1) * 2 + 1
    k_h = (s_y - 1) * 2 + 1
    k_d = (s_z - 1) * 2 + 1

    n_input_ch = x.get_shape().as_list()[-1]
    ish = tf.shape(x)
    batch_size = ish[0]
    shz = ish[1]
    shy = ish[2]
    shx = ish[3]
    osh = tf.stack([batch_size, shz * s_z, shy * s_y, shx * s_x, n_output_ch])

    with tf.variable_scope(name or 'tconv3d'):
        if bn_order == 'BCR':
            if batch_norm_train_phase is not None:
                if brn_rd_max:
                    x = batch_renorm(x, batch_norm_train_phase, name='bn',
                                     reuse=reuse, r_max=brn_rd_max[0], d_max=brn_rd_max[1])
                else:
                    x = batch_norm(x, batch_norm_train_phase, name='bn',
                                   reuse=reuse)

        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            w = tf.get_variable(name='W',
                                shape=[k_d, k_h, k_w, n_output_ch, n_input_ch],
                                initializer=tf.constant_initializer(value=wi, dtype=tf.float32),
                                dtype=tf.float32, trainable=trainable)

        w_n = spectral_normalized(w, reuse=reuse) if spectral_norm else w
        wx = tf.nn.conv3d_transpose(
            name='upscale',
            value=x,
            filter=w_n,
            output_shape=osh,
            strides=[1, s_z, s_y, s_x, 1],
            padding=padding)

        if bn_order == 'CBR' and batch_norm_train_phase is not None:
            if brn_rd_max:
                h = batch_renorm(wx, batch_norm_train_phase, name='bn',
                                 reuse=reuse, r_max=brn_rd_max[0], d_max=brn_rd_max[1])
            else:
                h = batch_norm(wx, batch_norm_train_phase, name='bn',
                               reuse=reuse)
        else:
            h = wx

        if activation is not None:
            x = activation(h, name=activation.__name__)
        else:
            x = wx
    return x, wx, w


def get_linear_upsampling_kernel2d(s_x=2, s_y=2):
    """
    Helper for creating a three-linear filter in 3D.

    Parameters
    ----------
    s_x (int): Width stride
    s_y (int): Height stride

    Returns
    -------
    arr (np.ndarray): convolution matrix for the transpose 3D convolution
    """
    n_x = (s_x - 1) * 2 + 1
    n_y = (s_y - 1) * 2 + 1
    vx = get_linear_upsampling_vector(s_x).reshape((1, n_x))
    vy = get_linear_upsampling_vector(s_y).reshape((n_y, 1))
    m_xy = np.matmul(vy, vx)
    # print(m_xy)
    return m_xy


def upscale_2D(x,
               n_output_ch,
               s_x=2,
               s_y=2,
               activation=None,
               trainable=True,
               name='tconv2d',
               padding='VALID',
               batch_norm_train_phase=None,
               bn_order='CBR',
               brn_rd_max=None,
               spectral_norm=False,
               reuse=None):
    """
    Helper for creating a 2D upsampling operation of factor 2.
    Kernel size is obtained from strides;

    Args:
        x (tf.Tensor): Input tensor to convolve.
        n_output_ch (int): Number of filters.
        s_x (int): Width stride
        s_y (int): Height stride
        activation (tf.Function): activation function to be applied
        trainable (bool): Flag whether upsampling filters should be adjusted
        padding (str): Padding type: 'SAME' or 'VALID'
        name (str): Variable scope
        batch_norm_train_phase (tf.Variable): if not none, batch_norm is applied,
            and this boolean variable controls phase
        bn_order (string): 'BCR' - BatchNorm-Conv-ReLU; 'CBR' - Conv-BatchNorm-ReLU, default
        brn_rd_max(tuple of Tensor/floats): max values of r/d to be used for batch renorm
        spectral_norm (bool): flag whether to apply spectral normalization to weights
        reuse (tf.Flag): Flag whether to use existing variable. Can be False(None), True, or tf.AUTO_REUSE

    Returns:
        op (tf.Tensor, tf.Tensor, tf.Tensor): Output of activation, Output of convolution, weights
    """

    wi = get_linear_upsampling_kernel2d(s_x, s_y)
    k_w = (s_x - 1) * 2 + 1
    k_h = (s_y - 1) * 2 + 1

    n_input_ch = x.get_shape().as_list()[-1]
    ish = tf.shape(x)
    batch_size = ish[0]
    shy = ish[1]
    shx = ish[2]
    osh = tf.stack([batch_size, shy * s_y, shx * s_x, n_output_ch])

    with tf.variable_scope(name or 'tconv2d'):
        if bn_order == 'BCR':
            if batch_norm_train_phase is not None:
                if brn_rd_max:
                    x = batch_renorm(x, batch_norm_train_phase, name='bn',
                                     reuse=reuse, r_max=brn_rd_max[0], d_max=brn_rd_max[1])
                else:
                    x = batch_norm(x, batch_norm_train_phase, name='bn',
                                   reuse=reuse)

        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            w = tf.get_variable(name='W',
                                shape=[k_h, k_w, n_output_ch, n_input_ch],
                                initializer=tf.constant_initializer(value=wi, dtype=tf.float32),
                                dtype=tf.float32, trainable=trainable)

        w_n = spectral_normalized(w, reuse=reuse) if spectral_norm else w
        wx = tf.nn.conv2d_transpose(
            name='upscale',
            value=x,
            filter=w_n,
            output_shape=osh,
            strides=[1, s_y, s_x, 1],
            padding=padding)
        # print(w, wx, batch_size, ish[1] * s_y, ish[2] * s_x, n_output_ch)

        if bn_order == 'CBR' and batch_norm_train_phase is not None:
            if brn_rd_max:
                h = batch_renorm(wx, batch_norm_train_phase, name='bn',
                                 reuse=reuse, r_max=brn_rd_max[0], d_max=brn_rd_max[1])
            else:
                h = batch_norm(wx, batch_norm_train_phase, name='bn',
                               reuse=reuse)
        else:
            h = wx

        if activation is not None:
            x = activation(h, name=activation.__name__)
        else:
            x = wx
    return x, wx, w


def softmax(x, name='softmax', reuse=None):
    """
    Helper for creating a softmax operation.

    Args:
        x (tf.Tensor): Input tensor to convolve.
        name (str): Variable scope
        reuse (tf.Flag): Flag whether to use existing variable. Can be False(None), True, or tf.AUTO_REUSE

    Returns:
        xmp (tf.Tensor): Output of softmax
    """

    with tf.name_scope(name):
        xmp = tf.nn.softmax(name='SM', logits=x, axis=-1)

    return xmp


def sigmoid(x, name='sigmoid', reuse=None):
    """
    Helper for creating a sigmoid operation.

    Args:
        x (tf.Tensor): Input tensor to convolve.
        name (str): Variable scope
        reuse (tf.Flag): Flag whether to use existing variable. Can be False(None), True, or tf.AUTO_REUSE

    Returns:
        xmp (tf.Tensor): Output of sigmoid
    """

    with tf.name_scope(name):
        xmp = tf.nn.sigmoid(name='SGM', x=x)

    return xmp


def get_label_strided_3d(labels, strides_xyz=(4, 4, 2), ofs_xyz=(22, 22, 11)):
    """
    Helper for obtaining label crop corresponding to net output.

    Args:
        labels (np.ndarray): Input label/stack of outputs.
        strides_xyz (int, int, int): strides along each of 3 dimension
        ofs_xyz (int, int, int): offsets wrt original image for cropping

    Returns:
        strided (np.ndarray): cropped and strided label
    """
    sh = labels.shape
    strided = labels[...,
                     ofs_xyz[2]: sh[-3] - ofs_xyz[2]:strides_xyz[2],
                     ofs_xyz[1]: sh[-2] - ofs_xyz[1]:strides_xyz[1],
                     ofs_xyz[0]: sh[-1] - ofs_xyz[0]:strides_xyz[0]]
    return strided


def get_stack_and_label_random_crop_3d(stack, lbl, stack_shape=(256, 256, 40), lbl_strides=(4, 4, 2),
                                       lbl_ofs=(22, 22, 11), nb=1):
    """
    Helper for obtaining random crops from original data for input and outputs.

    Args:
        stack (np.ndarray): Input data/stack of data.
        lbl (np.ndarray): Input label/stack of outputs.
        stack_shape (int, int, int): shape of the crops to be produced
        lbl_strides (int, int, int): strides along each of 3 dimension
        lbl_ofs (int, int, int): offsets wrt original image for cropping
        nb (int): number of stacks per minibatch

    Returns:
        stack, lbl_str (np.ndarray, np.ndarray): randomly cropped data and corresponding label
    """
    if nb > 1:
        s0, l0 = get_stack_and_label_random_crop_3d(stack, lbl, stack_shape, lbl_strides, lbl_ofs, 1)
        for i in range(nb - 1):
            si, li = get_stack_and_label_random_crop_3d(stack, lbl, stack_shape, lbl_strides, lbl_ofs, 1)
            s0 = np.concatenate((s0, si), axis=0)
            l0 = np.concatenate((l0, li), axis=0)
        return s0, l0

    fsh = stack.shape[1:-1][::-1]
    ofs_max = np.asarray(fsh) - np.asarray(stack_shape)
    ofs = [randint(0, o) for o in ofs_max]
    stack = stack[:, ofs[2]:ofs[2] + stack_shape[2],
                  ofs[1]:ofs[1] + stack_shape[1],
                  ofs[0]:ofs[0] + stack_shape[0], :]
    lbl = lbl[:, ofs[2]:ofs[2] + stack_shape[2],
              ofs[1]:ofs[1] + stack_shape[1],
              ofs[0]:ofs[0] + stack_shape[0]]
    lbl_str = np.asarray(get_label_strided_3d(lbl, lbl_strides, lbl_ofs), dtype=np.int32)
    return stack, lbl_str


def batch_norm_old(x, phase_train, name='bn', decay=0.9, reuse=None,
               affine=True):
    """
    Batch normalization on convolutional maps.
    from: https://stackoverflow.com/questions/33949786/how-could-i-
    use-batch-normalization-in-tensorflow
    Modified to infer shape from input tensor x, accept any input size
    and use the new function interfaces

    Args:
        x (tf.Tensor): 5D BDHWC, 4D BHWC or 2D BC input maps.
        phase_train (bool,tf.Variable): true indicates training phase
        name (str): variable name
        decay (float): exponential moving average constant
        reuse (bool): whether to reuse beta/gama variables
        affine (bool): whether to affine-transform outputs

    Return:
        normed (tf.Tensor): batch-normalized feature maps

    """
    with tf.variable_scope(name):
        shape = x.get_shape().as_list()

        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            beta = tf.get_variable(name='beta', shape=[shape[-1]],
                                   initializer=tf.constant_initializer(0.0),
                                   trainable=True)
            gamma = tf.get_variable(name='gamma', shape=[shape[-1]],
                                    initializer=tf.constant_initializer(1.0),
                                    trainable=affine) if affine else None

        averaging_axes = list(range(len(shape) - 1))
        batch_mean, batch_var = tf.nn.moments(x, averaging_axes, name='moments')

        ema = tf.train.ExponentialMovingAverage(decay=decay)

        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            ema_apply_op = ema.apply([batch_mean, batch_var])

        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema_mean, ema_var))

        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-6)
    return normed


def batch_norm(x, phase_train, name='bn', decay=0.99, reuse=None,
               affine=True
               ):
    """
    Batch normalization on convolutional maps.
    from: https://stackoverflow.com/questions/33949786/how-could-i-
    use-batch-normalization-in-tensorflow
    Modified to infer shape from input tensor x, accept any input size
    and use the new function interfaces

    Args:
        x (tf.Tensor): 5D BDHWC, 4D BHWC or 2D BC input maps.
        phase_train (bool,tf.Variable): true indicates training phase
        name (str): variable name
        decay (float): exponential moving average constant
        reuse (bool): whether to reuse beta/gama variables
        affine (bool): whether to affine-transform outputs
        train_decay(bool): exponential moving average constant for training or None to use single batch values

    Return:
        normed (tf.Tensor): batch-normalized feature maps

    """
    with tf.variable_scope(name):
        shape = x.get_shape().as_list()
        n_channels = shape[-1]

        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            beta = tf.get_variable(name='beta', shape=[n_channels],
                                   initializer=tf.constant_initializer(0.0),
                                   trainable=True)
            gamma = tf.get_variable(name='gamma', shape=[n_channels],
                                    initializer=tf.constant_initializer(1.0),
                                    trainable=affine) if affine else None

        averaging_axes = list(range(len(shape) - 1))
        batch_mean, batch_var = tf.nn.moments(x, averaging_axes, name='moments')

        ema = tf.train.ExponentialMovingAverage(decay=decay)

        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            ema_apply_op = ema.apply([batch_mean, batch_var])

        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

        def mean_var_with_update():
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                # following operations must be defined in function, otherwise
                # they are executed prior to the tf.cond call, irrelevant of the condition
                ema_apply_op_x = ema.apply([batch_mean, batch_var])

            dependent_ops = [ema_apply_op_x]
            with tf.control_dependencies(dependent_ops):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema_mean, ema_var))

        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-6)

    return normed  # , mean, var, ema_mean, ema_var, beta, gamma


def batch_renorm(x, phase_train, name='bn', decay=0.99, reuse=None,
               affine=True,
               r_max=None,
               d_max=None
               ):
    """
    Batch renormalization on convolutional maps.
    https://arxiv.org/pdf/1702.03275.pdf

    Args:
        x (tf.Tensor): 5D BDHWC, 4D BHWC or 2D BC input maps.
        phase_train (bool,tf.Variable): true indicates training phase
        name (str): variable name
        decay (float): exponential moving average constant
        reuse (bool): whether to reuse beta/gama variables
        affine (bool): whether to affine-transform outputs
        r_max(tf.Tensor): Tensor or float controlling range of the r factor 
        d_max(tf.Tensor): Tensor or float controlling range of the d factor

    Return:
        normed (tf.Tensor): batch-normalized feature maps

    """
    with tf.variable_scope(name):
        shape = x.get_shape().as_list()
        n_channels = shape[-1]

        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            beta = tf.get_variable(name='beta', shape=[n_channels],
                                   initializer=tf.constant_initializer(0.0),
                                   trainable=True)
            gamma = tf.get_variable(name='gamma', shape=[n_channels],
                                    initializer=tf.constant_initializer(1.0),
                                    trainable=affine) if affine else None

        averaging_axes = list(range(len(shape) - 1))
        batch_mean, batch_var = tf.nn.moments(x, averaging_axes, name='moments')

        ema = tf.train.ExponentialMovingAverage(decay=decay)

        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            ema_apply_op = ema.apply([batch_mean, batch_var])

        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

        def mean_var_with_update():
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                # following operations must be defined in function, otherwise
                # they are executed prior to the tf.cond call, irrelevant of the condition
                ema_apply_op_x = ema.apply([batch_mean, batch_var])

            with tf.control_dependencies([ema_apply_op_x]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema_mean, ema_var))

        nd_sh = [1] * (len(shape)-1) + [-1]
        gamma_nd = tf.reshape(gamma, nd_sh)
        beta_nd = tf.reshape(beta, nd_sh)

        def get_renormed():
            r_m = r_max or 2
            d_m = d_max or 1

            r2 = batch_var / ema_var
            d = (batch_mean - ema_mean) / tf.sqrt(tf.clip_by_value(ema_var, 1e-6, 1e6))

            r = tf.stop_gradient(tf.sqrt(tf.clip_by_value(r2, 1/r_m, r_m)))
            d = tf.stop_gradient(tf.clip_by_value(d, -d_m, d_m))

            x_n = tf.nn.batch_normalization(x, mean, var, 0, 1, 1e-6)
            x_n = x_n * r + d
            x_n = x_n * gamma_nd + beta_nd
            return x_n

        def get_normed():
            x_n = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-6)
            return x_n

        normed = tf.cond(phase_train, get_renormed, get_normed)


    return normed  # , mean, var, ema_mean, ema_var, beta, gamma


def batch_renorm_test1(x, phase_train, name='bn', decay=0.99, reuse=None,
               affine=True,
               train_decay=None
               ):
    """
    Batch normalization on convolutional maps.
    from: https://stackoverflow.com/questions/33949786/how-could-i-
    use-batch-normalization-in-tensorflow
    Modified to infer shape from input tensor x, accept any input size
    and use the new function interfaces

    Args:
        x (tf.Tensor): 5D BDHWC, 4D BHWC or 2D BC input maps.
        phase_train (bool,tf.Variable): true indicates training phase
        name (str): variable name
        decay (float): exponential moving average constant
        reuse (bool): whether to reuse beta/gama variables
        affine (bool): whether to affine-transform outputs
        train_decay(bool): exponential moving average constant for training or None to use single batch values

    Return:
        normed (tf.Tensor): batch-normalized feature maps

    """
    with tf.variable_scope(name):
        shape = x.get_shape().as_list()
        n_channels = shape[-1]

        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            beta = tf.get_variable(name='beta', shape=[n_channels],
                                   initializer=tf.constant_initializer(0.0),
                                   trainable=True)
            gamma = tf.get_variable(name='gamma', shape=[n_channels],
                                    initializer=tf.constant_initializer(1.0),
                                    trainable=affine) if affine else None

        averaging_axes = list(range(len(shape) - 1))
        batch_mean, batch_var = tf.nn.moments(x, averaging_axes, name='moments')

        ema = tf.train.ExponentialMovingAverage(decay=decay)
        ema_tr = tf.train.ExponentialMovingAverage(decay=train_decay) if train_decay is not None else None

        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            ema_apply_op = ema.apply([batch_mean, batch_var])
            ema_apply_op_tr = ema_tr.apply([batch_mean, batch_var]) if ema_tr else None

        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
        ema_mean_tr, ema_var_tr = (ema_tr.average(batch_mean),
                                   ema_tr.average(batch_var)) if ema_tr else (batch_mean, batch_var)

        def mean_var_with_update():
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                # following operations must be defined in function, otherwise
                # they are executed prior to the tf.cond call, irrelevant of the condition
                ema_apply_op_x = ema.apply([batch_mean, batch_var])
                ema_apply_op_tr_x = ema_tr.apply([batch_mean, batch_var]) if ema_tr else None

            dependent_ops = [ema_apply_op_x, ema_apply_op_tr_x] if ema_apply_op_tr_x else [ema_apply_op_x]
            with tf.control_dependencies(dependent_ops):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema_mean, ema_var))

        if ema_tr:
            def get_renormed():
                r = tf.stop_gradient(tf.sqrt(tf.clip_by_value(batch_var / ema_var_tr, 0.5, 2))) if ema_tr else 1
                d = tf.stop_gradient(tf.clip_by_value((batch_mean - ema_var_tr) /
                                                      tf.sqrt(tf.clip_by_value(ema_var_tr, 1e-6, 1e6)),
                                                      -1, 1
                                                      )) if ema_tr else 0

                normed = tf.nn.batch_normalization(x, mean, var, 0, 1, 1e-6)
                normed = normed * r + d
                normed = normed * tf.reshape(gamma, [1] * (len(shape)-1)+[-1]) + tf.reshape(beta, [1] * (len(shape)-1)+[-1])
                return normed

            def get_normed():
                normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-6)
                return normed

            normed = tf.cond(phase_train, get_renormed, get_normed)

            # return usual upon test
        else:
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-6)

    return normed  # , mean, var, ema_mean, ema_var, beta, gamma


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_normalized(w, iteration=1, reuse=None):
    with tf.name_scope('SN'):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])

        with tf.variable_scope('SN', reuse=reuse):
            u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

        u_hat = u
        v_hat = None
        for i in range(iteration):
            """
            power iteration
            Usually iteration = 1 will be enough
            """
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = l2_norm(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = l2_norm(u_)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
        w_norm = w / sigma

        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_norm, w_shape)

        return w_norm


def get_receptive_field_size(model_file, in_tensor_name, out_tensor_name, gpu_id):
    """
    Obtains receptive field by computing gradient of input image wrt single pixel.

    Args:
        model_file (str): model checkpoint name, eg 'models/model_A/model-1000'
        in_tensor_name (str): input image tensor name, eg 'stack:0'
        out_tensor_name (str): output tensor name, eg 'y:0'
        gpu_id (int): gpu ID to run calculations on

    Returns:
        radius_horizontal_left, radius_horizontal_right, radius_vertical_up, radius_vertical_down (int,int,int,int):
            receptive field radii around center pixel (not included)
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(gpu_id)

    g = tf.Graph()
    with tf.Session(graph=g, config=config) as sess:
        sess.run(tf.global_variables_initializer())

        _ = dir(tf.contrib)  # required for importing operations from contrib
        saver = tf.train.import_meta_graph(model_file + '.meta', clear_devices=True)

        file_name = model_file
        saver.restore(sess, file_name)

        y = g.get_tensor_by_name(out_tensor_name)
        x = g.get_tensor_by_name(in_tensor_name)
        sh = x.get_shape().as_list()
        # if sh[0] is None:
        sh[0] = 1

        gd = g.as_graph_def(add_shapes=False)

        # fix batch norm nodes
        for node in gd.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']

        gd_c = tf.graph_util.convert_variables_to_constants(sess,
                                                            gd,
                                                            output_node_names=[y.name.split(':')[0], ])

    gt = tf.Graph()
    with gt.as_default():  # new graph loads constant model and variable input
        x_t = tf.get_variable('test_x', shape=sh, initializer=tf.random_normal_initializer())

        y_t, = tf.import_graph_def(gd_c, input_map={in_tensor_name: x_t},
                                   return_elements=[out_tensor_name])

        yc = y_t[..., 128:129, 128:129, 0]  # single pixel selection

        loss = tf.reduce_mean(10 - yc)  # whatever difference
        opt = tf.train.AdamOptimizer(learning_rate=0.1)

        grad = opt.compute_gradients(loss)
        dep = tf.abs(grad[0][0])
        dep = tf.clip_by_value(dep * 100000, 0, 1)

        # upd = opt.apply_gradients(grad)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(gpu_id)

    fd = None
    t = None
    for op in gt.get_operations():
        if 'phase_train' in op.name:
            t = gt.get_tensor_by_name(op.name + ':0')
    if t is not None:
        fd = {t: False}

    gs = []
    for i in range(10):  # run 10 times to avoid random occurred zero gradients
        with tf.Session(graph=gt, config=config) as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            y_r, x_r, g_r = sess.run([y_t[..., 0], x_t, dep], feed_dict=fd)
            gs.append(g_r)

    len_sh = len(gs[0].shape)
    assert (4 <= len_sh <= 5)
    is_2d = len_sh == 4

    gs = np.concatenate(gs)
    gg = np.sum(gs, axis=(0, 3) if is_2d else (0, 1, 4)) # sum iterations+batch, C/D, BDHWC / BHWD

    ggh = np.clip(np.sum(gg, axis=0), 0, 1)
    ggv = np.clip(np.sum(gg, axis=1), 0, 1)

    def over_zero_range(arr):
        n = arr.shape[0]
        i_min = i_max = 0
        for idx in range(n):
            if arr[idx] > 0:
                i_min = idx
                break
        for j in range(n):
            idx = n - 1 - j
            if arr[idx] > 0:
                i_max = idx
                break
        return i_min, i_max

    rh = over_zero_range(ggh)
    rv = over_zero_range(ggv)

    r = (128 - rh[0], rh[1] - 128, 128 - rv[0], rv[1] - 128)

    # print('rh-=', 128-rh[0])
    # print('rh+=', rh[1]-128)
    # print('rv-=', 128-rv[0])
    # print('rv+=', rv[1]-128)

    return r


def get_timed_model_checkpoint(time_str, checkpoint_idx, model_name):
    cpf = 'model_%s_%s/model-%d' % (
        model_name,
        time_str,
        checkpoint_idx)
    return cpf


def get_checkpoint_file(start_params_dict, model_name):
    root = start_params_dict['root'] if 'root' in start_params_dict else None
    if root is None:
        raise ValueError('rood directory must be provided')
    file_name = get_timed_model_checkpoint(start_params_dict['time'],
                                           start_params_dict['idx'],
                                           model_name)
    file_name = os.path.join(root, file_name)
    return file_name


def load_model_graph(start_params_dict, model_name, dev_id, dev_ids=None):
    """
    Obtains receptive field by computing gradient of input image wrt single pixel.

    Args:
        start_params_dict (dict): if not `None` initialize trainable variables from checkpoint.
                                          'time'-> time string, e.g. '2018.03.19_16-28'
                                          'idx' -> checkpoint index, e.g. 50000
                                          'root' -> root dir of models, optional
        model_name (str): model name
        dev_id (int): CUDA device id to run on
        dev_ids (int, int,...): list of CUDA device ids to run on. If set, dev_id should be None

    Returns:
        graph (tf.Graph): read graph
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(dev_id) if dev_id else str(dev_ids)[1:-1]

    g = tf.Graph()
    with tf.Session(graph=g, config=config) as sess:
        sess.run(tf.global_variables_initializer())

        _ = dir(tf.contrib)  # required for importing operations from contrib
        model_file = get_checkpoint_file(start_params_dict, model_name)
        saver = tf.train.import_meta_graph(model_file + '.meta')

        saver.restore(sess, model_file)

    return g


def get_num_trainable_params(graph):
    """
        Returns total number of trainable parameters

        Args:
            graph (tf.Graph): model graph

        Returns:
            num (int): number of trainable parameters
        """
    with graph.as_default():
        tv = tf.trainable_variables()
        tv_sh = [v.shape.as_list() for v in tv]
        tv_sz = []
        for sh in tv_sh:
            sz = 1
            for d in sh:
                sz = sz * d
            tv_sz.append(sz)
        tot_sz = 0
        for sz in tv_sz:
            tot_sz += sz
    return tot_sz


class LearningRate(object):
    """
    wrapper TF objects that allows easy LR control
    """

    def __init__(self, value, name='LR', reuse=None):
        """
        Args:
            value (float): initial value
            name (str): objects namespace
            reuse (tf.Flag): Flag whether to use existing variable. Can be False(None), True, or tf.AUTO_REUSE
        """

        with tf.variable_scope(name, reuse=reuse):
            self.val = value
            self.lrv = tf.get_variable(
                name='lr_v',
                shape=(),
                dtype=tf.float32,
                initializer=tf.constant_initializer(value=value, dtype=tf.float32),
                trainable=False)

            self.lrp = tf.placeholder(
                name='lr_p',
                shape=(),
                dtype=tf.float32)

            self.set_lr_op = self.lrv.assign(self.lrp)

    def get_value(self):
        """
        Returns current LR value
        
        Returns:
            lr (float): current LR value
        """
        return self.val

    def get_var(self):
        """
        Returns tf variable that holds the current LR value
        
        Returns:
            lr (tf.Tensor): current LR value variable
        """
        return self.lrv

    def set_value(self, sess, lr_value):
        """
        Sets LR value
        
        Args:
            sess (tf.Session): session, in which operation will be executed
            lr_value (float): LR value
        """
        sess.run(self.set_lr_op, feed_dict={self.lrp: lr_value})
        self.val = lr_value


def TFSession(graph, devices=(0,), limited_memory=True):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = limited_memory
    config.gpu_options.visible_device_list = str(devices[0]) if len(devices) == 1 else str(devices)[1:-1]

    return tf.Session(graph=graph, config=config)
