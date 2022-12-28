import os
import ntpath
from .dataset_io_stream import dataset_io_stream as ds_ios
import pickle


def make_train_val_dataset(save_dir, description,
                           inputs_file_names, outputs_file_names,
                           normalize_inputs_outputs, binary_inputs_outputs,
                           normalization_percentile_range=None,
                           stack_depths=(180,),
                           train_tile_size=512,
                           train_tile_depth=35,
                           train_image_crop_regions=(((512, 10000), (0, 10000), (130, 180)),),
                           valid_tile_size=256,
                           valid_tile_depth=5,
                           valid_image_crop_regions=(((0, 256), (0, 10000), (130, 180)),),
                           pre_crop_xy=(((20, -20), (20, -20)),),
                           tile_depth_overlap=4):
    if os.path.exists(save_dir):
        print('dataset path "%s" already exist. Terminating' % save_dir)
        return
    else:
        os.makedirs(save_dir)

    train_ds = os.path.join(save_dir, 'train_set')
    valid_ds = os.path.join(save_dir, 'valid_set')
    dict_file = os.path.join(save_dir, 'info.pkl')
    info_file = os.path.join(save_dir, 'info.txt')

    p_crop_xy = [pre_crop_xy[0]]*len(inputs_file_names) if len(pre_crop_xy) == 1 else pre_crop_xy

    trn = ds_ios(name='train_preparer',
                 inputs_file_names=inputs_file_names,
                 outputs_file_names=outputs_file_names,
                 normalize_inputs_outputs=normalize_inputs_outputs,
                 normalization_percentile_range=normalization_percentile_range,
                 binary_inputs_outputs=binary_inputs_outputs,
                 stack_depths=stack_depths,
                 tile_size=train_tile_size, tile_depth=train_tile_depth, tile_depth_overlap=tile_depth_overlap,
                 image_crop_regions=train_image_crop_regions,
                 pre_crop_xy=p_crop_xy,
                 ds_name=train_ds)

    val = ds_ios(name='validation_preparer',
                 inputs_file_names=inputs_file_names,
                 outputs_file_names=outputs_file_names,
                 normalize_inputs_outputs=normalize_inputs_outputs,
                 normalization_percentile_range=normalization_percentile_range,
                 binary_inputs_outputs=binary_inputs_outputs,
                 stack_depths=stack_depths,
                 tile_size=valid_tile_size, tile_depth=valid_tile_depth, tile_depth_overlap=tile_depth_overlap,
                 image_crop_regions=valid_image_crop_regions,
                 pre_crop_xy=p_crop_xy,
                 ds_name=valid_ds)

    save_info_dict = {
        'tra_set': ntpath.basename(trn.file_name),
        'tra_tile_size': train_tile_size,
        'tra_tile_depth': train_tile_depth,
        'tra_batch_size': trn.n_tiles,

        'val_set': ntpath.basename(val.file_name),
        'val_tile_size': valid_tile_size,
        'val_tile_depth': valid_tile_depth,
        'val_batch_size': val.n_tiles,
    }

    print(save_info_dict, description)

    with open(dict_file, 'wb') as f:
        pickle.dump(save_info_dict, f, pickle.HIGHEST_PROTOCOL)
    with open(info_file, 'w') as f:
        f.write(description)


def load_train_val_dataset(dataset_dir, preproc_devs=('/gpu:0',),
                           graph=None,
                           minibatch_size=2,
                           shuffle_output=True,
                           strm_z_stride=1,
                           use_streaming=True, augment_training=True,
                           queue_len=100, num_proc_threads=8,
                           strm_stack_shape=(5, 256, 256)):
    if not os.path.exists(dataset_dir):
        print('dataset "%s" not found. Terminating' % dataset_dir)
        return

    dict_file = os.path.join(dataset_dir, 'info.pkl')
    info_file = os.path.join(dataset_dir, 'info.txt')

    with open(info_file, 'r') as f:
        desc = f.read()
        print('dataset description: "%s"' % desc)

    with open(dict_file, 'rb') as f:
        save_info_dict = pickle.load(f)

    # print(save_info_dict)

    train_ds = os.path.join(dataset_dir, save_info_dict['tra_set'])
    valid_ds = os.path.join(dataset_dir, save_info_dict['val_set'])

    trn = ds_ios(name='train_set',
                 pkl_file_name=train_ds,
                 graph=graph,
                 streaming=use_streaming,
                 strm_z_stride=strm_z_stride,
                 minibatch_size=minibatch_size,
                 augment=augment_training,
                 queue_len=queue_len,
                 shuffle_output=shuffle_output,
                 num_proc_threads=num_proc_threads,
                 preproc_devs=preproc_devs,
                 strm_stack_shape=strm_stack_shape)

    val = ds_ios(name='validation_set',
                 pkl_file_name=valid_ds,
                 graph=trn.get_graph(),
                 streaming=use_streaming,
                 strm_z_stride=strm_z_stride,
                 minibatch_size=minibatch_size,
                 augment=False,
                 queue_len=queue_len,
                 shuffle_output=False,
                 num_proc_threads=num_proc_threads,
                 preproc_devs=preproc_devs,
                 strm_stack_shape=strm_stack_shape)

    return trn, val, desc
