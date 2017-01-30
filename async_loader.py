from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

def get_multi_scale_crop_size(height, width, crop_size, scale_ratios, max_distort):
    crop_sizes = []
    base_size = min(height, width)
    for i in xrange(len(scale_ratios)):
        crop_h = int(base_size * scale_ratios[i])
        if abs(crop_h - crop_size) < 3:
            crop_h = crop_size
        for j in xrange(len(scale_ratios)):
            crop_w = int(base_size * scale_ratios[j])
            if abs(crop_w - crop_size) < 3:
                crop_w = crop_size
            # append this cropping size into the list
            if abs(j - i) <= max_distort:
                crop_sizes.append([crop_h, crop_w])
    return crop_sizes

def get_fix_offset(h, w, crop_height, crop_width):
    crop_offsets = []
    height_off = (h - crop_height) / 4
    width_off = (w - crop_width) / 4
    crop_offsets.append(tf.stack([0, 0]))
    crop_offsets.append(tf.stack([0, tf.to_int32(4 * width_off)]))
    crop_offsets.append(tf.stack([tf.to_int32(4 * height_off), 0]))
    crop_offsets.append(tf.stack([tf.to_int32(4 * height_off), tf.to_int32(4 * width_off)]))
    crop_offsets.append(tf.stack([tf.to_int32(2 * height_off), tf.to_int32(2 * width_off)]))
    # more fix crop
    crop_offsets.append(tf.stack([0, tf.to_int32(2 * width_off)]))
    crop_offsets.append(tf.stack([tf.to_int32(4 * height_off), tf.to_int32(2 * width_off)]))
    crop_offsets.append(tf.stack([tf.to_int32(2 * height_off), 0]))
    crop_offsets.append(tf.stack([tf.to_int32(2 * height_off), tf.to_int32(4 * width_off)]))
    crop_offsets.append(tf.stack([tf.to_int32(height_off), tf.to_int32(width_off)]))
    crop_offsets.append(tf.stack([tf.to_int32(height_off), tf.to_int32(3 * width_off)]))
    crop_offsets.append(tf.stack([tf.to_int32(3 * height_off), tf.to_int32(width_off)]))
    crop_offsets.append(tf.stack([tf.to_int32(3 * height_off), tf.to_int32(3 * width_off)]))

    crop_offsets = tf.stack(crop_offsets)
    return crop_offsets

def one_image(modality, image_name, offset_str, start_offset, height, width, crop,
              ho, wo, crop_size, crop_height, crop_width, preprocessing_fn, random_mirror, length=1):
    channels = 3 * length
    if modality is None:
        id = "0"
        file_contents = tf.read_file(image_name)
        image = tf.image.decode_jpeg(file_contents, channels=3)
    elif modality == 'RGB':
        images = []
        for o in xrange(length):
            id = tf.gather(offset_str, start_offset+o)
            file_contents = tf.read_file(image_name+"/flow_i_"+id+".jpg")
            image = tf.image.decode_jpeg(file_contents, channels=3)
            images.append(image)
        image = tf.concat_v2(images, 2)
    elif modality == 'flow' or modality == 'warp':
        images = []
        for o in xrange(length):
            id = tf.gather(offset_str, start_offset+o)
            file_contents = tf.read_file(image_name+"/flow_x_"+id+".jpg")
            image1 = tf.image.decode_jpeg(file_contents, channels=1)
            image1 = tf.to_float(image1)
            file_contents = tf.read_file(image_name+"/flow_y_"+id+".jpg")
            image2 = tf.image.decode_jpeg(file_contents, channels=1)
            image2 = tf.to_float(image2)
            if length <= 1:
                image3 = 0.7064*tf.sqrt(image1*image1+image2*image2)
                image = tf.concat_v2([image1, image2, image3], 2)
            else:
                image = tf.concat_v2([image1, image2], 2)
            images.append(image)
        image = tf.concat_v2(images, 2)
        if length > 1:
            channels = 2 * length
    else:
        raise NotImplementedError('Modality %s is not supported.'%modality)
    image = tf.image.resize_images(image, [height, width], method=0)
    image.set_shape([height, width, channels])
    if crop == 0:
        image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
        image.set_shape([crop_size, crop_size, channels])
    elif crop == 1 or crop == 2:
        image = tf.slice(image, tf.stack([ho, wo, 0]), tf.stack([crop_height, crop_width, -1]))
    else:
        raise NotImplementedError('Crop mode %d is not supported.'%crop)
    # augment after crop
    image = preprocessing_fn(image, crop_size, crop_size, random_mirror=random_mirror)

    return image

def read_video(name_label_length_stride_queue, offset_str, multi_scale_crop_sizes, config):
    # name: video name
    # label: video label
    # length: video length after truncate the sample length
    # stride: read stride for current video
    # offset_str: so that we don't have to convert int to formated string
    # multi_scale_crop_sizes: the multi scale crop sizes, select one to crop and resize to config['crop_size']
    # config:
    #   width, height, crop_size, n_steps, modality
    #   crop(0 for center crop, 1 for random crop, 2 for fix crop)
    #   augment(True, False)
    video_name = name_label_length_stride_queue[0]
    label = name_label_length_stride_queue[1]
    video_length = name_label_length_stride_queue[2]
    read_stride = name_label_length_stride_queue[3]

    offset = tf.random_uniform((), maxval=video_length, dtype=tf.int32)
    label = tf.to_int32(label)
    if config['merge_label']:
        labels = label
    else:
        labels = tf.fill([config['n_steps']], label)

    # mirror
    if config['mirror']:
        mirror = tf.less(tf.random_uniform([], 0, 1.0), 0.5)
    else:
        mirror = tf.less(1.0, 0.5)

    # crop
    crop_index = tf.random_uniform((), maxval=multi_scale_crop_sizes.get_shape()[0].value, dtype=tf.int32)
    crop_height = tf.gather(tf.gather(multi_scale_crop_sizes, crop_index), 0)
    crop_width = tf.gather(tf.gather(multi_scale_crop_sizes, crop_index), 1)
    if config['crop'] == 0:
        ho = None
        wo = None
    elif config['crop'] == 1:
        ho = tf.random_uniform((), maxval=config['height']-crop_height+1, dtype=tf.int32)
        wo = tf.random_uniform((), maxval=config['width']-crop_width+1, dtype=tf.int32)
    elif config['crop'] == 2:
        fix_offsets = get_fix_offset(int(config['width']), int(config['height']), crop_height, crop_width)
        offset_index = tf.random_uniform((), maxval=fix_offsets.get_shape()[0].value, dtype=tf.int32)
        ho = tf.gather(tf.gather(fix_offsets, offset_index), 0)
        wo = tf.gather(tf.gather(fix_offsets, offset_index), 1)
    else:
        raise NotImplementedError('Crop mode %d is not supported.'%config['crop'])

    images = []
    images2 = []
    for i in xrange(config['n_steps']):
        image = one_image(config['modality'], tf.add(config['data_path1'], video_name),
                          offset_str, offset+tf.to_int32(tf.floor(i*read_stride)),
                          config['height'], config['width'], config['crop'],
                          ho, wo, config['crop_size'], crop_height, crop_width,
                          config['preprocessing_fn_1'], False, config['length1'])
        image = tf.cond(mirror, lambda:tf.image.flip_left_right(image), lambda:image)
        images.append(image)
        if config['modality2'] is not None:
            image = one_image(config['modality2'], tf.add(config['data_path2'], video_name),
                            offset_str, offset+tf.to_int32(tf.floor(i*read_stride)),
                            config['height'], config['width'], config['crop'],
                            ho, wo, config['crop_size'], crop_height, crop_width,
                            config['preprocessing_fn_2'], False, config['length2'])
            image = tf.cond(mirror, lambda:tf.image.flip_left_right(image), lambda:image)
            images2.append(image)

    images = tf.stack(images)
    if config['modality2'] is not None:
        images2 = tf.stack(images2)
    return images, images2, labels


def video_inputs(groundtruth_path, data_path1, scale_size, crop_size,
            batch_size, n_steps, modality, read_stride,
            preprocessing_fn_1, preprocessing_fn_2=None,
            data_path2="", modality2=None,
            length1=1, length2=1,
            shuffle=False, label_from_one=False, crop=0,
            max_distort=1, scale_ratios=[1,.875,.75,.66],
            merge_label=False):
    data_path1 = data_path1 + '/'
    data_path2 = data_path2 + '/'
    config = {'width':scale_size, 'height':scale_size, 'crop_size':crop_size,
              'n_steps':n_steps, 'modality':modality, 'crop':crop,
              "data_path1":data_path1, "data_path2":data_path2,
              'length1':length1, 'length2':length2, 'mirror':shuffle,
              'preprocessing_fn_1':preprocessing_fn_1, 'preprocessing_fn_2':preprocessing_fn_2,
              'modality2':modality2, 'merge_label':merge_label}

    gt_lines = open(groundtruth_path).readlines()
    gt_pairs = [line.split() for line in gt_lines]
    #  paths = [os.path.join(data_path, p[0]) for p in gt_pairs]
    paths = [p[0] for p in gt_pairs]
    if len(gt_pairs[0]) == 2:
        labels = np.array([int(p[1]) for p in gt_pairs])
        if label_from_one:
            labels -= 1
    else:
        raise NotImplementedError('Ground truth file should contain one label.')
    print('%d samples in list.'%len(labels))

    nums = [len(os.listdir(data_path1+p))/3-max(length1, length2)+1 for p in paths]
    remove_list = [i for i in xrange(len(nums)) if nums[i] <= 0]
    if len(remove_list) > 0:
        for i in xrange(len(remove_list)):
            print("Removing %s"%(paths[remove_list[i]]))
        paths = [p for i,p in enumerate(paths) if i not in remove_list]
        labels = [l for i,l in enumerate(labels) if i not in remove_list]
        nums = [int(n) for i,n in enumerate(nums) if i not in remove_list]
    read_strides = [min(read_stride,float(n)/n_steps) for n in nums]
    # trancate sample lenth
    nums = [int(n-read_strides[i]*n_steps+read_strides[i]) for i,n in enumerate(nums)]

    dataset_size = len(labels)

    offset_str = ["%04d"%i for i in xrange(1,9999)]

    multi_scale_crop_sizes = get_multi_scale_crop_size(scale_size, scale_size, crop_size, scale_ratios, max_distort)

    paths = tf.convert_to_tensor(paths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    nums = tf.convert_to_tensor(nums, dtype=tf.int32)
    read_strides = tf.convert_to_tensor(read_strides, dtype=tf.float32)
    offset_str = tf.convert_to_tensor(offset_str, dtype=tf.string)
    multi_scale_crop_sizes = tf.convert_to_tensor(multi_scale_crop_sizes, dtype=tf.int32)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.slice_input_producer([paths, labels, nums, read_strides],
                                                    shuffle=shuffle)

    # Read examples from files in the filename queue.
    image, image2, label = read_video(filename_queue, offset_str, multi_scale_crop_sizes, config)

    # Ensure that the random shuffling has good mixing properties.
    min_queue_examples = 64

    # Generate a batch of images and labels by building up a queue of examples.
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        if modality2 is None:
            images, label_batch, name = tf.train.shuffle_batch(
                [image, label, filename_queue[0]],
                batch_size=int(batch_size/n_steps),
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples,
                min_after_dequeue=int(min_queue_examples/2))
        else:
            images, images2, label_batch, name = tf.train.shuffle_batch(
                [image, image2, label, filename_queue[0]],
                batch_size=int(batch_size/n_steps),
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples,
                min_after_dequeue=int(min_queue_examples/2))
    else:
        if modality2 is None:
            images, label_batch, name = tf.train.batch(
                [image, label, filename_queue[0]],
                batch_size=int(batch_size/n_steps),
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples)
        else:
            images, images2, label_batch, name = tf.train.batch(
                [image, image2, label, filename_queue[0]],
                batch_size=int(batch_size/n_steps),
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples)

    imgs_shape = images.get_shape()
    if len(imgs_shape) == 5:
        images = tf.transpose(images, [1,0,2,3,4])
        images = tf.reshape(images, [batch_size, imgs_shape[2].value, imgs_shape[3].value, imgs_shape[4].value])
        if not merge_label:
            label_batch = tf.transpose(label_batch, [1,0])
    else:
        raise NotImplementedError("Images shape length is %d"%len(imgs_shape))
    if merge_label:
        video_num = int(batch_size/n_steps)
        label_batch = tf.reshape(label_batch, [video_num])
    else:
        label_batch = tf.reshape(label_batch, [batch_size])

    if imgs_shape[4].value <= 4:
        tf.summary.image("Modality1_batch", images, max_outputs=batch_size)
        tf.summary.scalar("Modality1_batch_max_value", tf.reduce_max(images))
        tf.summary.scalar("Modality1_batch_min_value", tf.reduce_min(images))
        tf.summary.scalar("Modality1_batch_mean_value", tf.reduce_mean(images))
    if modality2 is None:
        return dataset_size, images, label_batch, name
    else:
        imgs_shape = images2.get_shape()
        if len(imgs_shape) == 5:
            images2 = tf.transpose(images2, [1,0,2,3,4])
            images2 = tf.reshape(images2, [batch_size, imgs_shape[2].value, imgs_shape[3].value, imgs_shape[4].value])
        else:
            raise NotImplementedError("Images shape length is %d"%len(imgs_shape))
        if imgs_shape[4].value <= 4:
            tf.summary.image("Modality2_batch", images2, max_outputs=batch_size)

        return dataset_size, images, images2, label_batch, name


