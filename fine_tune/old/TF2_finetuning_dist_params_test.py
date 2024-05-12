#!/usr/bin/env python
# coding: utf-8

# ## SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
# 
# This colab demonstrates how to load pretrained/finetuned SimCLR models from hub modules for fine-tuning
# 
# The checkpoints are accessible in the following Google Cloud Storage folders.
# 
# * Pretrained SimCLRv2 models with a linear classifier: [gs://simclr-checkpoints-tf2/simclrv2/pretrained](https://console.cloud.google.com/storage/browser/simclr-checkpoints-tf2/simclrv2/pretrained)
# * Fine-tuned SimCLRv2 models on 1% of labels: [gs://simclr-checkpoints-tf2/simclrv2/finetuned_1pct](https://console.cloud.google.com/storage/browser/simclr-checkpoints-tf2/simclrv2/finetuned_1pct)
# * Fine-tuned SimCLRv2 models on 10% of labels: [gs://simclr-checkpoints-tf2/simclrv2/finetuned_10pct](https://console.cloud.google.com/storage/browser/simclr-checkpoints-tf2/simclrv2/finetuned_10pct)
# * Fine-tuned SimCLRv2 models on 100% of labels: [gs://simclr-checkpoints-tf2/simclrv2/finetuned_100pct](https://console.cloud.google.com/storage/browser/simclr-checkpoints-tf2/simclrv2/finetuned_100pct)
# * Supervised models with the same architectures: [gs://simclr-checkpoints-tf2/simclrv2/pretrained](https://console.cloud.google.com/storage/browser/simclr-checkpoints-tf2/simclrv2/pretrained)
# 


import re
import numpy as np
import tensorflow.compat.v2 as tf
tf.compat.v1.enable_v2_behavior()
#import matplotlib.pyplot as plt
#import tensorflow_addons as tfa
from sklearn.metrics import f1_score
import pandas as pd
import pathlib, random, os, sys

AUTOTUNE = tf.data.experimental.AUTOTUNE



#inputs
#num_classes = 10
in_file = sys.argv[1]
in_data = sys.argv[2]
in_path = sys.argv[3]
num_classes = int(sys.argv[4])
file_path = "/rds/general/user/hm808/home/repos/" + in_file + "/saved_model/" + in_path +"/"
data_root_train = pathlib.Path(r"/rds/general/user/hm808/home/images/"+in_data+"/training")
data_root_val = pathlib.Path(r"/rds/general/user/hm808/home/images/"+in_data+"/validation")
data_root_test = pathlib.Path(r"/rds/general/user/hm808/home/images/"+in_data+"/testing")


#pre-processing of the images and preparation of dataset


def prepare_dataset(data_root):
  all_image_paths = list(data_root.glob('*/*'))
  all_image_paths = [str(path) for path in all_image_paths]
  image_count = len(all_image_paths)
  random.shuffle(all_image_paths)

  label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
  label_to_index = dict((name, index) for index, name in enumerate(label_names))
  all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                      for path in all_image_paths]

  path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
  image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
  label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
  image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

  ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
  # The tuples are unpacked into the positional arguments of the mapped function
  image_label_ds = ds.map(load_and_preprocess_from_path_label)
  return image_label_ds, image_count

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image_custom(image)

def preprocess_image_custom(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [224, 224])
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_from_path_label(path, label):
  #return load_and_preprocess_image(path), label
  return {'image':load_and_preprocess_image(path), 'label':label}

def lab_names(data_root):
  label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
  label_to_index = dict((name, index) for index, name in enumerate(label_names))
  return label_to_index



#preprocessing functions from data_util.py in SimCLR repository.

FLAGS_color_jitter_strength = 0.3
CROP_PROPORTION = 0.875  # Standard for ImageNet.


def random_apply(func, p, x):
  """Randomly apply function func to x with probability p."""
  return tf.cond(
      tf.less(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32),
              tf.cast(p, tf.float32)),
      lambda: func(x),
      lambda: x)


def random_brightness(image, max_delta, impl='simclrv2'):
  """A multiplicative vs additive change of brightness."""
  if impl == 'simclrv2':
    factor = tf.random_uniform(
        [], tf.maximum(1.0 - max_delta, 0), 1.0 + max_delta)
    image = image * factor
  elif impl == 'simclrv1':
    image = random_brightness(image, max_delta=max_delta)
  else:
    raise ValueError('Unknown impl {} for random brightness.'.format(impl))
  return image


def to_grayscale(image, keep_channels=True):
  image = tf.image.rgb_to_grayscale(image)
  if keep_channels:
    image = tf.tile(image, [1, 1, 3])
  return image


def color_jitter(image,
                 strength,
                 random_order=True):
  """Distorts the color of the image.
  Args:
    image: The input image tensor.
    strength: the floating number for the strength of the color augmentation.
    random_order: A bool, specifying whether to randomize the jittering order.
  Returns:
    The distorted image tensor.
  """
  brightness = 0.8 * strength
  contrast = 0.8 * strength
  saturation = 0.8 * strength
  hue = 0.2 * strength
  if random_order:
    return color_jitter_rand(image, brightness, contrast, saturation, hue)
  else:
    return color_jitter_nonrand(image, brightness, contrast, saturation, hue)


def color_jitter_nonrand(image, brightness=0, contrast=0, saturation=0, hue=0):
  """Distorts the color of the image (jittering order is fixed).
  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.
  Returns:
    The distorted image tensor.
  """
  with tf.name_scope('distort_color'):
    def apply_transform(i, x, brightness, contrast, saturation, hue):
      """Apply the i-th transformation."""
      if brightness != 0 and i == 0:
        x = random_brightness(x, max_delta=brightness)
      elif contrast != 0 and i == 1:
        x = tf.image.random_contrast(
            x, lower=1-contrast, upper=1+contrast)
      elif saturation != 0 and i == 2:
        x = tf.image.random_saturation(
            x, lower=1-saturation, upper=1+saturation)
      elif hue != 0:
        x = tf.image.random_hue(x, max_delta=hue)
      return x

    for i in range(4):
      image = apply_transform(i, image, brightness, contrast, saturation, hue)
      image = tf.clip_by_value(image, 0., 1.)
    return image


def color_jitter_rand(image, brightness=0, contrast=0, saturation=0, hue=0):
  """Distorts the color of the image (jittering order is random).
  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.
  Returns:
    The distorted image tensor.
  """
  with tf.name_scope('distort_color'):
    def apply_transform(i, x):
      """Apply the i-th transformation."""
      def brightness_foo():
        if brightness == 0:
          return x
        else:
          return random_brightness(x, max_delta=brightness)
      def contrast_foo():
        if contrast == 0:
          return x
        else:
          return tf.image.random_contrast(x, lower=1-contrast, upper=1+contrast)
      def saturation_foo():
        if saturation == 0:
          return x
        else:
          return tf.image.random_saturation(
              x, lower=1-saturation, upper=1+saturation)
      def hue_foo():
        if hue == 0:
          return x
        else:
          return tf.image.random_hue(x, max_delta=hue)
      x = tf.cond(tf.less(i, 2),
                  lambda: tf.cond(tf.less(i, 1), brightness_foo, contrast_foo),
                  lambda: tf.cond(tf.less(i, 3), saturation_foo, hue_foo))
      return x

    perm = tf.random_shuffle(tf.range(4))
    for i in range(4):
      image = apply_transform(perm[i], image)
      image = tf.clip_by_value(image, 0., 1.)
    return image


def _compute_crop_shape(
    image_height, image_width, aspect_ratio, crop_proportion):
  """Compute aspect ratio-preserving shape for central crop.
  The resulting shape retains `crop_proportion` along one side and a proportion
  less than or equal to `crop_proportion` along the other side.
  Args:
    image_height: Height of image to be cropped.
    image_width: Width of image to be cropped.
    aspect_ratio: Desired aspect ratio (width / height) of output.
    crop_proportion: Proportion of image to retain along the less-cropped side.
  Returns:
    crop_height: Height of image after cropping.
    crop_width: Width of image after cropping.
  """
  image_width_float = tf.cast(image_width, tf.float32)
  image_height_float = tf.cast(image_height, tf.float32)

  def _requested_aspect_ratio_wider_than_image():
    crop_height = tf.cast(tf.math.rint(
        crop_proportion / aspect_ratio * image_width_float), tf.int32)
    crop_width = tf.cast(tf.math.rint(
        crop_proportion * image_width_float), tf.int32)
    return crop_height, crop_width

  def _image_wider_than_requested_aspect_ratio():
    crop_height = tf.cast(
        tf.math.rint(crop_proportion * image_height_float), tf.int32)
    crop_width = tf.cast(tf.math.rint(
        crop_proportion * aspect_ratio *
        image_height_float), tf.int32)
    return crop_height, crop_width

  return tf.cond(
      aspect_ratio > image_width_float / image_height_float,
      _requested_aspect_ratio_wider_than_image,
      _image_wider_than_requested_aspect_ratio)


def center_crop(image, height, width, crop_proportion):
  """Crops to center of image and rescales to desired size.
  Args:
    image: Image Tensor to crop.
    height: Height of image to be cropped.
    width: Width of image to be cropped.
    crop_proportion: Proportion of image to retain along the less-cropped side.
  Returns:
    A `height` x `width` x channels Tensor holding a central crop of `image`.
  """
  shape = tf.shape(image)
  image_height = shape[0]
  image_width = shape[1]
  crop_height, crop_width = _compute_crop_shape(
      image_height, image_width, height / width, crop_proportion)
  offset_height = ((image_height - crop_height) + 1) // 2
  offset_width = ((image_width - crop_width) + 1) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_height, offset_width, crop_height, crop_width)

  image = tf.compat.v1.image.resize_bicubic([image], [height, width])[0]

  return image


def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using one of the bboxes randomly distorted.
  See `tf.image.sample_distorted_bounding_box` for more documentation.
  Args:
    image: `Tensor` of image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
        where each coordinate is [0, 1) and the coordinates are arranged
        as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding
        box supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
    scope: Optional `str` for name scope.
  Returns:
    (cropped image `Tensor`, distorted bbox `Tensor`).
  """
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
    shape = tf.shape(image)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, target_height, target_width)

    return image


def crop_and_resize(image, height, width):
  """Make a random crop and resize it to height `height` and width `width`.
  Args:
    image: Tensor representing the image.
    height: Desired image height.
    width: Desired image width.
  Returns:
    A `height` x `width` x channels Tensor holding a random crop of `image`.
  """
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  aspect_ratio = width / height
  image = distorted_bounding_box_crop(
      image,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),
      area_range=(0.08, 1.0),
      max_attempts=100,
      scope=None)
  return tf.compat.v1.image.resize_bicubic([image], [height, width])[0]


def gaussian_blur(image, kernel_size, sigma, padding='SAME'):
  """Blurs the given image with separable convolution.
  Args:
    image: Tensor of shape [height, width, channels] and dtype float to blur.
    kernel_size: Integer Tensor for the size of the blur kernel. This is should
      be an odd number. If it is an even number, the actual kernel size will be
      size + 1.
    sigma: Sigma value for gaussian operator.
    padding: Padding to use for the convolution. Typically 'SAME' or 'VALID'.
  Returns:
    A Tensor representing the blurred image.
  """
  radius = tf.to_int32(kernel_size / 2)
  kernel_size = radius * 2 + 1
  x = tf.to_float(tf.range(-radius, radius + 1))
  blur_filter = tf.exp(
      -tf.pow(x, 2.0) / (2.0 * tf.pow(tf.to_float(sigma), 2.0)))
  blur_filter /= tf.reduce_sum(blur_filter)
  # One vertical and one horizontal filter.
  blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
  blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
  num_channels = tf.shape(image)[-1]
  blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
  blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
  expand_batch_dim = image.shape.ndims == 3
  if expand_batch_dim:
    # Tensorflow requires batched input to convolutions, which we can fake with
    # an extra dimension.
    image = tf.expand_dims(image, axis=0)
  blurred = tf.nn.depthwise_conv2d(
      image, blur_h, strides=[1, 1, 1, 1], padding=padding)
  blurred = tf.nn.depthwise_conv2d(
      blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
  if expand_batch_dim:
    blurred = tf.squeeze(blurred, axis=0)
  return blurred


def random_crop_with_resize(image, height, width, p=1.0):
  """Randomly crop and resize an image.
  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    p: Probability of applying this transformation.
  Returns:
    A preprocessed image `Tensor`.
  """
  def _transform(image):  # pylint: disable=missing-docstring
    image = crop_and_resize(image, height, width)
    return image
  return random_apply(_transform, p=p, x=image)


def random_color_jitter(image, p=1.0):
  def _transform(image):
    color_jitter_t = functools.partial(
        color_jitter, strength=FLAGS_color_jitter_strength)
    image = random_apply(color_jitter_t, p=0.8, x=image)
    return random_apply(to_grayscale, p=0.2, x=image)
  return random_apply(_transform, p=p, x=image)


def random_blur(image, height, width, p=1.0):
  """Randomly blur an image.
  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    p: probability of applying this transformation.
  Returns:
    A preprocessed image `Tensor`.
  """
  del width
  def _transform(image):
    sigma = tf.random.uniform([], 0.1, 2.0, dtype=tf.float32)
    return gaussian_blur(
        image, kernel_size=height//10, sigma=sigma, padding='SAME')
  return random_apply(_transform, p=p, x=image)


def batch_random_blur(images_list, height, width, blur_probability=0.5):
  """Apply efficient batch data transformations.
  Args:
    images_list: a list of image tensors.
    height: the height of image.
    width: the width of image.
    blur_probability: the probaility to apply the blur operator.
  Returns:
    Preprocessed feature list.
  """
  def generate_selector(p, bsz):
    shape = [bsz, 1, 1, 1]
    selector = tf.cast(
        tf.less(tf.random_uniform(shape, 0, 1, dtype=tf.float32), p),
        tf.float32)
    return selector

  new_images_list = []
  for images in images_list:
    images_new = random_blur(images, height, width, p=1.)
    selector = generate_selector(blur_probability, tf.shape(images)[0])
    images = images_new * selector + images * (1 - selector)
    images = tf.clip_by_value(images, 0., 1.)
    new_images_list.append(images)

  return new_images_list


def preprocess_for_train(image, height, width,
                         color_distort=True, crop=True, flip=True):
  """Preprocesses the given image for training.
  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    color_distort: Whether to apply the color distortion.
    crop: Whether to crop the image.
    flip: Whether or not to flip left and right of an image.
  Returns:
    A preprocessed image `Tensor`.
  """
  if crop:
    image = random_crop_with_resize(image, height, width)
  if flip:
    image = tf.image.random_flip_left_right(image)
  if color_distort:
    image = random_color_jitter(image)
  image = tf.reshape(image, [height, width, 3])
  image = tf.clip_by_value(image, 0., 1.)
  return image


def preprocess_for_eval(image, height, width, crop=True):
  """Preprocesses the given image for evaluation.
  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    crop: Whether or not to (center) crop the test images.
  Returns:
    A preprocessed image `Tensor`.
  """
  if crop:
    image = center_crop(image, height, width, crop_proportion=CROP_PROPORTION)
  image = tf.reshape(image, [height, width, 3])
  image = tf.clip_by_value(image, 0., 1.)
  return image


def preprocess_image(image, height, width, is_training=False,
                     color_distort=True, test_crop=True):
  """Preprocesses the given image.
  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    is_training: `bool` for whether the preprocessing is for training.
    color_distort: whether to apply the color distortion.
    test_crop: whether or not to extract a central crop of the images
        (as for standard ImageNet evaluation) during the evaluation.
  Returns:
    A preprocessed image `Tensor` of range [0, 1].
  """
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  if is_training:
    return preprocess_for_train(image, height, width, color_distort)
  else:
    return preprocess_for_eval(image, height, width, test_crop)


#optimization set-up


EETA_DEFAULT = 0.001


class LARSOptimizer(tf.keras.optimizers.Optimizer):
  """Layer-wise Adaptive Rate Scaling for large batch training.

  Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
  I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
  """

  def __init__(self,
               learning_rate,
               momentum=0.9,
               use_nesterov=False,
               weight_decay=0.0,
               exclude_from_weight_decay=None,
               exclude_from_layer_adaptation=None,
               classic_momentum=True,
               eeta=EETA_DEFAULT,
               name="LARSOptimizer"):
    """Constructs a LARSOptimizer.

    Args:
      learning_rate: A `float` for learning rate.
      momentum: A `float` for momentum.
      use_nesterov: A 'Boolean' for whether to use nesterov momentum.
      weight_decay: A `float` for weight decay.
      exclude_from_weight_decay: A list of `string` for variable screening, if
          any of the string appears in a variable's name, the variable will be
          excluded for computing weight decay. For example, one could specify
          the list like ['batch_normalization', 'bias'] to exclude BN and bias
          from weight decay.
      exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
          for layer adaptation. If it is None, it will be defaulted the same as
          exclude_from_weight_decay.
      classic_momentum: A `boolean` for whether to use classic (or popular)
          momentum. The learning rate is applied during momeuntum update in
          classic momentum, but after momentum for popular momentum.
      eeta: A `float` for scaling of learning rate when computing trust ratio.
      name: The name for the scope.
    """
    super(LARSOptimizer, self).__init__(name)

    self._set_hyper("learning_rate", learning_rate)
    self.momentum = momentum
    self.weight_decay = weight_decay
    self.use_nesterov = use_nesterov
    self.classic_momentum = classic_momentum
    self.eeta = eeta
    self.exclude_from_weight_decay = exclude_from_weight_decay
    # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
    # arg is None.
    if exclude_from_layer_adaptation:
      self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
    else:
      self.exclude_from_layer_adaptation = exclude_from_weight_decay

  def _create_slots(self, var_list):
    for v in var_list:
      self.add_slot(v, "Momentum")

  def _resource_apply_dense(self, grad, param, apply_state=None):
    if grad is None or param is None:
      return tf.no_op()

    var_device, var_dtype = param.device, param.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))
    learning_rate = coefficients["lr_t"]

    param_name = param.name

    v = self.get_slot(param, "Momentum")

    if self._use_weight_decay(param_name):
      grad += self.weight_decay * param

    if self.classic_momentum:
      trust_ratio = 1.0
      if self._do_layer_adaptation(param_name):
        w_norm = tf.norm(param, ord=2)
        g_norm = tf.norm(grad, ord=2)
        trust_ratio = tf.where(
            tf.greater(w_norm, 0),
            tf.where(tf.greater(g_norm, 0), (self.eeta * w_norm / g_norm), 1.0),
            1.0)
      scaled_lr = learning_rate * trust_ratio

      next_v = tf.multiply(self.momentum, v) + scaled_lr * grad
      if self.use_nesterov:
        update = tf.multiply(self.momentum, next_v) + scaled_lr * grad
      else:
        update = next_v
      next_param = param - update
    else:
      next_v = tf.multiply(self.momentum, v) + grad
      if self.use_nesterov:
        update = tf.multiply(self.momentum, next_v) + grad
      else:
        update = next_v

      trust_ratio = 1.0
      if self._do_layer_adaptation(param_name):
        w_norm = tf.norm(param, ord=2)
        v_norm = tf.norm(update, ord=2)
        trust_ratio = tf.where(
            tf.greater(w_norm, 0),
            tf.where(tf.greater(v_norm, 0), (self.eeta * w_norm / v_norm), 1.0),
            1.0)
      scaled_lr = trust_ratio * learning_rate
      next_param = param - scaled_lr * update

    return tf.group(*[
        param.assign(next_param, use_locking=False),
        v.assign(next_v, use_locking=False)
    ])

  def _use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _do_layer_adaptation(self, param_name):
    """Whether to do layer-wise learning rate adaptation for `param_name`."""
    if self.exclude_from_layer_adaptation:
      for r in self.exclude_from_layer_adaptation:
        if re.search(r, param_name) is not None:
          return False
    return True

  def get_config(self):
    config = super(LARSOptimizer, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "momentum": self.momentum,
        "classic_momentum": self.classic_momentum,
        "weight_decay": self.weight_decay,
        "eeta": self.eeta,
        "use_nesterov": self.use_nesterov,
    })
    return config



#Load module and construct the computation graph

learning_rate = 0.1
momentum = 0.9
weight_decay = 0.


class Model(tf.keras.Model):
  def __init__(self, path):
    super(Model, self).__init__()
    self.saved_model = tf.saved_model.load(path)
    self.dense_layer = tf.keras.layers.Dense(units=num_classes, name="head_supervised_new")
    self.optimizer = LARSOptimizer(
      learning_rate,
      momentum=momentum,
      weight_decay=weight_decay,
      exclude_from_weight_decay=['batch_normalization', 'bias', 'head_supervised'])

  def call(self, x):
    with tf.GradientTape() as tape:
      outputs = self.saved_model(x['image'], trainable=False)
      print(outputs)
      logits_t = self.dense_layer(outputs['final_avg_pool'])
      loss_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels = tf.one_hot(x['label'], num_classes), logits=logits_t))
      dense_layer_weights = self.dense_layer.trainable_weights
      print('Variables to train:', dense_layer_weights)
      grads = tape.gradient(loss_t, dense_layer_weights)
      self.optimizer.apply_gradients(zip(grads, dense_layer_weights))
    return loss_t, x["image"], logits_t, x["label"]


# run the models

model = Model(file_path)

# Remove this for debugging.  
@tf.function
def train_step(x):
  return model(x)

#We fine-tune the new *linear layer* for just a few iterations.
def _preprocess(x):
  x['image'] = preprocess_image(
      x['image'], 224, 224, is_training=False, color_distort=False)
  return x

image_label_ds_val, image_count_val = prepare_dataset(data_root_val)
image_label_ds_test, image_count_test = prepare_dataset(data_root_test)

batch_size = 64
ds_val = image_label_ds_val.map(_preprocess).batch(batch_size)
ds_test = image_label_ds_test.map(_preprocess).batch(batch_size)
iter_list, loss_train_list, acc_train_list, loss_val_list, acc_val_list, F1_weight_val_list, loss_test_list, acc_test_list, F1_weight_test_list = [], [], [], [], [], [], [], [], []

for epoch in range(2):
    print("epoch: {}".format(epoch))
    image_label_ds_train, image_count_train = prepare_dataset(data_root_train)
    ds = image_label_ds_train.map(_preprocess).batch(batch_size)
    total_iterations = int(image_count_train / batch_size)
    loss_train_list_in, acc_train_list_in = [], []
    iterator = iter(ds)
    for it in range(total_iterations):
      x = next(iterator)
      loss, image, logits, labels = train_step(x)
      logits = logits.numpy()
      labels = labels.numpy()
      pred = logits.argmax(-1)
      loss = loss.numpy()
      acc = np.sum(pred == labels) / len(labels)
        
      loss_train_list_in.append(loss)
      acc_train_list_in.append(acc)
        
    top_1_accuracy = tf.keras.metrics.Accuracy('top_1_accuracy')
    loss_val_list_in, acc_val_list_in, F1_weight_val_list_in = [], [], []
        
    for i, features in enumerate(ds_val):
        outputs = model.saved_model(features['image'], trainable=False)
        logits_t = model.dense_layer(outputs['final_avg_pool'])
        loss_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
              labels = tf.one_hot(features['label'], num_classes), logits=logits_t))
        top_1_accuracy.update_state(features["label"], tf.argmax(logits_t, axis=-1))
        y_true = tf.cast(features["label"], tf.int32).numpy()
        y_pred = tf.cast(tf.argmax(logits_t, axis=-1), tf.int32).numpy()
        loss_val_list_in.append(loss_t)
        acc_val = np.sum(y_true == y_pred) / len(y_true)
        acc_val_list_in.append(acc_val)
        F1_weight_val_list_in.append(f1_score(y_true, y_pred, average='weighted'))

        
    loss_test_list_in, acc_test_list_in, F1_weight_test_list_in = [], [], []
        
    for i, features in enumerate(ds_test):
        outputs = model.saved_model(features['image'], trainable=False)
        logits_t = model.dense_layer(outputs['final_avg_pool'])
        loss_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
              labels = tf.one_hot(features['label'], num_classes), logits=logits_t))
        y_true = tf.cast(features["label"], tf.int32).numpy()
        y_pred = tf.cast(tf.argmax(logits_t, axis=-1), tf.int32).numpy()
        print(y_true)
        print(y_pred)
        print(logits_t)
        
        prediction = tf.nn.sigmoid(logits_t)
        print(prediction)
        y_pred2 = tf.cast(tf.argmax(prediction, axis=-1), tf.int32).numpy()
        print(y_pred2)

        #loss_test_list_in.append(loss_t)
        acc_test = np.sum(y_true == y_pred) / len(y_true)
        acc_test_list_in.append(acc_test)
        F1_weight_test_list_in.append(f1_score(y_true, y_pred, average='weighted'))

    print("[Epoch {}] Train Loss: {:.3f}, Train Acc: {:.3f}, Val Loss: {:.3f}, Val Acc: {:.3f}, Val F1 Weighted: {:.3f}, Test Acc: {:.3f}, Test F1 Weighted: {:.3f}".format(epoch+1, np.array(loss_train_list_in).mean(), np.array(acc_train_list_in).mean(), np.array(loss_val_list_in).mean(),np.array(acc_val_list_in).mean(), np.array(F1_weight_val_list_in).mean(),np.array(acc_test_list_in).mean(), np.array(F1_weight_test_list_in).mean()))      
    iter_list.append(epoch+1)
    loss_train_list.append(np.array(loss_train_list_in).mean())
    loss_val_list.append(np.array(loss_val_list_in).mean())
    acc_train_list.append(np.array(acc_train_list_in).mean())
    acc_val_list.append(np.array(acc_val_list_in).mean())
    acc_test_list.append(np.array(acc_test_list_in).mean())
    F1_weight_val_list.append(np.array(F1_weight_val_list_in).mean())
    F1_weight_test_list.append(np.array(F1_weight_test_list_in).mean())
    

dict_ = {'iter': iter_list, 'Train Loss': loss_train_list,'Val Loss': loss_val_list, 'Train Acc': acc_train_list, 'Val Acc': acc_val_list, 'Val F1_weighted': F1_weight_val_list, 'Test Acc': acc_test_list, 'Test F1_weighted': F1_weight_test_list} 
df = pd.DataFrame(dict_)
df_name = in_file + "_" + in_data +".csv"
df.to_csv(df_name)
