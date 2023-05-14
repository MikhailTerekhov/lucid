# Copyright 2018 The Lucid Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Helpers for getting responses from models over large collections."""


import itertools
from collections import defaultdict

import numpy as np
import tensorflow as tf
import tqdm
import timeit

from lucid.misc.iter_nd_utils import recursive_enumerate_nd, dict_to_ndarray, batch_iter
from lucid.misc.ndimage_utils import resize, quick_resize


def get_activations_knockout(model, images, knockout_layer, knockout_feature_ind, knockout_value,
                             output_layer, batch_size=256, outer_batch_size=8192):
    """
    Evaluate activations on input images, but replace the value of the specified feature in the layer with a constant value.
    This is called "knockout" in the literature.

    knockout_value can either be a scalar or a numpy array of the same shape as the corrsponding slice of the activations.
    In the latter case, the shape should be (images.shape[0], *activations.shape[1:-1]).
    """

    def is_float(x):
      return isinstance(x, float) or isinstance(x, np.float32) or isinstance(x,
                                                                             np.float64)

    out_vals = []
    with tqdm.tqdm(total=images.shape[0], desc='Evaluating knockout activations') as pbar:
        for bi in range(0, images.shape[0], outer_batch_size):
            o_batch = images[bi:bi + outer_batch_size, ...]
            vals = []
            with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session() as sess:
                t_img = tf.compat.v1.placeholder("float32", [None, None, None, 3])
                T = model.import_graph(t_img)
                t_knockout = T(knockout_layer)
                t_output = T(output_layer)

                for batch_i in range(0, o_batch.shape[0], batch_size):
                    batch = quick_resize(o_batch[batch_i:batch_i + batch_size, ...], (224, 224))

                    handle_first = sess.partial_run_setup([t_knockout], [t_img])
                    acts = sess.partial_run(handle_first, t_knockout, feed_dict={t_img: batch})
                    val = np.full(acts[..., knockout_feature_ind].shape, knockout_value) \
                        if is_float(knockout_value) \
                        else knockout_value[bi + batch_i:bi + batch_i + batch_size, ...]
                    acts[..., knockout_feature_ind] = val
                    handle_second = sess.partial_run_setup([t_output], [t_knockout])
                    outputs = sess.partial_run(handle_second, t_output, feed_dict={t_knockout: acts})
                    out_vals.append(outputs)
                    pbar.update(batch_size)
    return np.concatenate(out_vals, axis=0)


def get_activations_new(model, images, layer, feature_ind=None, mean_axes=None, batch_size=256, outer_batch_size=8192,
                        dtype=None):
    """Mikhail's version of get_activations that works with TF2 in compatibility mode.

    If feature_ind is not None, returns only the specified feature (activations[..., feature_ind]).
    If mean_axes is not None, returns the mean over the specified axes (np.mean(activations, axis=mean_axes)).
    For intermediate levels of the network one of them might need to be specified to avoid going OOM with the result.
    """
    out_vals = []
    with tqdm.tqdm(total=images.shape[0], desc='Evaluating activations') as pbar:
        for bi in range(0, images.shape[0], outer_batch_size):
            o_batch = images[bi:bi + outer_batch_size, ...]
            with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session() as sess:
                t_img = tf.compat.v1.placeholder("float32", [None, None, None, 3])
                T = model.import_graph(t_img)
                t_layer = T(layer)

                vals = []
                for batch_i in range(0, o_batch.shape[0], batch_size):
                    batch = quick_resize(o_batch[batch_i:batch_i + batch_size, ...], (224, 224))
                    acts = t_layer.eval({t_img: batch})
                    if dtype is not None:
                        acts = acts.astype(dtype)
                    if feature_ind is not None:
                        acts = acts[..., feature_ind]
                    elif mean_axes is not None:
                        acts = np.mean(acts, axis=mean_axes)
                    vals.append(acts)
                    pbar.update(batch_size)
                out_vals.append(tf.concat(vals, axis=0).eval())
    return np.concatenate(out_vals, axis=0)


def get_activations_iter(model, layer, generator, reducer="mean", batch_size=64,
                         dtype=None, ind_shape=None, center_only=False):
  """Collect center activtions of a layer over many images from an iterable obj.

  Note: this is mostly intended for large synthetic families of images, where
    you can cheaply generate them in Python. For collecting activations over,
    say, ImageNet, there will be better workflows based on various dataset APIs
    in TensorFlow.

  Args:
    model: model for activations to be collected from.
    layer: layer (in model) for activtions to be collected from.
    generator: An iterable object (intended to be a generator) which produces
      tuples of the form (index, image). See details below.
    reducer: How to combine activations if multiple images map to the same index.
      Supports "mean", "rms", and "max".
    batch_size: How many images from the generator should be processes at once?
    dtype: determines dtype of returned data (defaults to model activation
      dtype). Can be used to make function memory efficient.
    ind_shape: Shape that indices can span. Optional, but makes function orders
      of magnitiude more memory efficient.

  Memory efficeincy:
    Using ind_shape is the main tool for make this function memory efficient.
    dtype="float16" can further help.

  Returns:
    A numpy array of shape [ind1, ind2, ..., layer_channels]
  """


  assert reducer in ["mean", "max", "rms"]
  combiner, normalizer = {
      "mean" : (lambda a,b: a+b,             lambda a,n: a/n         ),
      "rms"  : (lambda a,b: a+b**2,          lambda a,n: np.sqrt(a/n)),
      "max"  : (lambda a,b: np.maximum(a,b), lambda a,n: a           ),
  }[reducer]

  with tf.Graph().as_default(), tf.Session() as sess:
    t_img = tf.placeholder("float32", [None, None, None, 3])
    T = model.import_graph(t_img)
    t_layer = T(layer)

    responses = None
    count = None

    # # If we know the total length, let's give a progress bar
    # if ind_shape is not None:
    #   total = int(np.prod(ind_shape))
    #   generator = tqdm(generator, total=total)

    for batch in batch_iter(generator, batch_size=batch_size):

      inds, imgs = [x[0] for x in batch], [x[1] for x in batch]

      # Get activations (middle of image)
      acts = t_layer.eval({t_img: imgs})
      if center_only:
        acts = acts[:, acts.shape[1]//2, acts.shape[2]//2]
      if dtype is not None:
        acts = acts.astype(dtype)

      # On the first iteration of the loop, create objects to hold responses
      # (we wanted to know acts.shape[-1] before creating it in the numpy case)
      if responses is None:
        # If we don't know what the extent of the indices will be in advance
        # we need to use a dictionary to support dynamic range
        if ind_shape is None:
          responses = {}
          count = defaultdict(lambda: 0)
        # But if we do, we can use a much more efficient numpy array
        else:
          responses = np.zeros(list(ind_shape) + list(acts.shape[1:]),
                               dtype=acts.dtype)
          count = np.zeros(ind_shape, dtype=acts.dtype)


      # Store each batch item in appropriate index, performing reduction
      for ind, act in zip(inds, acts):
        count[ind] += 1
        if ind in responses:
          responses[ind] = combiner(responses[ind], act)
        else:
          responses[ind] = act

  # complete reduction as necessary, then return
  # First the case where everything is in numpy
  if isinstance(responses, np.ndarray):
    count = np.maximum(count, 1e-6)[..., None]
    return normalizer(responses, count)
  # Then the dynamic shape dictionary case
  else:
    for k in responses:
      count_ = np.maximum(count[k], 1e-6)[None].astype(acts.dtype)
      responses[k] = normalizer(responses[k], count_)
    return dict_to_ndarray(responses)


def get_activations(model, layer, examples, batch_size=64,
                       dtype=None, ind_shape=None, center_only=False):
  """Collect center activtions of a layer over an n-dimensional array of images.

  Note: this is mostly intended for large synthetic families of images, where
    you can cheaply generate them in Python. For collecting activations over,
    say, ImageNet, there will be better workflows based on various dataset APIs
    in TensorFlow.

  Args:
    model: model for activations to be collected from.
    layer: layer (in model) for activtions to be collected from.
    examples: A (potentially n-dimensional) array of images. Can be any nested
      iterable object, including a generator, as long as the inner most objects
      are a numpy array with at least 3 dimensions (image X, Y, channels=3).
    batch_size: How many images should be processed at once?
    dtype: determines dtype of returned data (defaults to model activation
      dtype). Can be used to make function memory efficient.
    ind_shape: Shape that the index (non-image) dimensions of examples. Makes
      code much more memory efficient if examples is not a numpy array.

  Memory efficeincy:
    Have examples be a generator rather than an array of images; this allows
    them to be lazily generated and not all stored in memory at once. Also
    use ind_shape so that activations can be stored in an efficient data
    structure. If you still have memory problems, dtype="float16" can probably
    get you another 2x.

  Returns:
    A numpy array of shape [ind1, ind2, ..., layer_channels]
  """


  if ind_shape is None and isinstance(examples, np.ndarray):
    ind_shape = examples.shape[:-3]

  # Create a generator which recursive enumerates examples, stoppping at
  # the third last dimesnion (ie. an individual iamge) if numpy arrays.
  examples_enumerated = recursive_enumerate_nd(examples,
    stop_iter = lambda x: isinstance(x, np.ndarray) and len(x.shape) <= 3)

  # Get responses
  return get_activations_iter(model, layer, examples_enumerated,
                            batch_size=batch_size, dtype=dtype,
                            ind_shape=ind_shape, center_only=center_only)
