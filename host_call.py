"""
Copied from tensor2tensor library.

tensor2tensor/tensor2tensor/utils/t2t_model.py

"""

import tensorflow as tf
import collections
import six
import absl

_already_logged = set()


def _eager_log(level, *args):
  if tf.executing_eagerly() and args in _already_logged:
    return
  _already_logged.add(args)
  getattr(absl.logging, level)(*args)


def log_info(*args):
  _eager_log("info", *args)


def log_debug(*args):
  _eager_log("debug", *args)


def create_host_call(model_dir):
  """Construct a host_call writing scalar summaries.
  Args:
    model_dir: String containing path to train
  Returns:
    (fn, args) Pair to be called by TPUEstimator as the host_call.
  """
  graph = tf.compat.v1.get_default_graph()
  summaries = graph.get_collection(tf.compat.v1.GraphKeys.SUMMARIES)
  gs_t = tf.reshape(tf.cast(tf.compat.v1.train.get_global_step(), dtype=tf.int32), [1])
  summary_kwargs = collections.OrderedDict()
  for t in summaries:
    # TODO(aidangomez): enable ImageSummary support when we have a faster method
    # see @shibow's comment in cl/202344570
    if t.op.type not in ["ScalarSummary"]:
      tf.compat.v1.logging.warn("Ignoring unsupported tf.Summary type %s" % t.op.type)
      continue

    name = t.op.name
    tensor = t.op.inputs[1]
    if t.op.type == "ScalarSummary":
      assert tensor.shape.is_compatible_with([])
      if tensor.dtype == tf.int64:
        tensor = tf.cast(tensor, dtype=tf.int32)
      summary_kwargs["ScalarSummary" + name] = tf.reshape(tensor, [1])
    elif t.op.type == "ImageSummary":
      # TODO(aidangomez): as we move to support more types, update
      # common_layers.tpu_safe_image_summary
      if tensor.dtype != tf.float32:
        tf.compat.v1.logging.warn(
            "Currently T2T on TPU only supports ImageSummary of "
            "tf.float32-type Tensors. Skipping Tensor "
            "%s with dtype %s..." % (tensor.name, tensor.dtype))
        continue
      # tensor = tf.to_float(tensor)
      summary_kwargs["ImageSummary" + name] = tensor
  # When no supported summaries are found, don't create host_call. Otherwise,
  # TPU outfeed queue would enqueue global_step while host_call doesn't dequeue
  # it, eventually causing hang.
  if not summary_kwargs:
    return None
  summary_kwargs["global_step"] = gs_t
  log_info("summary_kwargs %s" % str(summary_kwargs))

  def host_call_fn(**kwargs):
    """Training host call. Creates summaries for training metrics.
    Args:
      **kwargs: Dict of {str: Tensor} , with `Tensor` of shape `[batch]`. Must
        contain key "global_step" with value of current global_step Tensor.
    Returns:
      List of summary ops to run on the CPU host.
    """
    gs = tf.cast(kwargs.pop("global_step")[0], dtype=tf.int64)
    with tf.compat.v2.summary.create_file_writer(logdir=model_dir).as_default():
      with tf.compat.v2.summary.record_if(True):
        # We need to use tf.contrib.summary in order to feed the `step`.
        for name, value in sorted(six.iteritems(kwargs)):
          if name.startswith("ScalarSummary"):
            name = name[len("ScalarSummary"):]
            tf.compat.v2.summary.scalar(
                name=name, data=tf.reduce_mean(input_tensor=tf.cast(value, dtype=tf.float32)), step=gs)
          elif name.startswith("ImageSummary"):
            name = name[len("ImageSummary"):]
            tf.compat.v2.summary.image(name=name, data=value, step=gs)

        return tf.compat.v1.summary.all_v2_summary_ops()

  return (host_call_fn, summary_kwargs)


def remove_summaries():
  """Remove summaries from the default graph."""
  g = tf.compat.v1.get_default_graph()
  key = tf.compat.v1.GraphKeys.SUMMARIES
  log_debug("Remove summaries %s" % str(g.get_collection(key)))
  del g.get_collection_ref(key)[:]
  assert not g.get_collection(key)