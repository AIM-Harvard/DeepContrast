"""Imports a protobuf model as a graph in Tensorboard."""


from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.summary import summary
from tensorflow.python.tools import saved_model_utils

# Try importing TensorRT ops if available
# TODO(aaroey): ideally we should import everything from contrib, but currently
# tensorrt module would cause build errors when being imported in
# tensorflow/contrib/__init__.py. Fix it.
# pylint: disable=unused-import,g-import-not-at-top,wildcard-import
try:
  from tensorflow.contrib.tensorrt.ops.gen_trt_engine_op import *
except ImportError:
  pass
# pylint: enable=unused-import,g-import-not-at-top,wildcard-import


def import_to_tensorboard(model_dir, log_dir, tag_set):
  """View an SavedModel as a graph in Tensorboard.
  Args:
    model_dir: The directory containing the SavedModel to import.
    log_dir: The location for the Tensorboard log to begin visualization from.
    tag_set: Group of tag(s) of the MetaGraphDef to load, in string format,
      separated by ','. For tag-set contains multiple tags, all tags must be
      passed in.
  Usage: Call this function with your SavedModel location and desired log
    directory. Launch Tensorboard by pointing it to the log directory. View your
    imported SavedModel as a graph.
  """
  with session.Session(graph=ops.Graph()) as sess:
    input_graph_def = saved_model_utils.get_meta_graph_def(model_dir,
                                                           tag_set).graph_def
    importer.import_graph_def(input_graph_def)

    pb_visual_writer = summary.FileWriter(log_dir)
    pb_visual_writer.add_graph(sess.graph)
    print("Model Imported. Visualize by running: "
          "tensorboard --logdir={}".format(log_dir))



import_to_tensorboard(model_dir, log_dir, tag_set)


