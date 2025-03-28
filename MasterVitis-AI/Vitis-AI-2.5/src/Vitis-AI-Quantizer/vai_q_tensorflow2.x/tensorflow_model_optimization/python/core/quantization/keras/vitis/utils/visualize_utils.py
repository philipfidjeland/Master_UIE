# Copyright 2019 Xilinx Inc.
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
"""Model Visualization Utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import re
import tensorflow as tf
from keras import activations
from keras.utils import io_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

logger = common_utils.VAILogger
keras = tf.keras

try:
  # pydot-ng is a fork of pydot that is better maintained.
  import pydot_ng as pydot
except ImportError:
  # pydotplus is an improved version of pydot
  try:
    import pydotplus as pydot
  except ImportError:
    # Fall back on pydot if necessary.
    try:
      import pydot
    except ImportError:
      pydot = None


def check_pydot():
  """Returns True if PyDot and Graphviz are available."""
  if pydot is None:
    return False
  try:
    # Attempt to create an image of a blank graph
    # to check the pydot/graphviz installation.
    pydot.Dot.create(pydot.Dot())
    return True
  except (OSError, pydot.InvocationException):
    return False


def is_wrapped_model(layer):
  from keras.engine import functional
  from keras.layers import wrappers
  return (isinstance(layer, wrappers.Wrapper) and
          isinstance(layer.layer, functional.Functional))


def add_edge(dot, src, dst):
  if not dot.get_edge(src, dst):
    dot.add_edge(pydot.Edge(src, dst))


def get_layer_index_bound_by_layer_name(model, layer_names):
  """Return specific range of layers to plot, mainly for sub-graph plot models.

  Args:
    model: tf.keras.Model
    layer_names: unique name of layer of the model, type(str)

  Returns:
    return the index value of layer based on its unique name (layer_names)
  """
  lower_index = []
  upper_index = []
  for idx, layer in enumerate(model.layers):
    if re.match(layer_names[0], layer.name):
      lower_index.append(idx)
    if re.match(layer_names[1], layer.name):
      upper_index.append(idx)
  if not lower_index or not upper_index:
    raise ValueError(
        'Passed layer_names does not match to layers in the model. '
        f'Recieved: {layer_names}')
  if min(lower_index) > max(upper_index):
    return [min(upper_index), max(lower_index)]
  return [min(lower_index), max(upper_index)]


def model_to_dot(model,
                 show_shapes=False,
                 show_dtype=False,
                 show_layer_names=True,
                 rankdir='TB',
                 expand_nested=False,
                 dpi=None,
                 subgraph=False,
                 layer_range=None,
                 show_layer_activations=False,
                 inspect_results=None,
                 show_inspect_results=True):
  """Convert a Keras model to dot format.

  Args:
    model: A Keras model instance.
    show_shapes: whether to display shape information.
    show_dtype: whether to display layer dtypes.
    show_layer_names: whether to display layer names.
    rankdir: `rankdir` argument passed to PyDot,
        a string specifying the format of the plot:
        'TB' creates a vertical plot;
        'LR' creates a horizontal plot.
    expand_nested: whether to expand nested models into clusters.
    dpi: Dots per inch.
    subgraph: whether to return a `pydot.Cluster` instance.
    layer_range: input of `list` containing two `str` items, which is the
        starting layer name and ending layer name (both inclusive) indicating
        the range of layers for which the `pydot.Dot` will be generated. It
        also accepts regex patterns instead of exact name. In such case, start
        predicate will be the first element it matches to `layer_range[0]`
        and the end predicate will be the last element it matches to
        `layer_range[1]`. By default `None` which considers all layers of
        model. Note that you must pass range such that the resultant subgraph
        must be complete.
    show_layer_activations: Display layer activations (only for layers that
        have an `activation` property).
    inspect_results: Dict contains inspect results for each layer.
    show_inspect_results: whether to show inspect results.

  Returns:
    A `pydot.Dot` instance representing the Keras model or
    a `pydot.Cluster` instance representing nested model if
    `subgraph=True`.

  Raises:
    ValueError: if `model_to_dot` is called before the model is built.
    ImportError: if graphviz or pydot are not available.
  """

  if not model.built:
    raise ValueError('This model has not yet been built. '
                     'Build the model first by calling `build()` or by calling '
                     'the model on a batch of data.')

  from keras.layers import wrappers
  from keras.engine import sequential
  from keras.engine import functional

  if not check_pydot():
    message = ('You must install pydot (`pip install pydot`) '
               'and install graphviz '
               '(see instructions at https://graphviz.gitlab.io/download/) '
               'for plot_model/model_to_dot to work.')
    if 'IPython.core.magics.namespace' in sys.modules:
      # We don't raise an exception here in order to avoid crashing notebook
      # tests where graphviz is not available.
      io_utils.print_msg(message)
      return
    else:
      raise ImportError(message)

  if subgraph:
    dot = pydot.Cluster(style='dashed', graph_name=model.name)
    dot.set('label', model.name)
    dot.set('labeljust', 'l')
  else:
    dot = pydot.Dot()
    dot.set('rankdir', rankdir)
    dot.set('concentrate', True)
    #  dot.set('dpi', dpi)
    dot.set_node_defaults(shape='Mrecord')

  if layer_range is not None:
    if len(layer_range) != 2:
      raise ValueError(
          'layer_range must be of shape (2,). Received: '
          f'layer_range = {layer_range} of length {len(layer_range)}')
    if (not isinstance(layer_range[0], str) or
        not isinstance(layer_range[1], str)):
      raise ValueError('layer_range should contain string type only. '
                       f'Received: {layer_range}')
    layer_range = get_layer_index_bound_by_layer_name(model, layer_range)
    if layer_range[0] < 0 or layer_range[1] > len(model.layers):
      raise ValueError('Both values in layer_range should be in range (0, '
                       f'{len(model.layers)}. Received: {layer_range}')

  sub_n_first_node = {}
  sub_n_last_node = {}
  sub_w_first_node = {}
  sub_w_last_node = {}

  layers = model.layers
  if not model._is_graph_network:
    node = pydot.Node(str(id(model)), label=model.name)
    dot.add_node(node)
    return dot
  elif isinstance(model, sequential.Sequential):
    if not model.built:
      model.build()
    layers = super(sequential.Sequential, model).layers

  # Create graph nodes.
  for i, layer in enumerate(layers):
    if (layer_range) and (i < layer_range[0] or i > layer_range[1]):
      continue

    layer_id = str(id(layer))

    # Append a wrapped layer's label to node's label, if it exists.
    layer_name = layer.name
    class_name = layer.__class__.__name__

    if isinstance(layer, wrappers.Wrapper):
      if expand_nested and isinstance(layer.layer, functional.Functional):
        submodel_wrapper = model_to_dot(
            layer.layer,
            show_shapes,
            show_dtype,
            show_layer_names,
            rankdir,
            expand_nested,
            subgraph=True)
        # sub_w : submodel_wrapper
        sub_w_nodes = submodel_wrapper.get_nodes()
        sub_w_first_node[layer.layer.name] = sub_w_nodes[0]
        sub_w_last_node[layer.layer.name] = sub_w_nodes[-1]
        dot.add_subgraph(submodel_wrapper)
      else:
        layer_name = '{}({})'.format(layer_name, layer.layer.name)
        child_class_name = layer.layer.__class__.__name__
        class_name = '{}({})'.format(class_name, child_class_name)

    if expand_nested and isinstance(layer, functional.Functional):
      submodel_not_wrapper = model_to_dot(
          layer,
          show_shapes,
          show_dtype,
          show_layer_names,
          rankdir,
          expand_nested,
          subgraph=True)
      # sub_n : submodel_not_wrapper
      sub_n_nodes = submodel_not_wrapper.get_nodes()
      sub_n_first_node[layer.name] = sub_n_nodes[0]
      sub_n_last_node[layer.name] = sub_n_nodes[-1]
      dot.add_subgraph(submodel_not_wrapper)

    # Create node's label.
    label = '{Type: %s}' % class_name

    # Rebuild the label as a table including the layer's activation.
    if (show_layer_activations and hasattr(layer, 'activation') and
        layer.activation is not None):
      label = '{%s|Act: %s}' % (label, activations.serialize(layer.activation))

    # Rebuild the label as a table including the layer's name.
    if show_layer_names:
      label = 'Name: %s|%s' % (layer_name, label)

    # Rebuild the label as a table including the layer's dtype.
    if show_dtype:

      def format_dtype(dtype):
        if dtype is None:
          return '?'
        else:
          return str(dtype)

      label = '%s|%s' % (label, format_dtype(layer.dtype))

    # Rebuild the label as a table including input/output shapes.
    if show_shapes:

      def format_shape(shape):
        return str(shape).replace(str(None), 'None')

      try:
        outputlabels = format_shape(layer.output_shape)
      except AttributeError:
        outputlabels = '?'
      if hasattr(layer, 'input_shape'):
        inputlabels = format_shape(layer.input_shape)
      elif hasattr(layer, 'input_shapes'):
        inputlabels = ', '.join(
            [format_shape(ishape) for ishape in layer.input_shapes])
      else:
        inputlabels = '?'
      label = '{%s}|{input:|output:}|{{%s}}|{{%s}}' % (label, inputlabels,
                                                       outputlabels)
    if show_inspect_results:
      res = inspect_results.get(layer, None)
      if res:
        device = inspect_results[layer].device
        notes = [s.strip('.') for s in res.get_notes()]
        tooltip = None
        if notes:
          tooltip = '; '.join(notes)
      else:
        device = 'None'
      label = '{%s}|{Device: %s}' % (label, device)
      color = 'lightskyblue' if device in ['DPU', 'INPUT'] else 'firebrick1'

    if not expand_nested or not isinstance(layer, functional.Functional):
      if show_inspect_results:
        if tooltip:
          node = pydot.Node(
              layer_id,
              label=label,
              style='filled',
              fillcolor=color,
              tooltip=tooltip)
        else:
          node = pydot.Node(
              layer_id,
              label=label,
              style='filled',
              fillcolor=color,
              tooltip=layer_name)
      else:
        node = pydot.Node(layer_id, label=label, color=None)
      dot.add_node(node)

  # Connect nodes with edges.
  for i, layer in enumerate(layers):
    if (layer_range) and (i <= layer_range[0] or i > layer_range[1]):
      continue
    layer_id = str(id(layer))
    for i, node in enumerate(layer._inbound_nodes):
      node_key = layer.name + '_ib-' + str(i)
      if node_key in model._network_nodes:
        for inbound_layer in tf.nest.flatten(node.inbound_layers):
          inbound_layer_id = str(id(inbound_layer))
          if not expand_nested:
            assert dot.get_node(inbound_layer_id)
            assert dot.get_node(layer_id)
            add_edge(dot, inbound_layer_id, layer_id)
          else:
            # if inbound_layer is not Model or wrapped Model
            if (not isinstance(inbound_layer, functional.Functional) and
                not is_wrapped_model(inbound_layer)):
              # if current layer is not Model or wrapped Model
              if (not isinstance(layer, functional.Functional) and
                  not is_wrapped_model(layer)):
                assert dot.get_node(inbound_layer_id)
                assert dot.get_node(layer_id)
                add_edge(dot, inbound_layer_id, layer_id)
              # if current layer is Model
              elif isinstance(layer, functional.Functional):
                add_edge(dot, inbound_layer_id,
                         sub_n_first_node[layer.name].get_name())
              # if current layer is wrapped Model
              elif is_wrapped_model(layer):
                add_edge(dot, inbound_layer_id, layer_id)
                name = sub_w_first_node[layer.layer.name].get_name()
                add_edge(dot, layer_id, name)
            # if inbound_layer is Model
            elif isinstance(inbound_layer, functional.Functional):
              name = sub_n_last_node[inbound_layer.name].get_name()
              if isinstance(layer, functional.Functional):
                output_name = sub_n_first_node[layer.name].get_name()
                add_edge(dot, name, output_name)
              else:
                add_edge(dot, name, layer_id)
            # if inbound_layer is wrapped Model
            elif is_wrapped_model(inbound_layer):
              inbound_layer_name = inbound_layer.layer.name
              add_edge(dot, sub_w_last_node[inbound_layer_name].get_name(),
                       layer_id)
  return dot


def plot_model(model,
               to_file='model.svg',
               show_shapes=False,
               show_dtype=False,
               show_layer_names=True,
               rankdir='TB',
               expand_nested=False,
               dpi=None,
               layer_range=None,
               show_layer_activations=True,
               inspect_results=None,
               show_inspect_results=True):
  """Converts a Keras model to dot format and save to a file.

  Example:

  ```python
  input = tf.keras.Input(shape=(100,), dtype='int32', name='input')
  x = tf.keras.layers.Embedding(
      output_dim=512, input_dim=10000, input_length=100)(input)
  x = tf.keras.layers.LSTM(32)(x)
  x = tf.keras.layers.Dense(64, activation='relu')(x)
  x = tf.keras.layers.Dense(64, activation='relu')(x)
  x = tf.keras.layers.Dense(64, activation='relu')(x)
  output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
  model = tf.keras.Model(inputs=[input], outputs=[output])
  dot_img_file = '/tmp/model_1.png'
  tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
  ```

  Args:
    model: A Keras model instance
    to_file: File name of the plot image.
    show_shapes: whether to display shape information.
    show_dtype: whether to display layer dtypes.
    show_layer_names: whether to display layer names.
    rankdir: `rankdir` argument passed to PyDot,
        a string specifying the format of the plot: 'TB' creates a vertical
          plot; 'LR' creates a horizontal plot.
    expand_nested: Whether to expand nested models into clusters.
    dpi: Dots per inch.
    layer_range: input of `list` containing two `str` items, which is the
      starting layer name and ending layer name (both inclusive) indicating the
      range of layers for which the plot will be generated. It also accepts
      regex patterns instead of exact name. In such case, start predicate will
      be the first element it matches to `layer_range[0]` and the end predicate
      will be the last element it matches to `layer_range[1]`. By default `None`
      which considers all layers of model. Note that you must pass range such
      that the resultant subgraph must be complete.
    show_layer_activations: Display layer activations (only for layers that
      have an `activation` property).
    inspect_results: Dict contains inspect results for each layer.
    show_inspect_results: whether to show inspect results.

  Raises:
    ValueError: if `plot_model` is called before the model is built.

  Returns:
    A Jupyter notebook Image object if Jupyter is installed.
    This enables in-line display of the model plots in notebooks.
  """

  if not model.built:
    raise ValueError('This model has not yet been built. '
                     'Build the model first by calling `build()` or by calling '
                     'the model on a batch of data.')

  dot = model_to_dot(
      model,
      show_shapes=show_shapes,
      show_dtype=show_dtype,
      show_layer_names=show_layer_names,
      rankdir=rankdir,
      expand_nested=expand_nested,
      dpi=dpi,
      layer_range=layer_range,
      show_layer_activations=show_layer_activations,
      inspect_results=inspect_results,
      show_inspect_results=show_inspect_results)
  to_file = io_utils.path_to_string(to_file)
  if dot is None:
    return
  _, extension = os.path.splitext(to_file)
  if not extension:
    extension = 'png'
  else:
    extension = extension[1:]
  # Save image to disk.
  dot.write(to_file, format=extension)
  # Return the image as a Jupyter Image object, to be displayed in-line.
  # Note that we cannot easily detect whether the code is running in a
  # notebook, and thus we always return the Image if Jupyter is available.
  if extension != 'pdf':
    try:
      from IPython import display
      return display.Image(filename=to_file)
    except:
      pass
