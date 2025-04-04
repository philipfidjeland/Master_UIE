

#
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
#

from collections import OrderedDict, ChainMap
import json
import torch
import weakref
from pytorch_nndct.utils.jit_utils import *
from nndct_shared.nndct_graph import GraphBase, NodeBase
from typing import Union

_NODE_NAME_SEPERATOR = '/'

class TorchGraph(object):
  _graph_id = 0
  def __init__(self, graph_name):
    super().__init__()
    self._name = graph_name
    self._nodes = []
    self._blocks = []
    self._blob_values = {}
    self._param_values = {}
    
    self._top_block = None
  @classmethod
  def new_graph(cls, graph_name):
    name = graph_name if graph_name else f"NndctGraph{TorchGraph._graph_id}" 
    TorchGraph._graph_id += 1
    return cls(name)
  
  
  def __str__(self):
    return json.dumps(self.description(), indent=4, separators=(',', ': '))
 
  def description(self):
    graph_des = {}
    graph_des['graph_name'] = "{}/{}:".format(self.__class__.__name__, self._name)
    graph_des['nodes'] = []
    for n in sorted(self.nodes, key=lambda n: n.block_idx):
      graph_des['nodes'].append(n.description())
    
    return graph_des
    
  def add_node(self, node):
    self._nodes.append(node)

  def add_blob_value(self, value):
    self._blob_values[value.name] = value

  def add_param_value(self, value):
    self._param_values[value.name] = value

  def param_names(self):
    for key in self._param_values.keys():
      yield key

  def blobs_name(self):
    for key in self._blob_values.keys():
      yield key

  def get_blob_value_by_name(self, name: str):
    return self._blob_values[name]

  def get_param_value_by_name(self, name: str):
    if isinstance(name, list):
      return [self._param_values[n] for n in name]
    else:
      return self._param_values[name]
 
            
  @property
  def nodes(self):
    return self._top_block.nodes
    

  @property
  def name(self):
    return self._name
  
  def children(self, node):
    return node.out_nodes
      
  def parents(self, node):
    return node.in_nodes  
  
  @property
  def op_types(self):
    return {node.kind for node in self.nodes}
  
  @property
  def top_block(self):
    return self._top_block
  
  def param_values(self):
    return list(self._param_values.values())
  
  @property
  def head_node(self):
    return self._top_block.head_node
  
  @property
  def return_node(self):
    return self._top_block.return_node

    
  def free_node(self, node):
    node.owning_graph = None
    self._nodes.remove(node)
  
  
  def connect_nodes(self):
    for nodeA in self._nodes:
      for ip in nodeA.flatten_inputs:
        for nodeB in self._nodes:
          if nodeB is not nodeA and ip in nodeB.outputs:
            nodeB.add_out_node(nodeA)
            nodeA.add_in_node(ip.node)
  
  def set_top_block(self, top_block):
    self._top_block = top_block
    
  
  def append_node(self, node):
    self._top_block.append_node(node)
  
  def add_block(self, block):
    self._blocks.append(block)
    
class TorchBlock(GraphBase):
  def __init__(self, graph, node):
    self._graph = graph
    self._node = node
    self._graph.add_block(self)
    self._nodes = []
  
  
  def children(self, node):
    return node.out_nodes
      
  def parents(self, node):
    return node.in_nodes  
  
  @property
  def op_types(self):
    return {node.kind for node in self.nodes}
  
  @property
  def head_node(self):
    return self._nodes[0]
  
  @property
  def return_node(self):
    return self._nodes[-1]
  
  @property
  def nodes(self):
    return self._nodes
  
  @property
  def owning_graph(self):
    return self._graph
  
  @property
  def owning_node(self):
    return self._node
  
  def append_node(self, node):
    node.owning_block = self
    self._nodes.append(node)
    
  def free_node(self, node):
    node.owning_block = None
    self._nodes.remove(node)
    
  def clean_nodes(self):
    self._nodes.clear()

  def reconnect_nodes(self):
    for node in self.nodes:
      node.clean_connection()
    self._graph.connect_nodes()
    
  def insert_node_with_idx(self, node, idx):
    node.owning_block = self
    original_node = self._nodes[idx]
    self._nodes[idx] = node
    self._graph.free_node(original_node)


class TorchValue(object): 
  
  def __init__(self, value, name=None):
    
    if isinstance(value, torch.Value):
      self._name = unique_name(value) if name is None else name
      self._scope_name = value.node().scopeName()
      self._node = None
      self._shape = list(
          value.type().sizes()) if value.isCompleteTensor() else None
      self._is_none = False
      self._is_plain_value = False
      self._type = str(value.type())
      self._data = None
      if value.node().mustBeNone():
        self._is_none = True
      elif value.node().kind().split("::")[-1] == "Constant":
        self._data = get_attr_value(value.node(), "value")
        if self._type != "Tensor":
          self._is_plain_value = True
        
    elif isinstance(value, (float, int, bool, tuple, list)):
      self._name = name
      self._scope_name = ''
      self._node = None
      self._shape = None
      self._is_none = False
      self._is_plain_value = True
      # self._type = {float: 'float', 
      #               int: 'int',  
      #               bool: 'bool'}.get(type(value), None)
      self._type = str(type(value))
      self._data = value
    else:
      raise RuntimeError(f"The type of value ({type(value)}) is unkown.")

    self._init_dtype(value)
    self._layout = None
    
  def _init_dtype(self, value):
    known_types = ['int', 'long', 'float', 'double', 'bool']
    if isinstance(value, torch.Value) and self.is_tensor() and value.isCompleteTensor():
      dtype = value.type().scalarType().lower()
    elif self.is_none():
      dtype = None
    else:
      dtype = self._type.lower()

    if dtype is None:
      self._dtype = None
    else:
      self._dtype = '.'.join(['torch', dtype
                             ]) if dtype in known_types else dtype

  def is_tensor(self):
    return self._type == 'Tensor'

  def is_plain_value(self):
    return self._is_plain_value
  
  def convert_plain_value_to_tensor(self):
    self._is_plain_value = False
    # self._dtype = self.dtype 
    self._type = 'Tensor'

  def is_none(self):
    return self._is_none

  @property
  def name(self):
    return self._name

  @property
  def node(self):
    return self._node() if self._node else self._node

  @node.setter
  def node(self, value):
    self._node = weakref.ref(value) if value else value

  @property
  def dtype(self):
    return self._dtype
  
  @dtype.setter
  def dtype(self, dtype):
    self._dtype = dtype

  @property
  def scope_name(self):
    return self._scope_name

  @property
  def shape(self):
    return self._shape

  @shape.setter
  def shape(self, shape):
    self._shape = shape
  
  @property
  def ndim(self):
    if self._shape is not None:
      return len(self._shape)

  @property
  def layout(self):
    return self._layout
  
  @layout.setter
  def layout(self, layout):
    self._layout = layout
    
    
  @property
  def data(self):
    return self._data

  @data.setter
  def data(self, data):
    self._data = data
    
    
class TorchNode(NodeBase):

  def __init__(self, node: torch.Node = None):
    self._kind = None
    self._is_custom_pyop = False
    self._pyobj = None
    self._scope_name = ""
    self._source_range = ""
     
    if node:
      if node.kind().split("::")[-1] != "PythonOp":
        self._kind = node.kind().split("::")[-1]
      else:
        self._kind = node.pyname()
        import pytorch_nndct.nn.modules.function as fn
        import inspect
        native_fn = [obj.__name__ for _, obj in inspect.getmembers(fn) if inspect.isclass(obj)]
        if self._kind not in native_fn:
          self._is_custom_pyop = True
          self._pyobj = node.pyobj()
      self._scope_name = node.scopeName()
      self._source_range = node.sourceRange()
          
    self._inputs = []
    self._outputs = []
    self._in_nodes = []
    self._out_nodes = []
    self._blocks = []
    self._dtype = "unknown"
    self._schema = None
  
   

  def __str__(self):
    return json.dumps(self.description(), indent=4, separators=(',', ': '))

  def description(self):
    node_des = {}
    node_des['name'] = self.name
    node_des['block_index'] = self.block_idx
    node_des['kind'] = self.kind
    node_des['dtype'] = self.dtype
    node_des['in_nodes'] = [i.name for i in self._in_nodes]
    node_des['out_nodes'] = [o.name for o in self._out_nodes]

    node_des['in_value'] = []

    def append_in_value(ip):
      if isinstance(ip, (list, tuple)):
        for value in ip:
          append_in_value(value)
      else:
        node_des['in_value'].append(ip.name)
    
    for ip in self._inputs:
      append_in_value(ip)

    node_des['out_value'] = [ot.name for ot in self._outputs]
    
    if self._blocks:
      for i, block in enumerate(self._blocks):
        node_des[f'block_{i}'] = []
        for n in sorted(block._nodes, key=lambda n: n.idx):
          node_des[f'block_{i}'].append(n.description())
    return node_des

  def add_input(self, value):
    self._inputs.append(value)

  def add_output(self, value):
    self._outputs.append(value)
    value.node = self
    self.owning_graph.add_blob_value(value)
  

  def add_out_node(self, node):
    self._out_nodes.append(node)

  def add_in_node(self, node):
    self._in_nodes.append(node)

  def clean_connection(self):
    self._out_nodes.clear()
    self._in_nodes.clear()
  
  
  @property
  def blocks(self):
    return self._blocks
  
  def add_block(self, block):
    self._blocks.append(block)

  @property
  def inputs(self):
    return self._inputs

  @property
  def flatten_inputs(self):
    def _flatten_inputs(inputs):
      if isinstance(inputs, (list, tuple)):
        for i in inputs:
          yield from _flatten_inputs(i)
      else:
        yield inputs
        
    for ip in self._inputs:
      yield from _flatten_inputs(ip)
    

  @property
  def outputs(self):
    return self._outputs

  @outputs.setter
  def outputs(self, outputs):
    self._outputs = outputs

  @property
  def block_idx(self):
    return self.owning_block.nodes.index(self)


  @property
  def kind(self):
    return self._kind

  @kind.setter
  def kind(self, kind):
    self._kind = kind
    
  @property
  def op_type(self):
    return self._kind
  
  @property
  def in_nodes(self):
    return self._in_nodes

  @property
  def out_nodes(self):
    return self._out_nodes

  @property
  def dtype(self):
    if self._dtype == "unkown":
      if self.outputs and self.outputs[0].dtype == "unknown" or not self.outputs:
        dtype = list(self.flatten_inputs)[0].dtype
      else:
        dtype = self.outputs[0].dtype
      if dtype == "unknown":
        raise RuntimeError(f"Can't infer dtype for node({self.name}, kind={self.kind})")
      return dtype
    else:
      return self._dtype

  @dtype.setter
  def dtype(self, value):
    self._dtype = value

  @property
  def name(self):
    if hasattr(self, "_name"):
      return getattr(self, "_name")
    else:
      if self.outputs:
        if self.outputs[0].scope_name:
          return _NODE_NAME_SEPERATOR.join(
            [self.outputs[0].scope_name, self.outputs[0].name])
        else:
          return self.outputs[0].name
      else:
        raise RuntimeError("The node must have a name")


  @name.setter
  def name(self, name):
    setattr(self, "_name", name)
    
  @property
  def schema(self):
    return self._schema
  
  @schema.setter
  def schema(self, schema):
    self._schema = schema
    
    
  @property
  def is_custom_pyop(self):
    return self._is_custom_pyop
  
  @property
  def pyobj(self):
    return self._pyobj


  @property
  def owning_graph(self):
    return self._graph
  
  @owning_graph.setter
  def owning_graph(self, graph):
    self._graph = graph
    if self._graph:
      self._graph.add_node(self)
  @property
  def owning_block(self):
    return self._block
  
  @owning_block.setter
  def owning_block(self, block):
    self._block = block
    
  def destroy(self):
    self.owning_block.free_node(self)
    self.owning_graph.free_node(self)


  @property
  def scope_name(self):
    return self._scope_name
  
  @scope_name.setter
  def scope_name(self, scope_name):
    self._scope_name = scope_name
  
  @property
  def source_range(self):
    return self._source_range
  
  @source_range.setter
  def source_range(self, source_range):
    self._source_range = source_range

  
  
        

