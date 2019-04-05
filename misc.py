import tensorflow as tf
import numpy as np

Number2Base = {0:'A', 1:'C', 2:'G', 3:'T'}

def filter2motif(a):
  '''Translate filter to motif of most weighted bases [positive, negative]'''
  d={0:'A', 1:'C', 2:'G', 3:'T'}
  return [ "".join([d[i] for i in np.argmax(a, axis=1)]), "".join([d[i] for i in np.argmin(a, axis=1)]) ]


def input_layers_of(layer):
  ''' returns the input layers of a given layer '''
  return layer.inbound_nodes[0].inbound_layers

def getLayerIndexByName(model, layername):
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx

def allow_growth_for_default_session():
  ''' Necessary in new version of tensorflow to make sure, that the memory is dinamicaly alocated.'''
  from tensorflow.compat.v1 import ConfigProto
  from tensorflow.compat.v1 import InteractiveSession
  config = ConfigProto()
  config.gpu_options.allow_growth = True
  session = InteractiveSession()
