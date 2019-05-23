import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import IPython
import matplotlib.pyplot as plt
import subprocess
import seaborn as sns

##
# @file
# This file implements various functions that can be used to disect and study the
# neural network trained using predict_promoters.py
#

from misc import *

allow_growth_for_default_session()

sns.set_style("darkgrid")

# ************************************************************************************
# INPUT DATA PREPARATION
# ************************************************************************************

from input_data_prep import *
inp = read_input_files(sys.argv[1])
globals().update(inp)

out = 7
hidden = 6
flat = 5
filter_directory = "./filters_easy_model/"
TSS_pos = 500

def neuron_represents(layer, nwl):
  prev = input_layers_of(layer)[0]
  ret = {}
  if layer.name == "seq_direct-flatten":
    ret['type'] = "nucl"
    ret['pos'] = (nwl / 4) - TSS_pos
    ret['base'] = Number2Base[nwl % 4]
    ret['layer'] = prev
    return ret
  elif layer.name == "seq_conv-flatten":
    prevprev = input_layers_of(prev)[0]
    pool_strides = prev.get_config()['strides'][0]
    filters = prevprev.get_config()['filters']
    ret["filters"] = filters
    ret['type'] = "motif"
    ret['pos'] = (nwl / filters) * pool_strides - TSS_pos
    ret['stride'] = pool_strides
    ret['filter_num'] = nwl % filters
    ret['filter'] = prevprev.get_weights()[0][:,:,ret['filter_num']]
    ret['layer'] = prevprev
    return ret
  elif layer.name == "seq_2conv_flatten":
    conv2 = prev
    maxpool = input_layers_of(prev)[0]
    conv = input_layers_of(maxpool)[0]
    #IPython.embed()
    pool_strides = maxpool.get_config()['strides'][0]
    filters = conv2.get_config()['filters']
    ret["filters"] = filters
    ret['type'] = "motif collection"
    ret['pos'] = (nwl / filters) * pool_strides - TSS_pos
    ret['stride'] = pool_strides
    ret['filter_num'] = nwl % filters
    ret['filter'] = conv2.get_weights()[0][:,:,ret['filter_num']]
    ret['layer'] = conv2
    return ret
  elif layer.name == "CG_flatten":
    prevprev = input_layers_of(prev)[0]
    pool_strides = prev.get_config()['strides'][0]
    filters = prevprev.get_config()['filters']
    ret["filters"] = filters
    ret['type'] = "CG"
    ret['pos'] = (nwl / filters) * pool_strides - TSS_pos
    ret['stride'] = pool_strides
    ret['filter_num'] = nwl % filters
    ret['filter'] = prevprev.get_weights()[0][:,:,ret['filter_num']]
    ret['layer'] = prevprev
    return ret
  if layer.name == "CG-avg_flatten":
    pool_strides = prev.get_config()['strides'][0]
    ret['type'] = "CG-avg"
    ret['pos'] = nwl * pool_strides - TSS_pos
    ret['stride'] = pool_strides
    return ret  
  if layer.name == "CA_flatten":
    ret['type'] = "CA"
    ret['pos'] = nwl - 20
    ret['layer'] = prev
    return ret
  elif layer.name == "COV_flatten":
    pool_strides = prev.get_config()['strides'][0]
    ret['type'] = "COV"
    ret['pos'] = nwl * pool_strides - TSS_pos
    ret['stride'] = pool_strides
    #ret['layer'] = prev
    return ret
  elif layer.name == "METH_flatten":
    pool_strides = prev.get_config()['strides'][0]
    ret['type'] = "METH"
    ret['pos'] = nwl * pool_strides - TSS_pos
    ret['stride'] = pool_strides
    #ret['layer'] = prev
    return ret
  elif layer.name == "SNPs_flatten":
    pool_strides = prev.get_config()['strides'][0]
    ret['type'] = "SNPs"
    ret['pos'] = nwl * pool_strides - TSS_pos
    ret['stride'] = pool_strides
    #ret['layer'] = prev
    return ret
  else:
    ret['type'] = "unknown"
    return ret
    
#  IPython.embed()
  #if layer.

def neuron_belongs_to(n):
  ''' Returns description of layer to which a neuron belongs to.'''
  
  for i,l in enumerate(model.layers):
    if l.name == "concatenate":
      concat_layer_num = i
  concat_layer = model.layers[concat_layer_num]
  layer_start = 0
  for inp_l in input_layers_of(concat_layer):
    if n < layer_start+inp_l.output_shape[1]:
      ret = {"start":layer_start,
             "layer":inp_l,
             "neurons":inp_l.output_shape[1],
             "represents": neuron_represents(inp_l,n-layer_start)
             }
      return ret
    else:
      layer_start += inp_l.output_shape[1]
  return None

def plot_filter(f, filename=None):
  ''' Plots convolutional filter of bases as barplot showing weights of bases on positions within motif.'''
  df = pd.DataFrame(f, columns=["A","C","G","T"])
  fig, ax = plt.subplots(1,1)
  df.plot.bar(ax=ax)
  title = "Consensus: {}(+) / {}(-)".format(filter2motif(f)[0], filter2motif(f)[1])
  plt.title(title)
  plt.xlabel("position")
  ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.subplots_adjust(bottom=0.10)
  if filename is None:
    plt.show()
  else:
    plt.savefig(filename)
    plt.close()

def plot_hyperfilter(f, filename=None):
  df = pd.DataFrame(f).transpose()
  so = df.sum(axis=1)
  df = df.loc[so.argsort()[::-1]]
  fig, ax = plt.subplots(figsize=(5,13)) 
  sns.heatmap(df,cmap="RdGy_r", yticklabels=1, ax=ax)
  plt.subplots_adjust(bottom=0.04, top=0.96)
  plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, fontsize=8)
  if filename is None:
    plt.show()
  else:
    plt.savefig(filename)
    plt.close()
  '''cbar_ax = plt.gca()
  cm = sns.clustermap(df, col_cluster=False, cmap="RdGy_r", yticklabels=1, figsize=(5,14))
  plt.setp(cm.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize=6)
  hm = cm.ax_heatmap.get_position()
  #cm.ax_heatmap.set_position([hm.x0, hm.y0, hm.width*0.25, hm.height])
  cm.ax_heatmap.set_position([hm.x0, hm.y0, hm.width, hm.height*1.4])
  col = cm.ax_col_dendrogram.get_position()
  cm.ax_col_dendrogram.set_position([col.x0, col.y0, col.width*0.25, col.height*0.5])'''

def neuron_label(neuron):
  '''Creates readable label for neuron of flat layer depending on which input layer corresponds to it. '''
  belongs = neuron_belongs_to(neuron)
  #print belongs
  nwl = neuron - belongs["start"] # neuron number within the layer

  if belongs["represents"]["type"]=="motif" :
    f = belongs["represents"]["filter_num"]
    pos = belongs["represents"]["pos"]
    stride = belongs["represents"]["stride"]
    plot_filter(belongs["represents"]["filter"],
                #belongs["represents"]["layer"].get_weights()[0][:,:,f],
                filename=filter_directory+"_f{}.png".format(f))
    return "filter:{} pos:<{};{}>".format(f,pos,pos+stride)
  
  elif belongs["represents"]["type"]=="motif collection" :
    mc = belongs["represents"]["filter_num"]
    pos = belongs["represents"]["pos"]
    stride = belongs["represents"]["stride"]
    plot_hyperfilter(belongs["represents"]["filter"],
                     filename=filter_directory+"_hf{}.png".format(mc))
    return "hyper-filter:{} pos:<{};{}>".format(mc,pos,pos+stride)
    
  elif belongs["represents"]["type"]=="nucl" :
    pos = belongs["represents"]["pos"]
    base = Number2Base[nwl%4]
    return "{} at {}".format(base,pos)

  elif belongs["represents"]["type"]=="CG" :
    f = belongs["represents"]["filter_num"]
    pos = belongs["represents"]["pos"]
    stride = belongs["represents"]["stride"]
    print f,belongs["represents"]["filter"]
    return "CG filter:{} pos:<{};{}>".format(f,pos,pos+stride)
  
  elif belongs["represents"]["type"]=="CG-avg" :
    pos = belongs["represents"]["pos"]
    stride = belongs["represents"]["stride"]
    return "Mean CG <{};{}>".format(pos,pos+stride)

  elif belongs["represents"]["type"]=="CA" :
    pos = belongs["represents"]["pos"]
    return "CA at {}".format(pos)

  elif belongs["represents"]["type"]=="COV" :
    pos = belongs["represents"]["pos"]
    stride = belongs["represents"]["stride"]
    return "Mean coverage <{};{}>".format(pos, pos+stride)
  
  elif belongs["represents"]["type"]=="METH" :
    pos = belongs["represents"]["pos"]
    stride = belongs["represents"]["stride"]
    return "Mean methylation <{};{}>".format(pos, pos+stride)
  
  elif belongs["represents"]["type"]=="SNPs" :
    pos = belongs["represents"]["pos"]
    stride = belongs["represents"]["stride"]
    return "Mean SNP <{};{}>".format(pos, pos+stride)
  else:
    return "Unknown layer type (neuron {})".format(neuron)
  
def important_for_layer(o, # output of neurons in the network
                        layer, # important neurons of this layer we already have
                        important_pos, # list of positively influencing neurons from this layer
                        important_neg, # list of negatively influencing neurons from this layer
                        prev,  # what are the important neurons of the prev layer
                        ratio = 0.3): # what proportion of max weight
  ''' As input takes:
         - Output values of whole network,
         - Layer number (important neurons of this layer we already have)
         - List of positively influencing neurons from this layer
         - List of negatively influencing neurons from this layer
         - Number of the previous layer. We're finding important neurons in this layer
         - [optional] propotion of max. contribution to neuron's value to be considered important
           default 0.3 means that all neurons having abs(activation) at least 0.3*abs(max_activation)
           are considered important
      Returns:
         - list of neurons in "prev" layer supporting result "region is a promoter" 
           by having POSITIVE value
         - list of neurons in "prev" layer supporting result "region is a promoter"
           by having NEGATIVE value
         - list of neurons in "prev" layer supporting result "region is NOT a promoter"
           by having POSITIVE value
         - list of neurons in "prev" layer supporting result "region is NOT a promoter"
           by having NEGATIVE value
  '''
         
  pos_p = set()
  pos_n = set()
  neg_p = set()
  neg_n = set()
  for i in important_pos:
    act = o[prev][0]*w[layer][0][:,i]
#    print act
    maxact = np.max(np.abs(act))
    t_act = np.abs(maxact)*ratio # threshold to be considered important for i-th neuron
#    print t_act
    important_pos_p_i = set(np.argwhere((act>t_act)&(o[prev][0]>0))[:,0])
    important_pos_n_i = set(np.argwhere((act>t_act)&(o[prev][0]<0))[:,0])
    important_neg_p_i = set(np.argwhere((act< -t_act)&(o[prev][0]>0))[:,0])
    important_neg_n_i = set(np.argwhere((act< -t_act)&(o[prev][0]<0))[:,0])
    pos_p = pos_p.union(important_pos_p_i)
    pos_n = pos_n.union(important_pos_n_i)
    neg_p = neg_p.union(important_neg_p_i)
    neg_n = neg_n.union(important_neg_n_i)
  
  for i in important_neg:
    act = o[prev][0]*w[layer][0][:,i]
    maxact = np.max(np.abs(act))
    t_act = np.abs(maxact)*ratio # threshold to be considered important for i-th neuron
    important_pos_p_i = set(np.argwhere((act < -t_act)&(o[prev][0]>0))[:,0])
    important_pos_n_i = set(np.argwhere((act < -t_act)&(o[prev][0]<0))[:,0])
    important_neg_p_i = set(np.argwhere((act > t_act)&(o[prev][0]>0))[:,0])
    important_neg_n_i = set(np.argwhere((act > t_act)&(o[prev][0]<0))[:,0])
    pos_p = pos_p.union(important_pos_p_i)
    pos_n = pos_n.union(important_pos_n_i)
    neg_p = neg_p.union(important_neg_p_i)
    neg_n = neg_n.union(important_neg_n_i)
    
  return (list(pos_p),list(pos_n),list(neg_p),list(neg_n))


def activation_contribution(o):
  #               output will broadcast over the weights
  dense_contrib = o[dense_l][0] * w[out_l][0].transpose()
  dense_contrib[1,:] = -dense_contrib[1,:] # this column corresponds to second output neuron
                                           # and is evidence for NON promoter
  dense_contrib_sum = dense_contrib.sum(axis=0)
  
  concat_contrib = o[concat_l][0][np.newaxis] * w[dense_l][0].transpose()
  concat_contrib = concat_contrib * dense_contrib_sum[:,np.newaxis]
  concat_contrib_sum = concat_contrib.sum(axis=0)
  # now return separately contributions of neurons that have positive/negative output
  concat_pos_indices = np.argwhere(o[concat_l][0]>0)
  concat_neg_indices = np.argwhere(o[concat_l][0]<0)
  concat_contrib_pos = concat_contrib_sum.copy()
  concat_contrib_pos[concat_neg_indices] = 0.0
  concat_contrib_neg = concat_contrib_sum.copy()
  concat_contrib_neg[concat_pos_indices] = 0.0
  return (concat_contrib_pos, concat_contrib_neg)
  # *******************
  
def plot_important_neurons(neurons, flat_layer=True):
  ''' Plots ?!? '''
  ind = np.argsort(neurons)[::-1][:20]
  if(flat_layer):
    labels = [neuron_label(i) for i in ind]
  else:
    labels = ind
  s = pd.Series(neurons[ind], index=labels).sort_values(ascending=False)
  s.plot.bar() #.iloc[:20]
  plt.subplots_adjust(bottom=0.3)
  plt.show()

def plot_importance_frequency(_p, _n, how_many=20, start=0, fname=None):
  df = pd.DataFrame([_p,_n]).transpose()
  df.columns = ["Positive","Negative"]
  df["sum"] = df["Positive"]+df["Negative"]
  df.sort_values(by="sum", ascending=True, inplace=True)
  df.drop("sum", axis=1, inplace=True)
  #print -(how_many+start), df.shape[0]-start
  df = df[-(how_many+start):df.shape[0]-start]
  #print df.index
  labels = [neuron_label(i) for i in df.index]
  df.index = labels
  fig, ax = plt.subplots(1,1, figsize=(7,8))
  df.plot.barh(stacked=True, ax=ax)
  plt.subplots_adjust(left=0.3)
  plt.xlabel('feature importance frequency')
  ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.subplots_adjust(hspace=0.65, left=0.35, top=0.95, bottom=0.10, right=0.8)
  if fname is None:
    plt.show()
  else:
    plt.savefig(fname)
    plt.close()

def plot_importance_by_layer(_p, _n, how_many=20, start=0, fname=None):
  for i,l in enumerate(model.layers):
    if l.name == "concatenate":
      concat_layer_num = i
  concat_layer = model.layers[concat_layer_num]
  flat_layers = input_layers_of(concat_layer)
  
  fig, axes = plt.subplots(len(flat_layers),1, sharex=True, figsize=(7,len(flat_layers)+1))
  df = pd.DataFrame([_p,_n]).transpose()
  df.columns = ["Positive","Negative"]
  df["sum"] = df["Positive"]+df["Negative"]  
  
  layer_start = 0
  for plot_pos,inp_l in enumerate(flat_layers):
    df2 = df[layer_start:layer_start+inp_l.output_shape[1]].copy()
    layer_start += inp_l.output_shape[1]
    df2.sort_values(by="sum", ascending=True, inplace=True)
    df2.drop("sum", axis=1, inplace=True)
    #print -(how_many+start), df2.shape[0]-start
    df2 = df2[-(how_many+start):df2.shape[0]-start]
    #print df2.index
    labels = [neuron_label(i) for i in df2.index]
    df2.index = labels
    ax = axes[plot_pos]
    df2.plot.barh(stacked=True, ax=ax)
    ax.set_title(inp_l.name)
    plt.subplots_adjust(left=0.3)
    plt.xlabel('feature importance frequency')
    #plt.ylabel('feature')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.subplots_adjust(hspace=0.65, left=0.35, top=0.90, bottom=0.15, right=0.8)
  if fname is None:
    plt.show()
  else:
    plt.savefig(fname)
    plt.close()


def sum_activations_by_layer(act_pos,act_n):
  '''
  app = activation contribution of neuron on training examples, where the '''
  a = pd.DataFrame([act_pos, act_neg])
  a = a.transpose()
  a.columns = ["Activation on promoters", "Activation on non-promoters"]
  inputs = input_layers_of(model.layers[concat_l])
  input_names = [i.name for i in inputs]
  ret = pd.DataFrame(0.0,index = input_names, columns = ["Activation on promoters", "Activation on non-promoters"])
  layer_start = 0
  for i in inputs:
    ret.loc[i.name] = a.loc[layer_start:layer_start+i.output_shape[1]-1].sum()
    layer_start += i.output_shape[1]
    
def plot_activations_summed_by_layer(df, fname=None):
  df.plot.barh()
  plt.legend(loc='lower left', bbox_to_anchor=(0.0, 0.98))
  plt.subplots_adjust(left=0.30, top=0.85, bottom=0.10, right=0.95)
  if fname is None:
    plt.show()
  else:
    plt.savefig(fname)
    plt.close()
  

# ************************************************************************************
# LOAD MODEL
# ************************************************************************************

# load model from file
for model_file in sys.argv[2:]:
  #tf.keras.backend.clear_session(config=config)
  model = tf.keras.models.load_model(model_file)
  n_layers = len(model.layers)
  out_l = len(model.layers)-1 # number of output layer
  dense_l = len(model.layers)-2 # number of dense layer
  concat_l = len(model.layers)-3 # number of concatenate layer
  
  if(model.layers[concat_l].name != "concatenate"):
    print "Layers not recognized! Fix!"
    exit(1)

  
  filter_directory = model_file.replace(".model","")
  # print layer names
  for i,l in enumerate(model.layers):
    print i,l.name
  
  # layer names
  n = [l.name for l in model.layers]
  # weights
  w = [l.get_weights() for l in model.layers]
  
  inp = model.input                                           # input placeholder
  if(type(inp)!=list):
    inp = [inp]
  inp.append(tf.keras.backend.learning_phase())
  outputs = [layer.output for layer in model.layers           # all layer outputs
                             if layer.name not in model.input_names]          
  functor = tf.keras.backend.function(inp, outputs )   # evaluation function
  
  # ************************************************************************************
  # 
  # ************************************************************************************
  
  
  pos_hidden = np.zeros(model.layers[dense_l].output_shape[1])
  neg_hidden = np.zeros(model.layers[dense_l].output_shape[1])
  pos_p_flat = np.zeros(model.layers[concat_l].output_shape[1])
  pos_n_flat = np.zeros(model.layers[concat_l].output_shape[1])
  neg_p_flat = np.zeros(model.layers[concat_l].output_shape[1])
  neg_n_flat = np.zeros(model.layers[concat_l].output_shape[1])
  
  concat_contrib_pos_p = np.zeros(model.layers[concat_l].output_shape[1])
  concat_contrib_pos_n = np.zeros(model.layers[concat_l].output_shape[1])
  concat_contrib_neg_p = np.zeros(model.layers[concat_l].output_shape[1])
  concat_contrib_neg_n = np.zeros(model.layers[concat_l].output_shape[1])
  
  for i in range(len(seq)):
    if i%3000 == 0:
      print i
    
    # collect inputs
    inputs = []
    for inp in model.inputs[:-1]: # <- use if last input layer is 'keras.learning_phase
      obj = eval(inp.name.split("_")[0])
      inputs.append(obj[i][np.newaxis])
    if model.inputs[-1].shape != []:
      obj = eval(model.inputs[-1].name.split("_")[0])
      inputs.append(obj[i][np.newaxis])
    # zero at the end means we want training output
    inputs.append(0)
    
    o = functor(inputs)
    
    # input layers are not producing output
    # to synchronize the indices, insert empty lists as output of input layers
    for il,l in enumerate(model.layers):
      if type(l) == tf.keras.layers.InputLayer:
        o.insert(il,[])
    
    #IPython.embed()
    # ********************************************************************************************
    # IDEA
    # find out which promoters were mis-classified and why 
    #
    # ********************************************************************************************
    
    
    # if the output is supposed to be positive
    if np.all(labels[i]==[1,0]):
      out_pos = [0] # then look into which neurons are responsible for positive outcome
      out_neg = []
    else:
      out_pos = []
      out_neg = [1]
    
    pos_p_hidden_s, pos_n_hidden_s, neg_p_hidden_s, neg_n_hidden_s = \
        important_for_layer(o,out_l,out_pos,out_neg,dense_l, ratio = 0.3)
    pos_hidden_s = pos_p_hidden_s + pos_n_hidden_s
    neg_hidden_s = neg_p_hidden_s + neg_n_hidden_s
    pos_hidden[pos_p_hidden_s] += 1
    pos_hidden[pos_n_hidden_s] += 1
    neg_hidden[neg_p_hidden_s] += 1
    neg_hidden[neg_n_hidden_s] += 1
    pos_p_flat_s, pos_n_flat_s, neg_p_flat_s, neg_n_flat_s = \
        important_for_layer(o,dense_l,pos_hidden_s,neg_hidden_s,concat_l, ratio = 0.3)
    pos_p_flat[pos_p_flat_s] += 1
    pos_n_flat[pos_n_flat_s] += 1
    neg_p_flat[neg_p_flat_s] += 1
    neg_n_flat[neg_n_flat_s] += 1
    
    # if the output is supposed to be positive
    if np.all(labels[i]==[1,0]):
      ac = activation_contribution(o)
      concat_contrib_pos_p += ac[0]
      concat_contrib_pos_n += ac[1]
    else:
      ac = activation_contribution(o)
      concat_contrib_neg_p += ac[0]
      concat_contrib_neg_n += ac[1]
    #IPython.embed()
  
  # normalize by number of inputs
  pos_hidden /= float(len(seq))
  neg_hidden /= float(len(seq))
  pos_p_flat /= float(len(seq))
  pos_n_flat /= float(len(seq))
  neg_p_flat /= float(len(seq))
  neg_n_flat /= float(len(seq))
  
  
  
  
  plot_fname = model_file.replace(".model","")
  
  plot_importance_by_layer(pos_p_flat, pos_n_flat, how_many=5, fname=plot_fname+"ImpByLayer_pos.png")
  plot_importance_by_layer(neg_p_flat, neg_n_flat, how_many=5, fname=plot_fname+"ImpByLayer_neg.png")
  
  plot_importance_frequency(pos_p_flat, pos_n_flat, how_many=50, start=0, fname=plot_fname+"Imp_pos.png")
  plot_importance_frequency(neg_p_flat, neg_n_flat, how_many=50, start=0, fname=plot_fname+"Imp_neg.png")
  
  act_pos = concat_contrib_pos_p + concat_contrib_pos_n
  act_neg = concat_contrib_neg_p + concat_contrib_neg_n
  act_sum_by_latyer = sum_activations_by_layer(act_pos, act_neg)
  plot_activations_summed_by_layer(act_sum_by_latyer, fname = plot_fname+"_ActByLayer.png")
  
# koukni na layer-by-layer prispevky k rozhodnuti NN site
def shelveit():
  shelve_file = model_file.replace(".model",".shelve")
  my_shelf = shelve.open(shelve_file,'n') # 'n' for new
  for key in dir():
    if key != "my_shelf":
      try:
          my_shelf[key] = globals()[key]
      except TypeError:
        print('ERROR shelving: {0}'.format(key))
  my_shelf.close()
  


def plot_filter_weights_over_position(f, tick_interval=50):
  p=[]
  lab=[]
  for pos in range(-500,500,10):
    n = (pos+500)*10 + f
    p.append(w[6][0][n,48])
    print pos, n, w[6][0][n,48]
    lab.append(pos)
  s=pd.Series(p, index=lab)
  ax=s.plot()
  plt.plot([-500,500],[0,0], c="red")
  ax.set_xlim(-500,500)
  plt.xticks(np.arange(-500, 501, tick_interval))
  plt.show()

'''
  belongs = neuron_belongs_to(row.prev,input_desc)
  nwl = row.prev - belongs["start"] # neuron number within the layer
  if belongs["type"]=="nucl_conv" :
    f = nwl % belongs["filters"]
    pos = (nwl / belongs["filters"])*10 -500
    print "Filter:{} pos:{} weight:{}".format(f,pos,w[layer][0][row.prev,row.this])
    plot_filter(w[belongs["layer_num"]][0][:,:,f])
  elif belongs["type"]=="nucl" :
    pass
  else:
    print "Unknown layer type:",belongs["type"]
'''
