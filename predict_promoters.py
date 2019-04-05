import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import IPython
import matplotlib.pyplot as plt
import time

from misc import *

allow_growth_for_default_session()

# PARAMETERS
test_size = 8000
TSS_pos=500

# ************************************************************************************
# INPUT DATA PREPARATION
# ************************************************************************************
from input_data_prep import *
# ************************************************************************************
# BUILD MODEL
# ************************************************************************************

def buid_model_and_train(use_inputs,
                         data,
                         epochs=10,
                         l1=0.0,
                         l2=0.01,
                         dense=[256],
                         optimizer="Adam",
                         learning_rate=0.001,
                         model=None,
                         distance=5,
                         activation="relu"):
  '''
  This function builds tensorflow.keras model and trains it on given input data.
  
  Params:
   - use_inputs: list of input types to be used for TSS detection. Chsoose from:
     "seq":       use DNA sequence (including one 1D convolution)
     "seq-2conv": also use second 1D convolution on the DNA sequence
     "CG":        use CG-skew data via 1D convolution layer
     "CG-avg":    use CG-skew data via simlpe average pooling (no convolution)
     "CA":        use presence of CA di-nucleotide (directm only in <-20;20> interval)
     "COV":       use Rna-seq coverage data
     "METH":      use methylation data (average pooling)
     "SNPs":      use SNP data (average pooling)
   
   - data: dictionary containing input data in form of pandas.Dataframe(s)
   - hidden: list of integers. Length of list = number of hidden dense layers. Each number
             in the list defines size of a hidden layer
   - epochs: number of epochs we want to run (in model.fit())
   - l1: l1 regularization coefficient
   - l2: l2 regularization coefficient
   - optimizer: optimizer type to use. Choose one of: ["Adam", "SGD", "RMSprop", "Adagrad", "Adadelta", "Adamax", "Nadam"]
                description of optimizers here: https://keras.io/optimizers/
   - activation: activation function for dense layers.
  '''

  run_name = "-".join(use_inputs)+"_"+optimizer+"_d_"+"-".join(map(str,dense))+"_l1-{}_l2-{}_{}".format(l1,l2,activation)
  
  callbacks = [
    # Interrupt training if `val_loss` stops improving for over 2 epochs
    #tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
    # saving the model
    tf.keras.callbacks.ModelCheckpoint(save_best_only=True,
      filepath="./models_dist{}/{}".format(distance,run_name)+"-ep{epoch:02d}-acc{val_acc:.2f}_conv.model"),
    # Write TensorBoard logs to `./logs` directory
    tf.keras.callbacks.TensorBoard(log_dir='./logs_dist{}/{}'.format(distance,run_name))
  ]
  
  inputs = []
  to_concat = []
  train_data = {}
  test_data = {}
  if "seq" in use_inputs:
    seq_input       = tf.keras.Input(shape=(1000,4), name="seq_input")
    seq_conv        = tf.keras.layers.Conv1D(100, 5, 1, name="seq_conv")(seq_input)
    seq_maxpool     = tf.keras.layers.MaxPool1D(10, name="seq_maxpool")(seq_conv)
    #seq_2conv_maxpool= tf.keras.layers.MaxPool1D(10, name="seq_2conv_maxpool")(seq_conv_conv)
    seq_flatten     = tf.keras.layers.Flatten(name="seq_conv-flatten")(seq_maxpool)
    to_concat.append(seq_flatten)
    inputs.append(seq_input)
    # now let's add raw nucleotides without convolution
    seq_flatten2    = tf.keras.layers.Flatten(name="seq_direct-flatten")(seq_input)
    to_concat.append(seq_flatten2)
    train_data["seq_input"] = data["train_seq"]
    test_data["seq_input"] = data["test_seq"]
    # and finally output of 2conv layer
    if "seq-2conv" in use_inputs:
      seq_2conv       = tf.keras.layers.Conv1D(10, 5, 1, name="seq_2conv")(seq_maxpool)
      seq_flatten3    = tf.keras.layers.Flatten(name="seq_2conv_flatten")(seq_2conv)
      to_concat.append(seq_flatten3)

  
  if "CG" in use_inputs:
    CG_input         = tf.keras.Input(shape=(1000,1), name="CG_input")
    #CG_avgpool       = tf.keras.layers.AveragePooling1D(20)(CG_input)
    CG_conv        = tf.keras.layers.Conv1D(100, 5, 1, name="CG_conv")(CG_input)
    CG_maxpool     = tf.keras.layers.MaxPool1D(5, name="CG_maxpool")(CG_conv)
    CG_flatten     = tf.keras.layers.Flatten(name="CG_flatten")(CG_maxpool)
    to_concat.append(CG_flatten)
    inputs.append(CG_input)
    train_data["CG_input"] = data["train_CG"]
    test_data["CG_input"] = data["test_CG"]
    
  if "CG-avg" in use_inputs:
    CG_input         = tf.keras.Input(shape=(1000,1), name="CG_input")
    CG_avgpool       = tf.keras.layers.AveragePooling1D(20, name="CG_avgpool")(CG_input)
    CG_flatten     = tf.keras.layers.Flatten(name="CG-avg_flatten")(CG_avgpool)
    to_concat.append(CG_flatten)
    inputs.append(CG_input)
    train_data["CG_input"] = data["train_CG"]
    test_data["CG_input"] = data["test_CG"]
  
  if "CA" in use_inputs:
    CA_input         = tf.keras.Input(shape=(40,1), name="CA_input")
    CA_flatten     = tf.keras.layers.Flatten(name="CA_flatten")(CA_input)
    to_concat.append(CA_flatten)
    inputs.append(CA_input)
    train_data["CA_input"] = data["train_CA"]
    test_data["CA_input"] = data["test_CA"]
  
  if "COV" in use_inputs:
    COV_input         = tf.keras.Input(shape=(1000,1), name="COV_input")
    COV_avgpool       = tf.keras.layers.AveragePooling1D(20, name="COV_avgpool")(COV_input)
    COV_flatten     = tf.keras.layers.Flatten(name="COV_flatten")(COV_avgpool)
    to_concat.append(COV_flatten)
    inputs.append(COV_input)
    train_data["COV_input"] = data["train_COV"]
    test_data["COV_input"] = data["test_COV"]
  
  if "METH" in use_inputs:
    METH_input         = tf.keras.Input(shape=(1000,1), name="METH_input")
    METH_avgpool       = tf.keras.layers.AveragePooling1D(20, name="METH_avgpool")(METH_input)
    METH_flatten     = tf.keras.layers.Flatten(name="METH_flatten")(METH_avgpool)
    to_concat.append(METH_flatten)
    inputs.append(METH_input)
    train_data["METH_input"] = data["train_METH"]
    test_data["METH_input"] = data["test_METH"]
  
  if "SNPs" in use_inputs:
    SNPs_input         = tf.keras.Input(shape=(1000,1), name="SNPs_input")
    SNPs_avgpool       = tf.keras.layers.AveragePooling1D(20, name="SNPs_avgpool")(SNPs_input)
    SNPs_flatten     = tf.keras.layers.Flatten(name="SNPs_flatten")(SNPs_avgpool)
    to_concat.append(SNPs_flatten)
    inputs.append(SNPs_input)
    train_data["SNPs_input"] = data["train_SNPs"]
    test_data["SNPs_input"] = data["test_SNPs"]
  
  concat     = tf.keras.layers.Concatenate()(to_concat)
  # define regularizer
  if(l1==0)and(l2==0):
    regularizer = None
  elif(l1==0):
    regularizer = tf.keras.regularizers.l2(l2)
  elif(l2==0):
    regularizer = tf.keras.regularizers.l1(l1)
  else:
    regularizer = tf.keras.regularizers.l1_l2(l1=l1, l2=l2)
  
  dense_l= [tf.keras.layers.Dense(dense[0], activation=activation,
                                  kernel_regularizer=regularizer)(concat)]
  for i in range(1,len(dense)):
    dense_l.append(tf.keras.layers.Dense(dense[i], activation=activation,
                                         kernel_regularizer=regularizer)(dense_l[i-1]))
  prediction = tf.keras.layers.Dense(2, activation='softmax')(dense_l[-1])
  # if model is provided, keep training that model
  if model is None:
    model = tf.keras.Model(inputs=inputs, outputs=prediction)
  
  if(optimizer=="Adam"):
    op = tf.keras.optimizers.Adam(learning_rate)
  elif(optimizer=="SGD"):
    op = tf.keras.optimizers.SGD(learning_rate)
  elif(optimizer=="Adagrad"):
    op = tf.keras.optimizers.Adagrad(learning_rate)
  elif(optimizer=="RMSprop"):
    op = tf.keras.optimizers.RMSprop(learning_rate)
  elif(optimizer=="Nadam"):
    op = tf.keras.optimizers.Nadam(learning_rate)
  else:
    print "Unknown optimizer"
    return

  model.compile(op,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  model.summary()
  tf.keras.utils.plot_model(model, show_shapes=True)
  model.fit(train_data,
            data["train_labels"],
            epochs=epochs,
            batch_size=60,
            callbacks=callbacks,
            validation_data=(
              test_data,
              data["test_labels"])
           )
  return model


def main():
  use = ["seq", "seq-2conv", "CG-avg", "COV", "METH", "CA", "SNPs"]
  l1 = 0.02
  l2 = 0.00
  optimizer = "Adam"
  dense = [50]
  distance=350
  inp = read_input_files(sys.argv[1])
  m = buid_model_and_train(use,
                           inp,
                           epochs=30,
                           l1=l1,
                           l2=l2,
                           dense=d,
                           optimizer=optimizer,
                           distance=distance,
                           activation="relu")


if __name__=="__main__":
  main()
