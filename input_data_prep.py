import numpy as np
import pandas as pd
import sys
import IPython

def read_input_files(filename):
  ''' This function implements the loading of the input data, shuffling it and spliting it
      into training and testing set. 
      Params:
        - filename: tab-delimited file containing locations of all input tables in format:
                    <input_type>  <positive example file> <negative example file>
      Example of such file:
        CA	true_promoters/CA_pos.csv	false_promoters_dist350/CA_neg.csv
        CG	true_promoters/CG_pos.csv	false_promoters_dist350/CG_neg.csv
        COV	true_promoters/Coverage_pos.csv	false_promoters_dist350/Coverage_neg.csv
        MAF	true_promoters/MAF_pos.csv	false_promoters_dist350/MAF_neg.csv
        METH	true_promoters/METH_pos.csv	false_promoters_dist350/METH_neg.csv
        seq	true_promoters/promoters_pos.csv	false_promoters_dist350/promoters_neg.csv
        SNPs	true_promoters/SNPs_pos.csv	false_promoters_dist350/SNPs_neg.csv
        TATA	true_promoters/TATA_pos.csv	false_promoters_dist350/TATA_neg.csv
    '''
  input_loc = pd.read_csv(filename, sep="\t", index_col=0, header=None)
  
  system_variables = locals().keys()
  
  test_size = 8000
  TSS_pos=500
  
  # PROMOTOR DNA SEQUENCE
  # read true promoter sequences
  seq_pos = pd.read_csv(input_loc.loc["seq"].iloc[0], header=None, index_col=0)
  # read false promoter sequences
  seq_neg = pd.read_csv(input_loc.loc["seq"].iloc[1], header=None, index_col=0)
  # merge the ndarrays
  seq = np.concatenate((seq_pos.values, seq_neg.values), axis=0)
  # reshape promotor sequences from 2d to 3d (gene,pos,nucleotide)
  seq = seq.reshape(seq.shape[0], seq.shape[1]/4, 4)
  
  # CG-SKEW
  # read true promoter skew
  CG_pos = pd.read_csv(input_loc.loc["CG"].iloc[0], header=None, index_col=0)
  # read false promoter skew
  CG_neg = pd.read_csv(input_loc.loc["CG"].iloc[1], header=None, index_col=0)
  # merge the ndarrays
  CG = np.concatenate((CG_pos.values, CG_neg.values), axis=0)
  # add the channel layer (as axis[2])
  CG = np.expand_dims(CG,2)
  
  # CA-dinucl
  # read true promoter CA mask
  CA_pos = pd.read_csv(input_loc.loc["CA"].iloc[0], header=None, index_col=0)
  # read false promoter CA mask
  CA_neg = pd.read_csv(input_loc.loc["CA"].iloc[1], header=None, index_col=0)
  # merge the ndarrays
  CA = np.concatenate((CA_pos.values, CA_neg.values), axis=0)
  CA = CA[:,TSS_pos-20:TSS_pos+20]
  # add the channel layer (as axis[2])
  CA = np.expand_dims(CA,2)
  
  # COV
  # read true promoter coverage
  COV_pos = pd.read_csv(input_loc.loc["COV"].iloc[0], header=None, index_col=0)
  # read false promoter coverage
  COV_neg = pd.read_csv(input_loc.loc["COV"].iloc[1], header=None, index_col=0)
  # merge the ndarrays
  COV = np.concatenate((COV_pos.values, COV_neg.values), axis=0)
  # add the channel layer (as axis[2])
  COV = np.expand_dims(COV,2)
  
  # SNPs
  # read true promoter SNPs
  SNPs_pos = pd.read_csv(input_loc.loc["SNPs"].iloc[0], header=None, index_col=0)
  # read false promoter SNPs
  SNPs_neg = pd.read_csv(input_loc.loc["SNPs"].iloc[1], header=None, index_col=0)
  # merge the ndarrays
  SNPs = np.concatenate((SNPs_pos.values, SNPs_neg.values), axis=0)
  # add the channel layer (as axis[2])
  SNPs = np.expand_dims(SNPs,2)
  
  # METH
  # read true promoter Methylation
  METH_pos = pd.read_csv(input_loc.loc["METH"].iloc[0], header=None, index_col=0)
  # read false promoter Methylation
  METH_neg = pd.read_csv(input_loc.loc["METH"].iloc[1], header=None, index_col=0)
  # merge the ndarrays
  METH = np.concatenate((METH_pos.values, METH_neg.values), axis=0)
  # add the channel layer (as axis[2])
  METH = np.expand_dims(METH,2)
  
  # LABELS
  # create array with lables
  labels_1 = np.column_stack([np.ones(seq_pos.shape[0]), np.zeros(seq_pos.shape[0])] )
  labels_0 = np.column_stack([np.zeros(seq_neg.shape[0]), np.ones(seq_neg.shape[0])] )
  labels = np.concatenate([labels_1, labels_0])
  
  # SHUFFLE
  shuffle_index = np.random.permutation(np.arange(seq.shape[0]))
  seq = seq[shuffle_index]
  CG = CG[shuffle_index]
  CA = CA[shuffle_index]
  COV = COV[shuffle_index]
  SNPs = SNPs[shuffle_index]
  METH = METH[shuffle_index]
  labels = labels[shuffle_index]
  
  # DEVIDE DATASETS INTO TRAIN AND TEST SETS
  test_seq = seq[:test_size,:]
  test_CG = CG[:test_size,:]
  test_CA = CA[:test_size,:]
  test_COV = COV[:test_size,:]
  test_SNPs = SNPs[:test_size,:]
  test_METH = METH[:test_size,:]
  test_labels = labels[:test_size]
  
  train_seq = seq[test_size:,:]
  train_CG = CG[test_size:,:]
  train_CA = CA[test_size:,:]
  train_COV = COV[test_size:,:]
  train_SNPs = SNPs[test_size:,:]
  train_METH = METH[test_size:,:]
  train_labels = labels[test_size:]
  
  ret = {}
  ret_variables = [i for i in locals().keys() if i not in system_variables]
  for l in ret_variables:
    ret[l] = eval(l)
  return ret
  
