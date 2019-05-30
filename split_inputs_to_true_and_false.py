import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

##
# @file
# This file contains code to automaticaly extract examples of "Promoters" and "Non-promoters"
# Promoters will be extracted from all available datasets as region <-500; 500)
# while Non-promoters will be 1000nt long shifted off center to one side by at least *min_shift*
# nucleotides.
#
# args:
#  All files are expected to have shape (N, 2000) TSS is expected at [1000]
#  Files should not have any header, first columns is gene name
# [1] = promoters in numeric csv format (A=1,0,0,0; C=0,1,0,0; G=0,0,1,0; T=0,0,0,1)
# [2] = CG skew   
# [3] = CA        
# [4] = TATA      
# [5] = Sum_of_coverage_after_position - Sum_of_coverage_before_position
# [6] = SNPs
# [7] = Minor Allele Frequency
# [8] = methylation
# [9] = Coverage

neg_prom_shift_interval = (-500,+400)
min_shift = 5

# read inputs
prom = pd.read_csv(sys.argv[1], header=None, index_col=0, sep=",")
CG   = pd.read_csv(sys.argv[2], header=None, index_col=0, sep=",")
CA   = pd.read_csv(sys.argv[3], header=None, index_col=0, sep=",")
TATA = pd.read_csv(sys.argv[4], header=None, index_col=0, sep=",")
coverage = pd.read_csv(sys.argv[5], header=None, index_col=0, sep=",")
SNPs = pd.read_csv(sys.argv[6], header=None, index_col=0, sep=",")
MAF  = pd.read_csv(sys.argv[7], header=None, index_col=0, sep=",")
METH = pd.read_csv(sys.argv[8], header=None, index_col=0, sep=",")
newcov = pd.read_csv(sys.argv[1], header=None, index_col=0, sep=",")


# write promoter files
prom.iloc[:,500*4:1500*4].to_csv("promoters_pos.csv", sep=",", header=False)
CG.iloc[:,500:1500].to_csv("CG_pos.csv", sep=",", header=False)
CA.iloc[:,500:1500].to_csv("CA_pos.csv", sep=",", header=False)
CG.iloc[:,500:1500].to_csv("TATA_pos.csv", sep=",", header=False)
coverage.iloc[:,500:1500].to_csv("Coverage_pos.csv", sep=",", header=False)
SNPs.iloc[:,500:1500].to_csv("SNPs_pos.csv", sep=",", header=False)
MAF.iloc[:,500:1500].to_csv("MAF_pos.csv", sep=",", header=False)
METH.iloc[:,500:1500].to_csv("METH_pos.csv", sep=",", header=False)
newcov.iloc[:,500:1500].to_csv("mycov_pos.csv", sep=",", header=False)

with open("promoters_neg.csv","w") as prom_f, \
     open("CG_neg.csv","w") as CG_f, \
     open("CA_neg.csv","w") as CA_f, \
     open("TATA_neg.csv","w") as TATA_f, \
     open("Coverage_neg.csv","w") as coverage_f, \
     open("SNPs_neg.csv","w") as SNPs_f, \
     open("MAF_neg.csv","w") as MAF_f, \
     open("METH_neg.csv","w") as METH_f \
     open("mycov_neg.csv","w") as newcov_f:
  for i,n in enumerate(prom.index):
    offset = 0
    while abs(offset) < min_shift:
      offset = np.random.randint(neg_prom_shift_interval[0], neg_prom_shift_interval[1])
    prom_f.write(n+","+",".join(map(str,prom.iloc[i, (500+offset)*4:(1500+offset)*4]))+"\n")
    CG_f.write(n+","+",".join(map(str,CG.iloc[i, 500+offset:1500+offset]))+"\n")
    CA_f.write(n+","+",".join(map(str,CA.iloc[i, 500+offset:1500+offset]))+"\n")
    TATA_f.write(n+","+",".join(map(str,TATA.iloc[i, 500+offset:1500+offset]))+"\n")
    coverage_f.write(n+","+",".join(map(str,coverage.iloc[i, 500+offset:1500+offset]))+"\n")
    SNPs_f.write(n+","+",".join(map(str,SNPs.iloc[i, 500+offset:1500+offset]))+"\n")
    MAF_f.write(n+","+",".join(map(str,MAF.iloc[i, 500+offset:1500+offset]))+"\n")
    METH_f.write(n+","+",".join(map(str,METH.iloc[i, 500+offset:1500+offset]))+"\n")
    newcov_f.write(n+","+",".join(map(str,newcov.iloc[i, 500+offset:1500+offset]))+"\n")
