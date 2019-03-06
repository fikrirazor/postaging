# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 02:57:00 2019

@author: Rozan
"""

import pandas as pd
from collections import Counter
import numpy as np
file = r'idn-tagged-corpus-master/Indonesian_Manually_Tagged_Corpus.tsv'
dt =  pd.read_table(file,header=None)

X=dt.iloc[:,0]
Y=dt.iloc[:,1]
Z=[]
for i in range(len(Y)-1):
    Z.append((Y[i],Y[i+1]))

#tes=dt.apply(pd.value_counts)
#hk=X.value_counts()
#ht=Y.value_counts()
#hk.plot('barh')
#ht..plot('barh')
table=pd.crosstab(X,Y)
cnames=table.columns.values
peluangemisi=table/table[cnames].sum()

a=np.asarray(Z)
elems=np.unique(Z)
dim=len(elems)
P=np.zeros((dim,dim))

for j, x_in in enumerate(elems):
    for k, x_out in enumerate(elems):
        P[j,k] = (a == [x_in, x_out]).all(axis=1).sum()
        
        if P[j,:].sum() > 0:
            P[j,:] /= P[j,:].sum()