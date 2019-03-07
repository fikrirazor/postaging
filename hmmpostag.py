# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 02:57:00 2019

@author: Rozan
"""

import pandas as pd
import numpy as np
file = r'idn-tagged-corpus-master/Indonesian_Manually_Tagged_Corpus.tsv'
dt =  pd.read_table(file,header=None)

X=dt.iloc[0:1000,0]
Y=dt.iloc[0:1000,1]   
c=np.asarray(Y.iloc[1:1000])
f=np.asarray(Y.iloc[0:999])

def pt(pd1,pd2):
    t= pd.crosstab(pd1,pd2)
    cn=t.columns.values
    w=t[cn].sum()
    p=t/w
    return p

peluangemisi=pt(X,Y)
peluangtransisi=pt(f,c)
