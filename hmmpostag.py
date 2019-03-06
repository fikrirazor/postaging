# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 02:57:00 2019

@author: Rozan
"""

import pandas as pd
from collections import Counter
file = r'idn-tagged-corpus-master/Indonesian_Manually_Tagged_Corpus.tsv'
dt =  pd.read_table(file,header=None)

X=dt.iloc[:,0]
Y=dt.iloc[:,1]
#tes=dt.apply(pd.value_counts)
#hk=X.value_counts()
#ht=Y.value_counts()
#hk.plot('barh')
#ht..plot('barh')
table=pd.crosstab(X,Y)
cnames=table.columns.values
peluangemisi=table/table[cnames].sum()

