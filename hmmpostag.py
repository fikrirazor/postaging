# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 02:57:00 2019

@author: Rozan
"""

import pandas as pd

file = r'idn-tagged-corpus-master/Indonesian_Manually_Tagged_Corpus.tsv'
dt =  pd.read_table(file,header=None)

#tes=dt.apply(pd.value_counts)
hk=dt.iloc[:,0].value_counts()
ht=dt.iloc[:,1].value_counts()
#hk.plot('barh')
#ht..plot('barh')
