# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 02:57:00 2019

@author: Rozan
"""

import pandas as pd

file = r'idn-tagged-corpus-master/Indonesian_Manually_Tagged_Corpus.tsv'
data =  pd.read_table(file,header=None)