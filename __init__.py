# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 02:57:00 2019

@author: Rozan
"""

import pandas as pd
import numpy as np
import monsterautomata as h

file = r'idn-tagged-corpus-master/Indonesian_Manually_Tagged_Corpus.tsv'
dt =  pd.read_table(file,header=None)

X=dt.iloc[0:1000,0]
Y=dt.iloc[0:1000,1]   

f=np.asarray(Y.iloc[0:999])
c=np.asarray(Y.iloc[1:1000])
peluangawal=h.sp(X,Y)
peluangemisi=h.pt(X,Y) #P(Xt|Yt)
peluangtransisi=h.pt(f,c) #P(Yt|Yt-1)


   
y=np.asarray(peluangawal)
A=np.asarray(peluangtransisi)
B=np.asarray(peluangemisi).T
     
gg=h.viterbi(y,A,B)
