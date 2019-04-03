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
obs=h.getobs(X,Y)
states=h.getstates(f,c)
peluangawal=h.sp(X,Y)
peluangemisi=h.pt(X,Y).T #P(Yt|Xt)
peluangtransisi=h.pt(f,c) #P(Yt|Yt-1)


   
y=np.asarray(peluangawal)#np.asarray(peluangawal)#np.asarray(peluangawal/np.sum(peluangawal))
A=np.asarray(peluangtransisi)
B=np.asarray(peluangemisi)
     



X_test=dt.iloc[1000:10100,0]

path, delta, phi = h.viterbi(y,A,B)
print('\nsingle best state path: \n', path)
print('delta:\n', delta)
print('phi:\n', phi)

