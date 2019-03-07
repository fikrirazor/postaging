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
def viterbi(y, A, B, Pi=None):
    # Cardinality of the state space
    K = A.shape[0]
    # Initialize the priors with default (uniform dist) if not given by caller
    Pi = Pi if Pi is not None else np.full(K, 1 / K)
    T = len(y)
    T1 = np.empty((K, T), 'd')
    T2 = np.empty((K, T), 'B')

    # Initilaize the tracking tables from first observation
    T1[:, 0] = Pi * B[:, y[0]]
    T2[:, 0] = 0

    # Iterate throught the observations updating the tracking tables
    for i in range(1, T):
        T1[:, i] = np.max(T1[:, i - 1] * A.T * B[np.newaxis, :, y[i]].T, 1)
        T2[:, i] = np.argmax(T1[:, i - 1] * A.T, 1)

    # Build the output, optimal model trajectory
    x = np.empty(T, 'B')
    x[-1] = np.argmax(T1[:, T - 1])
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]

    return x, T1, T2

     
gg=viterbi(y,A,B)
