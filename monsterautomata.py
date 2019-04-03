# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:48:18 2019

@author: Rozan
"""
import pandas as pd
import numpy as np
def getobs(pd1,pd2):
    t=pd.crosstab(pd1,pd2)
    cn=pd1.values
    return cn

def getstates(pd1,pd2):
    t=pd.crosstab(pd1,pd2)
    cn=t.columns.values
    return cn

def sp(pd1,pd2):
    t=pd.crosstab(pd1,pd2)
    cn=t.columns.values
    w=t[cn].sum()
    return w
    

def pt(pd1,pd2):
    t= pd.crosstab(pd1,pd2)
    cn=t.columns.values
    w=t[cn].sum()
    p=t/w
    return p

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
