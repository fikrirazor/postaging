# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 02:57:00 2019

@author: Rozan
"""

import pandas as pd
import numpy as np
file = r'idn-tagged-corpus-master/Indonesian_Manually_Tagged_Corpus.tsv'
import collections
dt =  pd.read_table(file,header=None)#baca file
X=dt.iloc[0:28146,0] #data 1000 kalimat
Y=dt.iloc[0:28146,1]#data tag dari setiap kata 
test=dt.iloc[28147:28666,0] #datatest 20 kalimat
teskata = test.values.tolist()
Z=[]
P=[]
tab = {}
tabp = {}
tabprob = {}
valueprob = collections.defaultdict(dict) 
valuemisi = collections.defaultdict(dict)  
tabemisi = {} 
C = collections.Counter(Y)
'''training'''
for i in range(len(Y)-1):
    Z.append((Y[i],Y[i+1]))
    if Z[i] in tab:
        tab[Z[i]] += 1
    else:
        tab[Z[i]] = 1
        
for i in tab:
    tag1 = i[0]
    tag2 = i[1]
    tabprob[i] = (tab.get(i))/(C.get(tag1))  
    valueprob[tag1][tag2] = tabprob.get(i)

for i in range(len(X)-1):
    P.append((X[i],Y[i]))
    if P[i] in tabp:
        tabp[P[i]] += 1
    else:
        tabp[P[i]] = 1
                 
for i in tabp:
    tag1 = i[0]
    tag2 = i[1]
    tabemisi[i] = (tabp.get(i))/(C.get(tag2))  
    valuemisi[tag1][tag2] = tabemisi.get(i)    

table=pd.crosstab(X,Y)
cnames=table.columns.values
peluangemisi=table/table[cnames].sum()
peluangdict = peluangemisi.to_dict('index') 
        
''''Baseline'''
peluang = {}
for i in teskata:
        if i in valuemisi:
            peluang[i] = valuemisi.get(i)
        else:
            peluang[i]  = 0
newd = collections.defaultdict(dict)
for k, v in peluang.items():  
     if v != 0:
         print(k, max(v))
     else:
         print(k, v)
#       newd[k] = max(v)       
'''HMM'''        
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

'''
f=np.asarray(Y.iloc[0:28145])
c=np.asarray(Y.iloc[1:28146])
obs=(X,Y)
states=(f,c)
peluangawal=sp(X,Y)
peluangemisi=pt(X,Y) #P(Xt|Yt)
peluangtransisi=pt(f,c) #P(Yt|Yt-1)
y=np.asarray(peluangawal)#np.asarray(peluangawal/np.sum(peluangawal))
A=np.asarray(peluangtransisi)
B=np.asarray(peluangemisi).T

abc=viterbi(y,A,B)
'''
