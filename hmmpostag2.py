# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 02:57:00 2019

@author: Rozan
"""
import pandas as pd
from collections import Counter
file = r'Indonesian_Manually_Tagged_Corpus_ID.csv'
import collections
from math import log  
dt = pd.read_csv(file, encoding='latin-1')#baca file
X=dt.iloc[0:27048,0] #data 1000 kalimat
Y=dt.iloc[0:27048,1]#data tag dari setiap kata 
test=dt.iloc[27049:27547,0] #datatest 20 kalimat
teskata = test.values.tolist()#datatest 20 kalimat dalam bentuk list
Z=[]#list antar tag
P=[]#list kata dengan tag
tab = {}#jumlah list antar tag
tabp = {}#jumlahc
tabprob = {}#peluang antar tag
tabemisi = {} #peluang list kata dengan tag
C = collections.Counter(Y)#hitung kemunculan tag
D = collections.Counter(X)#hitung kemunculan kata
valueprob = collections.defaultdict(dict) #peluang peluang antar tag dalam bentuk dict
valuemisi = collections.defaultdict(dict) #peluang list kata dengan tag dalam bentuk dict 

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
        
''''Baseline'''
peluang = {}
for i in teskata:
        if i in valuemisi:
            peluang[i] = valuemisi.get(i)
        else:
            peluang[i]  = 0
            
for k, v in peluang.items():  
     if v != 0:
         print(k, max(v))
     else:
         print(k, v)
           
'''HMM'''

tagtag = collections.defaultdict(Counter) #hitung kemunculan antar tag
tokentag = collections.defaultdict(Counter)#hitung kemunculan antar kata dan tag
total = len(P)#total pasangan tag dan kata

for i in C.keys():
    C[i] = C[i]/total
for i in tab:
    tag1 = i[0]
    tag2 = i[1]
    tagtag[tag1][tag2] = tab.get(i) 

for i in tabp:
    tag1 = i[0]
    tag2 = i[1]
    tokentag[tag1][tag2] = tabp.get(i) 
    
trans = {} #peluang antarstate
StateProbs = {}#peluang state

def viterbi(prior,transition,num):
    trans[num] = []
    StateProbs[num+1] = []
    emmision = tokentag[teskata[num]]
    wn = D[teskata[num]]
    p = {}
    for ik,ii in emmision.items():
        #hold probs
        min = 100000
        for jk,ji in prior.items():
            if transition[jk][ik] != 0:
                if num==0:
                    prob = log((ii/wn),2)  + (log((transition[jk][ik]/(total-1)), 2))
                else:
                    prob = ji + log((ii/wn), 2) + (log((transition[jk][ik]/(total-1)), 2))
                trans[num].append([(jk,ik),prob])
                if min > prob:
                    min = prob
        p[ik] = min
        StateProbs[num+1].append([ik,min])
    return p

#tes kalimat
prev = viterbi(C,tagtag,0)#statesaat ini
for i in range(1,len(teskata)):
    prev = viterbi(prev,tagtag,i)
   
del trans[0]

prevP = 0 #statesebelum
prev = ''
order = [] #urutan state    
kat = [] #urutan kata

#backpropogation
for i in range(len(teskata)-1,-1,-1):
    if i == len(teskata)-1:
        min = 100000000
        for j in StateProbs[i+1]:
            if min > j[1]:
                prev = j[0]
                min = j[1]
                prevP = j[1]
        order.append(prev)
        kat.append(teskata[i])
    else:
        for g in trans[i+1]:
            if prevP == g[1]:
                x,y = g[0]
                prev = x
                order.append(prev)
                kat.append(teskata[i])
        for k in StateProbs[i+1]:
            if k[0] == prev:
                prevP = k[1]

#solution
kata = [] #urutan kata berurutan               
sol = [] #urutan tag berurutan
for i in reversed(kat):
    kata.append(i)
for i in reversed(order):
    sol.append(i)

hmm = list(zip(kata,sol))
print(hmm)
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 02:57:00 2019

@author: Rozan
"""
import pandas as pd
from collections import Counter
file = r'Indonesian_Manually_Tagged_Corpus_ID.csv'
import collections
from math import log  
dt = pd.read_csv(file, encoding='latin-1')#baca file
X=dt.iloc[0:27048,0] #data 1000 kalimat
Y=dt.iloc[0:27048,1]#data tag dari setiap kata 
test=dt.iloc[27049:27547,0] #datatest 20 kalimat
teskata = test.values.tolist()#datatest 20 kalimat dalam bentuk list
Z=[]#list antar tag
P=[]#list kata dengan tag
tab = {}#jumlah list antar tag
tabp = {}#jumlahc
tabprob = {}#peluang antar tag
tabemisi = {} #peluang list kata dengan tag
C = collections.Counter(Y)#hitung kemunculan tag
D = collections.Counter(X)#hitung kemunculan kata
valueprob = collections.defaultdict(dict) #peluang peluang antar tag dalam bentuk dict
valuemisi = collections.defaultdict(dict) #peluang list kata dengan tag dalam bentuk dict 

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
        
''''Baseline'''
peluang = {}
for i in teskata:
        if i in valuemisi:
            peluang[i] = valuemisi.get(i)
        else:
            peluang[i]  = 0
            
for k, v in peluang.items():  
     if v != 0:
         print(k, max(v))
     else:
         print(k, v)
           
'''HMM'''

tagtag = collections.defaultdict(Counter) #hitung kemunculan antar tag
tokentag = collections.defaultdict(Counter)#hitung kemunculan antar kata dan tag
total = len(P)#total pasangan tag dan kata

for i in C.keys():
    C[i] = C[i]/total
for i in tab:
    tag1 = i[0]
    tag2 = i[1]
    tagtag[tag1][tag2] = tab.get(i) 

for i in tabp:
    tag1 = i[0]
    tag2 = i[1]
    tokentag[tag1][tag2] = tabp.get(i) 
    
trans = {} #peluang antarstate
StateProbs = {}#peluang state

def viterbi(prior,transition,num):
    trans[num] = []
    StateProbs[num+1] = []
    emmision = tokentag[teskata[num]]
    wn = D[teskata[num]]
    p = {}
    for ik,ii in emmision.items():
        #hold probs
        min = 100000
        for jk,ji in prior.items():
            if transition[jk][ik] != 0:
                if num==0:
                    prob = log((ii/wn),2)  + (log((transition[jk][ik]/(total-1)), 2))
                else:
                    prob = ji + log((ii/wn), 2) + (log((transition[jk][ik]/(total-1)), 2))
                trans[num].append([(jk,ik),prob])
                if min > prob:
                    min = prob
        p[ik] = min
        StateProbs[num+1].append([ik,min])
    return p

#tes kalimat
prev = viterbi(C,tagtag,0)#statesaat ini
for i in range(1,len(teskata)):
    prev = viterbi(prev,tagtag,i)
   
del trans[0]

prevP = 0 #statesebelum
prev = ''
order = [] #urutan state    
kat = [] #urutan kata

#backpropogation
for i in range(len(teskata)-1,-1,-1):
    if i == len(teskata)-1:
        min = 100000000
        for j in StateProbs[i+1]:
            if min > j[1]:
                prev = j[0]
                min = j[1]
                prevP = j[1]
        order.append(prev)
        kat.append(teskata[i])
    else:
        for g in trans[i+1]:
            if prevP == g[1]:
                x,y = g[0]
                prev = x
                order.append(prev)
                kat.append(teskata[i])
        for k in StateProbs[i+1]:
            if k[0] == prev:
                prevP = k[1]

#solution
kata = [] #urutan kata berurutan               
sol = [] #urutan tag berurutan
for i in reversed(kat):
    kata.append(i)
for i in reversed(order):
    sol.append(i)

hmm = list(zip(kata,sol))
print(hmm)
