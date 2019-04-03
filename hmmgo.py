# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:26:25 2019

@author: Rozan
"""

from math import log 
def baseline(teskata,valuemisi):
   peluang = {}
   for i in teskata:
        if i in valuemisi:
            peluang[i] = valuemisi.get(i)
        else:
            peluang[i]  = 0
   baseline=[]
   for k, v in peluang.items():  
         if v != 0:
             baseline.append((k, max(v)))
         else:
             baseline.append((k, v))
    
   return baseline

def viterbi(prior,transition,num,trans,StateProbs,tokentag,teskata,D,total):
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