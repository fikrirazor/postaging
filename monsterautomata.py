# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:48:18 2019

@author: Rozan
"""
import pandas as pd

def pt(pd1,pd2):
    t= pd.crosstab(pd1,pd2)
    cn=t.columns.values
    w=t[cn].sum()
    p=t/w
    return p