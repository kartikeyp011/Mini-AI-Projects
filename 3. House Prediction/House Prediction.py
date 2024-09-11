# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 22:52:55 2024

@author: knpra
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams["figure.figsize"] = (20,10)

df1 = pd.read_csv("Bengaluru_House_Data.csv")
print(df1.head())

print(df1.shape)

print(df1.groupby('area_type')['area_type'].agg('count'))