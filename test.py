import os
import sys
import collections
from datetime import datetime
import random
import itertools
import functools
import operator
import re
import math
import cmath
import fractions
import enum
import pickle
import threading
import multiprocessing
import subprocess
import queue
import asyncio

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import preprocessing

human = preprocessing.get_human_dataframe("train")

target = "kp1_x_ratio"
features = ["kp{num}_{d}_ratio".format(num=num, d=d) for num in range(1, 15) for d in ["x", "y"]] + \
           ["kp{num}_status".format(num=num) for num in range(1, 15)] + ['l/w', 'length', 'width']
features.remove(target)
dtrain = xgb.DMatrix(human.loc[:100, features], label=human.loc[:100, target], feature_names=features)
params = {
    "max_depth": 3, 
    "learning_rate": 0.1, 
    "objective": 'reg:linear', 
    "nthread": -1, 
    "gamma": 0, 
    "min_child_weight": 1, 
    "max_delta_step": 0, 
    "subsample": 1, 
    "colsample_bytree": 1, 
    "colsample_bylevel": 1, 
    "reg_alpha": 0, 
    "reg_lambda": 1, 
    "scale_pos_weight": 1, 
    "base_score": 0.5, 
    "seed": 2017
}

xgb.cv(params=params, dtrain=dtrain, num_boost_round=10, nfold=3, metrics=["mse"], verbose_eval=True)