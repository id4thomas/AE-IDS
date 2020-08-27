import pandas as pd
import numpy as np

from data_utils import *

#Labels
SAFE=1
ATK=0

#Load Processed Data

#Train Data
train_d,train_l=load_processed('train')

#Sample 1:1 (Safe:Atk)
safe_idx=[train_l==SAFE]
atk_idx=[train_l==ATK]

#Split Data
train_d_s=train_d[safe_idx]
train_l_s=train_l[safe_idx]

train_d_a=train_d[atk_idx]
train_l_a=train_l[atk_idx]

#Sample Index
sample_idx=np.random.choice(train_d_a.shape[0], int(train_d_s.shape[0]))

train_d_a_sampled=train_d_a[sample_idx]
train_l_a_sampled=train_l_a[sample_idx]

train_d=np.concatenate((train_d_s,train_d_a_sampled),axis=0)
train_l=np.concatenate((train_l_s,train_l_a_sampled),axis=0)
print(train_d.shape)

import os
if not os.path.exists('./data/data_configs'):
    os.makedirs('./data/data_configs')
#Save to numpy File
np.save('./data/data_configs/train_data_1_1.npy',train_d)
np.save('./data/data_configs/train_label_1_1.npy',train_l)
