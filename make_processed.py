#HAI -> Numpy processed File
import sys

from pathlib import Path
from datetime import timedelta

import dateutil
import numpy as np
import pandas as pd

from reduce_utils import *

TRAIN_DATASET = sorted([x for x in Path('./data/training/').glob("*.csv")])
VAL_DATASET = sorted([x for x in Path('./data/validation/').glob("*.csv")])
#TEST_DATASET = sorted([x for x in Path('hai1/data/test-dataset/').glob("*.csv")])

def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())


def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])

TRAIN_DF_RAW = dataframe_from_csvs(TRAIN_DATASET)
VAL_DF_RAW = dataframe_from_csvs(VAL_DATASET)

print('Train raw',TRAIN_DF_RAW.shape)
print('VAL raw',VAL_DF_RAW.shape)

TIMESTAMP_FIELD = "time"
ATTACK_FIELD = "attack"

VALID_COLUMNS_IN_TRAIN_DATASET = TRAIN_DF_RAW.columns.drop([TIMESTAMP_FIELD])


#MIN, MAX values taken from training df
TAG_MIN = TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET].min()
TAG_MAX = TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET].max()


def normalize(df):
    ndf = df.copy()
    for c in df.columns:
        #no dif in value -> make 0
        if TAG_MIN[c] == TAG_MAX[c]:
            ndf[c] = df[c] - TAG_MIN[c]
        #Normalize other values
        else:
            ndf[c] = (df[c] - TAG_MIN[c]) / (TAG_MAX[c] - TAG_MIN[c])
    return ndf

#ewm: exponential weighted function - noise smoothing
TRAIN_DF = normalize(TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=0.9).mean()
VAL_DF = normalize(VAL_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET])

def boundary_check(df):
    x = np.array(df, dtype=np.float32)
    return np.any(x > 1.0), np.any(x < 0), np.any(np.isnan(x))

train_tag_vals=np.array(TRAIN_DF, dtype=np.float32)
val_tag_vals=np.array(VAL_DF, dtype=np.float32)

np.save('./processed/train.npy', train_tag_vals)
np.save('./processed/val.npy', val_tag_vals)

val_atk=np.array(VAL_DF_RAW[ATTACK_FIELD])
np.save('./processed/val_atk.npy',val_atk)
