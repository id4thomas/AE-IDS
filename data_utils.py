import os
import h5py

import pandas as pd
import numpy as np

#For kyoto data
def load_data_config(type,config):
    #print(df.head)
    data=np.load('./data/data_configs/'+type+'_data_'+config+'.npy')
    label=np.load('./data/data_configs/'+type+'_label_'+config+'.npy')
    #print(df.head)
    return data,label

def load_processed(type):
    train_file='./data/processed/'+type+'.csv'
    df=pd.read_csv(train_file,sep="\t", header = None)
    #print(df.head)
    data=df.iloc[:,range(0,119)].to_numpy()
    #idx3: atk label
    label=np.load('./data/processed/'+type+'_label.npy')[:,3]
    #print(df.head)
    return data,label

def get_hdf5_data(dirpath):
    hdf5_files = os.listdir(dirpath)
    print(dirpath)
    data = []
    label=[]
    for hdf5_file in hdf5_files:
        with h5py.File(dirpath+'/'+hdf5_file,'r') as f:
            data.append(f['data'].value)
            label.append(f['label'].value)
            l=f['label'].value
        print(hdf5_file)
    return np.concatenate(data),np.concatenate(label).flatten()
