import pandas as pd
import numpy as np

from data_utils import *
from reduce_utils import *
from plot_utils import *

import argparse
#Find Best Number of Components

def find_best(data,label,reduc_type='pca',kernel='rbf',dis='l1'):
    #Search Range 2~30
    n_c_range=range(2,31)
    best_auc=0
    best_i=-1
    aucs=[]

    for i in n_c_range:
        print("Number of Components",i)
        reduced,reduc=train_reduc(data,reduc_type=reduc_type,kernel=kernel,n_c=i)

        _,auc,_=test_reduc(data,label,reduc,reduc_type,dis=dis)
        if auc>best_auc:
            best_auc=auc
            best_i=i
        aucs.append(auc)
    print("Best n_c {} auc {}".format(best_i,best_auc))
    fig=plt.figure()
    pltauc=fig.add_subplot(1,1,1)
    pltauc.plot(n_c_range,aucs)
    pltauc.set_title('AUC Score Plot')
    fig.savefig('./plot/aucs_{}_{}.png'.format(reduc_type,dis))
    return best_i,best_auc

#Parse Arguments
parser = argparse.ArgumentParser(description='Reduction Restoration Performance')
parser.add_argument('--r', type=str, default='pca',
                    help='Reduction Algorithm')
#L1: Manhattan, L2: MSE
parser.add_argument('--dis', type=str, default='l1',
                    help='Reconstruction Distance')
args = parser.parse_args()

reduc_type=args.r
dis=args.dis

#Set Params
WINDOW_SIZE = args.win
#Load Data
#HDF5 Data
# x_train,y_train=get_hdf5_data('../kyoto_data/hdf5/train')
x_test,y_test=get_hdf5_data('./data/hdf5/test')
y_test=y_test.flatten()

#Processed Data
train_cfg='1_1'
x_train,y_train=load_data_config('train',train_cfg)


#Find Best Performance for Train Data
best_nc,best_auc=find_best(x_train,y_train,reduc_type,dis=dis)


#Check Test Data Performance
_,reduc=train_reduc(x_train,reduc_type,n_c=best_nc)
roc,auc_test,desc=test_reduc(x_test,y_test,reduc=reduc,reduc_type=reduc_type,dis=dis)

roc.savefig('./plot/roc_{}_{}.png'.format(reduc_type,dis))
desc.to_csv('./{}_test_{}.csv'.format(reduc_type,best_nc))
