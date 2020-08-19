import pandas as pd
import numpy as np

from sklearn.decomposition import PCA, KernelPCA,SparsePCA,FastICA,TruncatedSVD

from perf_utils import *
from data_utils import *

from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report

ATK=0
SAFE=1

#Plot variance ratio for pca
def plot_var(data,reduc_type,kernel='rbf',n_c=8):
    _,reduc=train_reduc(data,reduc_type=reduc_type,kernel=kernel,n_c=n_c)
    vr=np.array(reduc.explained_variance_ratio_)
    print(reduc.explained_variance_ratio_)
    print(np.cumsum(vr))

    vrfig=plt.figure()
    pltauc=vrfig.add_subplot(1,1,1)
    pltauc.plot(range(vr.shape[0]),vr)
    pltauc.set_title('Variance Ratio')
    # fig.savefig('./plot/{}_vr.png'.format(reduc_type))
    # plt.clf()

    cvrfig=plt.figure()
    pltauc=cvrfig.add_subplot(1,1,1)
    pltauc.plot(range(vr.shape[0]),np.cumsum(vr))
    pltauc.set_title('Accumulated Variance Ratio')
    # fig.savefig('./plot/{}_cum_vr.png'.format(reduc_type))
    return vrfig,cvrfig

def train_reduc(data,reduc_type='pca',kernel='rbf',n_c=8):
    if reduc_type=='pca':
        reduc=PCA(n_components=n_c)
    elif reduc_type=='spca':
        reduc=SparsePCA(n_components=n_c)
    elif reduc_type=='kpca':
        reduc=KernelPCA(n_components=n_c,kernel=kernel)
    elif reduc_type=='ica':
        reduc=FastICA(n_components=n_c)
    elif reduc_type=='grp':
        pass
    elif reduc_type=='srp':
        pass

    reduced=reduc.fit_transform(data)
    print('Reduc Complete')
    return reduced,reduc

def test_reduc(data,label,reduc,reduc_type,dis='l1'):
    #Apply Reduc
    data_reduc=reduc.transform(data)
    #Recon
    if reduc_type in ['pca','kpca','ica']:
        #If inverse available
        data_recon=reduc.inverse_transform(data_reduc)
    elif reduc_type=='spca':
        #spca
        data_recon=np.array(data_reduc).dot(reduc.components_)+np.array(data.mean(axis=0))
    else:
        pass

    #Calculate Recon Loss
    if dis=='l1':
        dist=np.mean(np.abs(data-data_recon),axis=1)
    elif dis=='l2':
        dist=np.mean(np.square(data - data_recon),axis=1)

    roc,auc,desc=make_roc(dist,label,make_desc=True)
    return roc,auc,desc
