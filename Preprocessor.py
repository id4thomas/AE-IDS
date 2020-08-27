import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import KFold
import math
import h5py

class Preprocessor:
    def __init__(self):
        #read categorical data - service, flag
        service = open('./data/service.txt', 'r')
        self.serviceData = service.read().split('\n')
        service.close()
        flag = open('./data/flag.txt', 'r')
        self.flagData = flag.read().split('\n')
        flag.close()

    def getDataFrame(self, dir) :
        #make df from txt
        filelist = os.listdir(dir)
        frames = []
        for filename in filelist :
            print(dir+'/'+str(filename), end=' ')
            df = pd.read_csv(dir+'/'+filename, sep="\t", header = None)
            print(df.size)
            frames.append(df)
        sumDF = pd.concat(frames, ignore_index=True)
        return sumDF

    def toNumericData(self,df,save=None,filter=True):
        print('Data size')
        print(df.shape)

        if filter:
            print('filtering')
            #drop labels
            df = df[(df[17] < 0) | ((df[17]>0) & (df[14] == '0') & (df[15] == '0') & (df[16] == '0'))]
            print(df.shape)

        #service to categorical data
        print('phase 1 - service')
        df[1].replace(self.serviceData, range(len(self.serviceData)), inplace=True)
        #make flag to categorical data
        print('phase 13 - flag')
        df[13].replace(self.flagData, range(len(self.flagData)), inplace=True)
        #make protocol to categorical data
        print('phase 23 - protocol')
        df[23].replace(['tcp','udp','icmp'], range(0,3), inplace=True)

        #don't drop mal,ids,ash labels
        df.drop([18, 20, 22], axis=1, inplace=True)

        if save!=None:
            if not os.path.exists('./csv'):
                os.makedirs('./csv')
            df.to_csv('./csv/'+save+'.csv', sep="\t", header = None, index=False)
        return df

    def toAutoEncoderData(self,df):
        scaler = MinMaxScaler()
        #85+13+3+3+3 = 107
        enc = OneHotEncoder(categories=[range(len(self.serviceData)),range(len(self.flagData)),[0,1,2],[0,1,2],[0,1,2]])
        # enc = OneHotEncoder()
        numericDataDesc = df.loc[:, [0,2,3]].describe()

        print('phase 0 - duration')
        #0: duration
        iqr = (numericDataDesc[0].values[6]-numericDataDesc[0].values[4])*1.5
        standard = numericDataDesc[0].values[5]+iqr
        df[0] = df[0].map(lambda x : standard if x > standard else x)

        print('phase 2 - source bytes')
        #source bytes
        iqr = (numericDataDesc[2].values[6]-numericDataDesc[2].values[4])*1.5
        standard = numericDataDesc[2].values[5]+iqr
        if standard == 0 :
            df[2] = df[2].map(lambda x : 1 if x > 0 else 0)
        else :
            df[2] = df[2].map(lambda x : standard if x > standard else x)

        print('phase 3 - destination bytes')
        #destination bytes
        iqr = (numericDataDesc[3].values[6]-numericDataDesc[3].values[4])*1.5
        standard = numericDataDesc[3].values[5]+iqr
        if standard == 0 :
            df[3] = df[3].map(lambda x : 1 if x > 0 else 0)
        else :
            df[3] = df[3].map(lambda x : standard if x > standard else x)

        print('phase 4 - count')
        #count
        df[4] = df[4]/100

        print('phase 8 - dst host count')
        #dst host count
        df[8] = df[8]/100

        print('phase 9 - dst host srv count')
        #dst host srv count
        df[9] = df[9]/100

        #duration, source bytes, destination bytes
        scaler.fit(df[[0,2,3]].values)
        df[[0,2,3]] = scaler.transform(df[[0,2,3]].values)

        print('phase 17 - label')#feature idx 17 (label)
        df[17] = df[17].map(lambda x : 1 if x > 0 else 0)
        label = df[17].values.astype(np.int)
        label = label.reshape((label.shape[0],1))

        #make port_number as one-hot encoding
        #14,15,16,17,18,19,20 -> 14,15,16,17,19 -> 17,19
        print('phase 19') #port number reserved port, well-know port, unknown port => one hot encoding
        df[18] = df[18].map(lambda x : 2 if x > 49152 else 1 if x > 1024 else 0)

        print('phase 21') #port number reserved port, well-know port, unknown port => one hot encoding
        df[19] = df[19].map(lambda x : 2 if x > 49152 else 1 if x > 1024 else 0)

        #service, flag,
        print(df[[1,13,18,19,20]].values.shape)
        enc.fit(df[[1,13,18,19,20]].values)
        oneHotEncoding = enc.transform(df[[1,13,18,19,20]].values).toarray()

        #don't drop atk types
        df.drop([1,13,17,18,19,20], axis = 1, inplace=True)
        #df.drop([1,13,14,15,16,17,18,19,20], axis = 1, inplace=True)
        #122 features - 12,13,14 -> atk types

        #Atk Types
        print(df.head)
        #map atk labels 1: detected, 0: safe
        df[14] = df[14].map(lambda x : 0 if x == '0' else 1)
        df[15] = df[15].map(lambda x : 0 if x == '0' else 1)
        df[16] = df[16].map(lambda x : 0 if x == '0' else 1)

        atk_types=df.loc[:,[14,15,16]].to_numpy()
        df.drop([14,15,16],axis=1,inplace=True)
        #Atk_types + Label
        label=np.concatenate((atk_types,label),axis=1)

        return pd.DataFrame(np.concatenate((df,oneHotEncoding),axis=1)),label
