import Preprocessor
import os
import pandas as pd
import numpy as np

import os
def preprocess(type):
    pp=Preprocessor.Preprocessor()
    df=pp.getDataFrame('./data/original/'+type)
    pp.toNumericData(df,save=type)
    df=pd.read_csv('./csv/'+type+'.csv', sep="\t", header = None)


    df,label=pp.toAutoEncoderData(df)
    print('df {} Label {}'.format(df.shape,label.shape))
    print(df.head)
    df.to_csv('./data/processed/'+type+'.csv', sep="\t", header = None, index=False)
    np.save('./data/processed/'+type+'_label.npy',label)

if not os.path.exists('./data/processed'):
    os.makedirs('./data/processed')

preprocess('train')
preprocess('val')
