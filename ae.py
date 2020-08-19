import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from math import sin,cos,sqrt

from data_utils import *
from plot_utils import *

#Simple AE Model
DATA_DIM = 119
LATENT_DIM = 32
MODEL_NAME="ae"

class AE():
    def __init__(self):
        #Learning Rate
        self.lr=1e-3
        #Model
        self.encoder,self.decoder=self.make_model()
        #Optimizer
        self.ae_op=tf.keras.optimizers.Adam(lr=self.lr)
        #Trainables
        self.ae_vars=self.encoder.trainable_weights+self.decoder.trainable_weights


    def make_model(self):
        #119 -> 128 -> 64 -> 32 -> 64 -> 128 -> 119
        #Encoder
        e_in=tf.keras.layers.Input(shape=(DATA_DIM,))
        e1=tf.keras.layers.Dense(128, activation='relu',kernel_initializer='he_uniform')(e_in)
        e2=tf.keras.layers.Dense(64, activation='relu',kernel_initializer='he_uniform')(e1)
        z=tf.keras.layers.Dense(LATENT_DIM,kernel_initializer='he_uniform')(e2)
        encoder=keras.models.Model(inputs=e_in,outputs=z)
        encoder.summary()

        #Decoder
        d_in=tf.keras.layers.Input(shape=(LATENT_DIM,))
        d1=tf.keras.layers.Dense(64, activation='relu',kernel_initializer='he_uniform')(d_in)
        d2=tf.keras.layers.Dense(128, activation='relu',kernel_initializer='he_uniform')(d1)
        out=tf.keras.layers.Dense(DATA_DIM, activation='sigmoid',kernel_initializer='he_uniform')(d2)
        decoder=keras.models.Model(inputs=d_in,outputs=out)
        decoder.summary()

        return encoder,decoder

    def update_ae(self,grad):
        self.ae_op.apply_gradients(zip(grad,self.ae_vars))

    def save_model(self,save_path,epoch,batch):
        self.encoder.save(save_path+'/encoder{}_{}.h5'.format(epoch,batch))
        self.decoder.save(save_path+'/decoder{}_{}.h5'.format(epoch,batch))

class TrainAE:
    def __init__(self):
        #Get Model
        self.net=AE()

    def calc_loss(self,x):
        encoded=self.net.encoder(x)
        out=self.net.decoder(encoded)

        #Reconstruction Loss
        #Categorical Cross Entropy
        recon_loss=tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, out))*DATA_DIM
        #MSE
        # recon_loss=
        return recon_loss

    def train_batch(self,batch):
        with tf.GradientTape(persistent=True) as t:
            t.watch(self.net.ae_vars)
            recon_loss=self.calc_loss(batch)
        #Get Gradients
        ae_grads=t.gradient(recon_loss,self.net.ae_vars)

        #Apply Grads
        self.net.update_ae(ae_grads)

        return recon_loss

    def train(self,num_epochs,batch_size):
        #load data
        x_train,y_train=load_processed('train')
        x_val,y_val=load_processed('val')

        train_losses=[]
        val_losses=[]

        batch_print=1000
        print("Start Training For {} Epochs".format(num_epochs))
        for ep in range(num_epochs):
            #Shuffle
            np.random.shuffle(x_train)
            batch_iters=int(x_train.shape[0]/batch_size)

            batch_loss=0
            print("\nEpoch {}".format(ep+1))
            for i in range(batch_iters):
                #run batch
                cur_idx=i*batch_size
                batch=x_train[cur_idx:cur_idx+batch_size]
                # idx = np.random.randint(0, x_train.shape[0], batch_size)
                # batch = x_train[idx]

                train_recon=self.train_batch(batch)
                batch_loss+=train_recon
                #self.net.apply_grads(grads)
                if (i+1)%batch_print==0:
                    print('Batch loss {} recon:{:.5f}'.format(i+1,train_recon))

            #train epoch loss
            ep_loss=batch_loss/batch_iters
            train_losses.append(ep_loss)

            #Val Loss
            val_recon=self.calc_loss(x_val)
            val_losses.append(val_recon)

            print('Epoch recon loss Train:{:.5f} Val:{:.5f}'.format(ep_loss,val_recon))

        recon_plot=self.plot_losses(train_losses,val_losses,'Recon Loss')
        # recon_plot.set_title('Recon Loss')
        recon_plot.savefig('./plot/ae_recon_loss.png')
        # plt.clf()

kvae=TrainAE()
kvae.train(10,64)
