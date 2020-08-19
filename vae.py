#Simple VAE Model example
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from math import sin,cos,sqrt

from data_utils import *
from plot_utils import *

def myDistance(u, v):
    distance = 0.0
    # u = u[0]
    # v = v[0]
    for idx in range(u.shape[0]):
        distance += abs(u[idx]-v[idx])

    return distance

DATA_DIM=119
LATENT_DIM=2
MODEL_NAME="vae"

class VAE():
    def __init__(self):
        self.lr=1e-3
        self.encoder,self.decoder=self.make_model()
        self.op=tf.keras.optimizers.Adam()

    def make_model(self):
        #119 -> 128 -> 64 -> 32 -> 64 -> 128 -> 119
        #Encoder
        e_in=tf.keras.layers.Input(shape=(DATA_DIM,))
        e1=tf.keras.layers.Dense(128, activation='relu',kernel_initializer='he_uniform')(e_in)
        e2=tf.keras.layers.Dense(64, activation='relu',kernel_initializer='he_uniform')(e1)
        mu=tf.keras.layers.Dense(LATENT_DIM,kernel_initializer='he_uniform')(e2)
        sigma=tf.keras.layers.Dense(LATENT_DIM,kernel_initializer='he_uniform')(e2)

        encoder=keras.models.Model(inputs=e_in,outputs=[mu,sigma])

        d_in=tf.keras.layers.Input(shape=(LATENT_DIM,))
        d1=tf.keras.layers.Dense(64, activation='relu',kernel_initializer='he_uniform')(d_in)
        d2=tf.keras.layers.Dense(128, activation='relu',kernel_initializer='he_uniform')(d1)
        out=tf.keras.layers.Dense(DATA_DIM, activation='sigmoid',kernel_initializer='he_uniform')(d2)

        decoder=keras.models.Model(inputs=d_in,outputs=out)
        return encoder,decoder

    def encode(self,x):
        mu,sigma=self.encoder(x)
        return mu,sigma

    def sample(self,mu,sigma):
        sample=mu+tf.exp(sigma)*tf.random.normal(tf.shape(mu))
        return sample

    def decode(self,sample):
        return self.decoder(sample)

    def apply_grads(self,enc_grads,dec_grads):
        grads=enc_grads+dec_grads
        trainables=self.encoder.trainable_weights+self.decoder.trainable_weights
        self.op.apply_gradients(zip(grads,trainables))

    def save_model(self,save_path,epoch,train_config):
        self.encoder.save(save_path+'/encoder_{}_{}.h5'.format(train_config,epoch))
        self.decoder.save(save_path+'/decoder_{}_{}.h5'.format(train_config,epoch))

    def update_ae(self,grad):
        enc_vars=self.encoder.trainable_weights
        dec_vars=self.decoder.trainable_weights
        self.op.apply_gradients(zip(grad,enc_vars+dec_vars))

class TrainVAE:
    def __init__(self):
        self.net=VAE()

    def train_batch(self,batch):
        enc_vars=self.net.encoder.trainable_weights
        dec_vars=self.net.decoder.trainable_weights

        with tf.GradientTape(persistent=True) as t:
            t.watch(self.net.encoder.trainable_weights)
            t.watch(self.net.decoder.trainable_weights)
            batch_loss=self.calc_loss(batch)
            recon_loss=batch_loss[0]
            kl_loss=batch_loss[1]
            loss=recon_loss+kl_loss

        ae_grads=t.gradient(loss,enc_vars+dec_vars)
        self.net.update_ae(ae_grads)
        return [recon_loss,kl_loss]

    def calc_loss(self,x):
        mu,sigma=self.net.encoder(x)
        z=self.net.sample(mu,sigma)
        out=self.net.decoder(z)

        #MSE
        # recon_loss = tf.reduce_mean(tf.square(batch - out))
        #BCE
        recon_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, out))*DATA_DIM

        #KL Divergence Loss
        kl_loss=1+sigma-tf.square(mu)-tf.exp(sigma)
        kl_loss=-0.5*tf.reduce_sum(kl_loss,1)
        kl_loss=tf.reduce_mean(kl_loss)

        return [recon_loss,kl_loss]

    def plot_classes(self,data,label,epoch,batch_size,train_config):
        mu,sigma = self.net.encoder.predict(data)
        latent=np.array(mu)
        print(latent.shape)
        fig=plt.figure(figsize=(10, 10))
        plt.scatter(latent[:, 0], latent[:, 1], c=label)
        plt.colorbar()
        fig.savefig("{}/mu_{}_{}_{}.png".format(MODEL_NAME,train_config,epoch,batch_size))
        plt.clf()
        plt.close(fig)

        latent=np.array(sigma)
        fig=plt.figure(figsize=(10, 10))
        plt.scatter(latent[:, 0], latent[:, 1], c=label)
        plt.colorbar()
        fig.savefig("{}/sigma_{}_{}_{}.png".format(MODEL_NAME,train_config,epoch,batch_size))
        plt.clf()
        plt.close(fig)

    def train(self,num_epochs,batch_size):
        train_config='1_1'
        # x_train,y_train=self.load_one_class('train',1)
        x_train,y_train=load_data_config('train',train_config)
        x_val,y_val=load_data_config('val','1_1')
        x_test,y_test=get_hdf5_data('../kyoto_data/hdf5/test')

        # x_train,_=get_hdf5_data('../20_etri/hdf5/train')
        # x_val,y_val=get_hdf5_data('../20_etri/hdf5/val')

        train_losses=[[],[]]
        val_losses=[[],[]]

        batch_print=1000
        for ep in range(num_epochs):
            batch_iters=int(x_train.shape[0]/batch_size)
            batch_loss=[0,0]
            print("Epoch {}".format(ep+1))
            for i in range(batch_iters):
                #run batch
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                batch = x_train[idx]

                losses=self.train_batch(batch)
                batch_loss[0]+=losses[0]
                batch_loss[1]+=losses[1]

                if i%batch_print==0:
                    print('Batch loss {} recon:{:.5f}, kl:{:.5f}'.format(i,losses[0],losses[1]))
                #print("Batch {} Loss".format(i),loss)

            ep_loss=[l/batch_iters for l in batch_loss]
            losses.append(ep_loss)
            val_loss=self.calc_loss(x_val)

            for i in range(2):
                train_losses[i].append(ep_loss[i])
                val_losses[i].append(val_loss[i])

            print('Epoch loss recon:{:.5f}, kl:{:.5f}'.format(ep_loss[0],ep_loss[1]))
            print('Epoch val loss recon:{:.5f}, c_loss{:.5f}'.format(val_loss[0],val_loss[1]))

            self.net.save_model('./{}'.format(MODEL_NAME),ep+1,train_config)

            # val_idx = np.random.choice(x_val.shape[0], int(x_val.shape[0]*0.1), replace=False)
            # self.plot_classes(x_val,y_val,ep,batch_size,train_config)

        #Plot epoch Losses
        recon_plot=plot_losses(train_losses[0],val_losses[0],'Recon Loss')
        recon_plot.savefig('./vae/recon_loss.png')

        kl_plot=plot_losses(train_losses[1],val_losses[1],'KL Divergence Loss')
        kl_plot.savefig('./vae/kl_loss.png')


kvae=TrainVAE()
kvae.train(1,256)
