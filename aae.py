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

#Simple AAE Model

DATA_DIM = 119
LATENT_DIM = 32
MODEL_NAME="aae"

def gaussian(batch_size, n_dim, mean=0, var=1):
    #np.random.seed(0)
    z = np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)
    return z

class AAE():
    def __init__(self):
        self.encoder,self.decoder,self.disc=self.make_model()
        # self.e_op=tf.keras.optimizers.Adam(lr=0.001)
        # self.d_op=tf.keras.optimizers.Adam(lr=0.001)
        # self.disc_op=tf.keras.optimizers.Adam(lr=0.001)

        self.lr=1e-3
        self.ae_op=tf.keras.optimizers.Adam(lr=self.lr)
        self.disc_op=tf.keras.optimizers.Adam(lr=self.lr)
        self.gen_op=tf.keras.optimizers.Adam(lr=self.lr)


    def make_model(self):
        #119 -> 128 -> 64 -> 32 -> 64 -> 128 -> 119
        #Encoder
        e_in=tf.keras.layers.Input(shape=(DATA_DIM,))
        e1=tf.keras.layers.Dense(256, activation='relu',kernel_initializer='he_uniform')(e_in)
        # e1=tf.keras.layers.Dropout(0.5)(e1)
        e2=tf.keras.layers.Dense(128, activation='relu',kernel_initializer='he_uniform')(e1)
        # e2=tf.keras.layers.Dropout(0.5)(e2)
        z=tf.keras.layers.Dense(LATENT_DIM,kernel_initializer='he_uniform')(e2)
        encoder=keras.models.Model(inputs=e_in,outputs=z)
        encoder.summary()

        #Decoder
        d_in=tf.keras.layers.Input(shape=(LATENT_DIM,))
        d1=tf.keras.layers.Dense(128, activation='relu',kernel_initializer='he_uniform')(d_in)
        # d1=tf.keras.layers.Dropout(0.5)(d1)
        d2=tf.keras.layers.Dense(256, activation='relu',kernel_initializer='he_uniform')(d1)
        # d2=tf.keras.layers.Dropout(0.5)(d2)
        out=tf.keras.layers.Dense(DATA_DIM, activation='sigmoid',kernel_initializer='he_uniform')(d2)
        decoder=keras.models.Model(inputs=d_in,outputs=out)
        decoder.summary()

        #Discriminator
        z_in=tf.keras.layers.Input(shape=(LATENT_DIM,)) #Latent Dist
        dc1=tf.keras.layers.Dense(128, activation='relu',kernel_initializer='he_uniform')(z_in)
        # dc1=tf.keras.layers.Dropout(0.5)(dc1)
        dc2=tf.keras.layers.Dense(64, activation='relu',kernel_initializer='he_uniform')(dc1)
        # dc2=tf.keras.layers.Dropout(0.5)(dc2)
        dc_out=tf.keras.layers.Dense(1, activation='sigmoid',kernel_initializer='he_uniform')(dc2)
        disc=keras.models.Model(inputs=z_in,outputs=dc_out)
        return encoder,decoder,disc

    def disc_latent(self,real_g,fake_g):
        d_real=self.disc(real_g)
        d_fake=self.disc(fake_g)
        return d_real,d_fake


    def apply_grads(self,grads):
        enc_vars=self.encoder.trainable_weights
        dec_vars=self.decoder.trainable_weights
        disc_vars=self.disc.trainable_weights

        self.ae_op.apply_gradients(zip(grads[0],enc_vars+dec_vars))
        self.disc_op.apply_gradients(zip(grads[1],disc_vars))
        self.gen_op.apply_gradients(zip(grads[2],enc_vars))

    def update_ae(self,grad):
        enc_vars=self.encoder.trainable_weights
        dec_vars=self.decoder.trainable_weights
        self.ae_op.apply_gradients(zip(grad,enc_vars+dec_vars))

    def update_disc(self,grad):
        disc_vars=self.disc.trainable_weights
        self.disc_op.apply_gradients(zip(grad,disc_vars))

    def update_gen(self,grad):
        enc_vars=self.encoder.trainable_weights
        self.gen_op.apply_gradients(zip(grad,enc_vars))

    def save_model(self,save_path,epoch,train_config):
        self.encoder.save(save_path+'/encoder_{}_{}.h5'.format(train_config,epoch))
        self.decoder.save(save_path+'/decoder_{}_{}.h5'.format(train_config,epoch))
        self.disc.save(save_path+'/disc_{}_{}.h5'.format(train_config,epoch))

class TrainAAE:
    def __init__(self):
        self.net=AAE()

    def calc_loss(self,x):
        #get latent z
        z=self.net.encoder(x)
        out=self.net.decoder(z)

        #get real gaussian
        real_g=gaussian(x.shape[0],LATENT_DIM)

        #Recon Loss
        #BCE
        # recon_loss=tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, out))*DATA_DIM
        #MSE
        recon_loss = tf.reduce_mean(tf.square(x - out))

        #Discriminator Loss
        d_real,d_fake=self.net.disc_latent(real_g,z)
        dc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
        dc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
        disc_loss = 0.5*dc_loss_fake + 0.5*dc_loss_real

        #Generator Loss
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

        return recon_loss,disc_loss,g_loss

    def train_batch(self,batch):
        enc_vars=self.net.encoder.trainable_weights
        dec_vars=self.net.decoder.trainable_weights
        disc_vars=self.net.disc.trainable_weights

        with tf.GradientTape(persistent=True) as t:
            t.watch(enc_vars+dec_vars+disc_vars)
            recon_loss,disc_loss,g_loss=self.calc_loss(batch)

        ae_grads=t.gradient(recon_loss,enc_vars+dec_vars)
        self.net.update_ae(ae_grads)

        disc_grads=t.gradient(disc_loss,disc_vars)
        self.net.update_disc(disc_grads)

        gen_grads=t.gradient(g_loss,enc_vars)
        self.net.update_gen(gen_grads)

        return[recon_loss,disc_loss,g_loss]

    def train(self,num_epochs,batch_size):
        # x_train,y_train=self.load_data('train')
        # x_val,y_val=self.load_data('val')
        train_config='1_1'
        x_train,y_train=load_data_config('train',train_config)
        x_val,y_val=load_data_config('val','1_1')
        x_test,y_test=get_hdf5_data('../kyoto_data/hdf5/test')

        #Train Losses - Recon, Disc, Gen
        train_losses=[[],[],[]]
        val_losses=[[],[],[]]

        batch_print=1000
        print("Start Training For {} Epochs".format(num_epochs))
        for ep in range(num_epochs):
            np.random.shuffle(x_train)
            batch_iters=int(x_train.shape[0]/batch_size)
            batch_loss=[0,0,0]
            print("\nEpoch {}".format(ep+1))
            for i in range(batch_iters):
                #run batch
                cur_idx=i*batch_size
                batch=x_train[cur_idx:cur_idx+batch_size]
                # idx = np.random.randint(0, x_train.shape[0], batch_size)
                # batch = x_train[idx]

                losses=self.train_batch(batch)
                batch_loss[0]+=losses[0]
                batch_loss[1]+=losses[1]
                batch_loss[2]+=losses[2]
                #self.net.apply_grads(grads)
                if i%batch_print==0:
                    print('Batch loss {} recon:{:.5f}, disc:{:.5f} g_loss:{:.5f}'.format(i,losses[0],losses[1],losses[2]))

            ep_loss=[l/batch_iters for l in batch_loss]
            #Val Loss
            val_loss=self.calc_loss(x_val)

            for i in range(3):
                train_losses[i].append(ep_loss[i])
                val_losses[i].append(val_loss[i])

            print('Epoch loss recon:{:.5f}, disc:{:.5f} g_loss:{:.5f}'.format(ep_loss[0],ep_loss[1],ep_loss[2]))
            print('Val loss recon:{:.5f}, disc:{:.5f} g_loss:{:.5f}'.format(val_loss[0],val_loss[1],val_loss[2]))
            self.net.save_model('./'+MODEL_NAME,ep+1,train_config)

        recon_plot=plot_losses(train_losses[0],val_losses[0],'Recon Loss')
        recon_plot.savefig('./aae/recon_loss.png')

        disc_plot=plot_losses(train_losses[1],val_losses[1],'Disc Loss')
        disc_plot.savefig('./aae/disc_loss.png')

        g_plot=plot_losses(train_losses[2],val_losses[2],'Generator Loss')
        g_plot.savefig('./aae/g_loss.png')


kvae=TrainAAE()
kvae.train(15,256)
