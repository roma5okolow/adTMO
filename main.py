import os 

import numpy as np
import cv2

from keras import models

import tensorflow as tf
from tensorflow import keras
from keras import Input
from keras.models import Model

from keras.layers import Concatenate
from keras.layers import LeakyReLU
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import ReLU
from keras.layers import Conv2DTranspose
from keras.layers import Add
from keras.layers import Dropout

from keras.optimizers import Adam

from keras.initializers import RandomNormal

from keras import activations

from keras.applications import VGG19

from keras.layers import Layer, InputSpec
import keras.backend as K
from tqdm import tqdm
from tqdm.contrib import tzip
from PIL import Image



import shutil

IMG_SIZE = 256
TRAIN_SIZE = 516
TEST_SIZE = 105
#Format for opencv is BGR

# train_data_hdr = []
# train_data_ldr = []
# train_files_hdr = os.listdir('tmo_dataset_256/train/hdr')
# train_files_ldr = os.listdir('tmo_dataset_256/train/ldr')

# for idx, (file_hdr, file_ldr) in enumerate(tzip(train_files_hdr, train_files_ldr)):
#     try:
#       img = cv2.imread(os.path.join('tmo_dataset_256/train/hdr', file_hdr), flags=cv2.IMREAD_ANYDEPTH)
#       img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
#       img = np.moveaxis(img, -1, 0)
#       img = (img - img.min()) / (img.max() - img.min())
#       img_lum = 0.2959 * img[2] + 0.587 * img[1] + 0.114 * img[0]
#       train_data_hdr.append(img_lum)

#       img = cv2.imread(os.path.join('tmo_dataset_256/train/ldr', file_ldr))
#       img = np.moveaxis(img, -1, 0)
#       img = img / 255
#       img_lum = 0.2959 * img[2] + 0.587 * img[1] + 0.114 * img[0]
#       train_data_ldr.append(img_lum)
#     except:
#       print(file_hdr, ' FAILED TO READ')
    
# np.savez('train_image_pairs.npz', np.array(train_data_hdr), np.array(train_data_ldr))


# test_data_hdr = []
# test_data_ldr = []
# test_files_hdr = os.listdir('tmo_dataset_256/test/hdr')
# test_files_ldr = os.listdir('tmo_dataset_256/test/ldr')

# for idx, (file_hdr, file_ldr) in enumerate(tzip(test_files_hdr, test_files_ldr)):
#     try:
#       img = cv2.imread(os.path.join('tmo_dataset_256/test/hdr', file_hdr), flags=cv2.IMREAD_ANYDEPTH)
#       img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
#       img = np.moveaxis(img, -1, 0)
#       img = (img - img.min()) / (img.max() - img.min())
#       img_lum = 0.2959 * img[2] + 0.587 * img[1] + 0.114 * img[0]
#       test_data_hdr.append(img_lum)

#       img = cv2.imread(os.path.join('tmo_dataset_256/test/ldr', file_ldr))
#       img = np.moveaxis(img, -1, 0)
#       img = img / 255
#       img_lum = 0.2959 * img[2] + 0.587 * img[1] + 0.114 * img[0]
#       test_data_ldr.append(img_lum)
#     except:
#       print(file_hdr, ' FAILED TO READ')

# np.savez('test_image_pairs.npz', np.array(test_data_hdr), np.array(test_data_ldr))

# test_data_hdr = []
# test_data_ldr = []
# test_files_hdr = os.listdir('tmo_dataset_256/test/hdr')
# test_files_ldr = os.listdir('tmo_dataset_256/test/ldr')

# for idx, (file_hdr, file_ldr) in enumerate(tzip(test_files_hdr, test_files_ldr)):
#     try:
#       img = cv2.imread(os.path.join('tmo_dataset_256/test/hdr', file_hdr), flags=cv2.IMREAD_ANYDEPTH)
#       img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
#       img = (img - img.min()) / (img.max() - img.min())
#       test_data_hdr.append(img)

#       img = cv2.imread(os.path.join('tmo_dataset_256/test/ldr', file_ldr))
#       img = img / 255
#       test_data_ldr.append(img)
#     except:
#       print(file_hdr, ' FAILED TO READ')

# np.savez('test_image_pairs_color.npz', np.array(test_data_hdr), np.array(test_data_ldr))


from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution
disable_eager_execution()
class GAN_model():
    
    def __init__(self):
        self.train_batch_size = 1
        self.test_batch_size = 1
        self.image_row = 256
        self.image_col = 256
        self.channel = 1
        self.image_shape = (self.image_row, self.image_col, self.channel)
        self.patch_row = self.image_row // 16
        self.patch_col = self.image_col // 16
        self.patch_shape = (self.train_batch_size, self.patch_row, self.patch_col, 1)
        self.mode = 'inference'
        
        self.train_file_name = 'train_image_pairs.npz'
        if not os.path.exists(self.train_file_name):
            self.imgs_to_train_npy()
        self.test_file_name = 'test_image_pairs.npz'
        if not os.path.exists(self.test_file_name):
            self.imgs_to_test_npy()
        
        self.test_file_name_color = 'test_image_pairs_color.npz'
        
        self.iters_per_check = 2000
        
        self.my_vgg_models = self.define_my_vgg()
        self.D = self.build_d()
        self.D_light = self.build_d_light()
        self.G = self.build_g()
        self.GAN = self.build_gan()
        
        self.iterations = 300000
        self.real_patch_out = np.ones(self.patch_shape, dtype=np.float32)
        self.fake_patch_out = np.zeros(self.patch_shape, dtype=np.float32)
       
        self.train_data_generator = self.train_data_loader()
        self.test_data_generator = self.test_data_loader(mode=self.mode)
    def build_d(self):
     # with strategy.scope():
        features = []
        init = RandomNormal(stddev=0.02)
        hdr = Input(shape=(None, None, self.channel))    
        ldr = Input(shape=(None, None, self.channel))
        x = Concatenate()([hdr, ldr])
        
        x = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x) # kernel_initializer=init?
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        features.append(x)
        
        x = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        features.append(x)
        
        x = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        features.append(x)
        
        x = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        features.append(x)
        
        x = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        features.append(x)
        
        x = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(x)
        patch_out = Activation('sigmoid')(x)
        
        model = Model([hdr, ldr], [patch_out] + features)
        model.summary()
        return model
    
    def build_d_light(self):
        #with strategy.scope():
          hdr = Input(shape=(None, None, self.channel))    
          ldr = Input(shape=(None, None, self.channel))
          x = self.D([hdr, ldr])[0]
          opt = Adam(lr=0.0002, beta_1=0.5, beta_2 = 0.999)
          model = Model([hdr, ldr], x)
          model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
          return model
    
    
    # define an encoder block
    def define_encoder_block(self, layer_in, n_filters, batchnorm=True):
      #with strategy.scope():
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # add downsampling layer
        g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
        # conditionally add batch normalization
        if batchnorm:
              g = BatchNormalization()(g, training=True)
        # leaky relu activation
        g = LeakyReLU(alpha=0.2)(g)
        return g
    
    # define a decoder block
    def decoder_block(self, layer_in, skip_in, n_filters, dropout=True):
      #with strategy.scope():  # weight initialization
        init = RandomNormal(stddev=0.02)
        # add upsampling layer
        g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
        # add batch normalization
        g = BatchNormalization()(g, training=True)
        # conditionally add dropout
        if dropout:
            g = Dropout(0.5)(g, training=True)
        # merge with skip connection
        g = Concatenate()([g, skip_in])
        # relu activation
        g = Activation('relu')(g)
        return g
    
    # define the standalone generator model
    def build_g(self):
      #with strategy.scope():
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # image input
        # in_image = Input(shape=self.image_shape)
        in_image = Input(shape=(None, None, self.channel))
            
        # encoder model
        e1 = self.define_encoder_block(in_image, 64, batchnorm=False)
        e2 = self.define_encoder_block(e1, 128)
        e3 = self.define_encoder_block(e2, 256)
        e4 = self.define_encoder_block(e3, 512)
        e5 = self.define_encoder_block(e4, 512)
        e6 = self.define_encoder_block(e5, 512)
        e7 = self.define_encoder_block(e6, 512)
        # bottleneck, no batch norm and relu
        b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
        b = Activation('relu')(b)
        # decoder model
        d1 = self.decoder_block(b, e7, 512)
        d2 = self.decoder_block(d1, e6, 512)
        d3 = self.decoder_block(d2, e5, 512)
        d4 = self.decoder_block(d3, e4, 512, dropout=False)
        d5 = self.decoder_block(d4, e3, 256, dropout=False)
        d6 = self.decoder_block(d5, e2, 128, dropout=False)
        d7 = self.decoder_block(d6, e1, 64, dropout=False)
        # output
        g = Conv2DTranspose(self.channel , (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
        out_image = Activation('sigmoid')(g)
        # define model
        model = Model(in_image, out_image)
        model.summary()
        return model
    
    def build_gan(self):
       # with strategy.scope():
          self.D.trainable = False
          
          hdr = Input(shape=(None, None, self.channel))
          ldr = Input(shape=(None, None, self.channel))
          
          def feature_matching_loss(real_ldr, fake_ldr, hdr=hdr, D=self.D):
              loss_feature_match = 0
              fake_features = D([hdr, fake_ldr])[1:]
              real_features = D([hdr, real_ldr])[1:]
              
              for i in range(len(real_features)):
                  loss_feature_match += K.mean(K.abs(fake_features[i] - real_features[i]))
              return loss_feature_match
          
          def VGG19_loss(y_true, y_pred, vgg_models=self.my_vgg_models, channels = self.channel):
              loss_vgg19 = 0
              if channels == 1:
                  y_true = Concatenate()([y_true, y_true, y_true])
                  y_pred = Concatenate()([y_pred, y_pred, y_pred])
                      
              for model in vgg_models:
                  model.trainable = False
                  loss_vgg19 += K.mean(K.abs(model(y_true) - model(y_pred)))
              return loss_vgg19
          
          
          generated_ldr = self.G(hdr)        
          patch_out = self.D_light([hdr, generated_ldr])        
          
          
          model = Model(inputs=[hdr, ldr], outputs=[patch_out, generated_ldr, generated_ldr])
          # model = Model(inputs=[hdr, ldr], outputs=[patch_out, generated_ldr])
          model.summary()
          opt = Adam(lr=0.0002, beta_1=0.5, beta_2 = 0.999)
          # model.compile(loss=['binary_crossentropy', feature_matching_loss, VGG19_loss], optimizer=opt, loss_weights=[1, 10, 10])
          model.compile(loss=['binary_crossentropy', feature_matching_loss, VGG19_loss], optimizer=opt, loss_weights=[1, 10, 10])
        # , metrics = {"model_18":'binary_crossentropy', "model_19":[feature_matching_loss, VGG19_loss]}, 
          return model
    
    def define_my_vgg(self):
      #with strategy.scope():
        vgg = VGG19(include_top=False, input_shape=(self.image_row, self.image_row, 3))
        vgg.trainable = False
        models = []
        for i in [1,2,4,5,7,8,9,10,12,13,14,15,17,18,19,20]:
            model = Model(inputs=vgg.input, outputs=vgg.layers[i].output)
            model.trainable = False
            models.append(model)
        return models
    
    def train(self):
     # with strategy.scope():
        for iter_idx in tqdm(range(self.iterations)):
            hdr, ldr = next(self.train_data_generator)
            generated_ldr = self.G.predict(hdr)
            self.D_light.trainable = True
            self.G.trainable = False
            self.D_light.train_on_batch([hdr, ldr], self.real_patch_out)
            self.D_light.train_on_batch([hdr, generated_ldr], self.fake_patch_out)       
            self.D_light.trainable = False
            self.G.trainable = True
            self.GAN.train_on_batch([hdr, ldr], [self.real_patch_out, ldr, ldr])
            if (iter_idx + 1) % self.iters_per_check == 0:
                print(self.GAN.evaluate([hdr, ldr], [self.real_patch_out, ldr, ldr]))
                self.show_performance(iter_idx+1)
                self.save_model()
                
    def show_performance(self, iter_idx):
        hdr, ldr = next(self.test_data_generator)
        print(self.GAN.evaluate([hdr, ldr], [self.real_patch_out, ldr, ldr]))
        generated_ldr = self.G.predict(hdr)        
        hdr = np.reshape(hdr, self.image_shape)
        ldr = np.reshape(ldr, self.image_shape)
        generated_ldr = np.reshape(generated_ldr, self.image_shape)
        hdr *= 255
        hdr = hdr.astype(np.uint8)        
        hdr = np.reshape(hdr, (256,256))
        hdr = Image.fromarray(hdr, 'L')
        hdr.save('imgs/' + str(iter_idx) + 'test_hdr.jpg')
        
        
        ldr *= 255
        ldr = ldr.astype(np.uint8)
        ldr = np.reshape(ldr, (256,256))
        ldr = Image.fromarray(ldr, 'L')
        ldr.save('imgs/' + str(iter_idx) + 'test_ldr.jpg')        
        
        generated_ldr *= 255
        generated_ldr = generated_ldr.astype(np.uint8)
        generated_ldr = np.reshape(generated_ldr, (256,256))
        generated_ldr = Image.fromarray(generated_ldr, 'L')
        generated_ldr.save('imgs/' + str(iter_idx) + 'test_generated_ldr.jpg')

    def inference(self):
        test_files_names = list(map(lambda x: x[:-4], os.listdir('tmo_dataset_256/test/hdr')))

        for idx, (hdr, ldr) in enumerate(tqdm(self.test_data_generator)):
            print(idx)
            if idx == TEST_SIZE:
                break
            hdr_lum = 0.2959 * hdr[:, :, 2] + 0.587 * hdr[:, :, 1] + 0.114 * hdr[:, :, 0]
            hdr_lum = np.reshape(hdr_lum, (self.test_batch_size, 256, 256, self.channel))
            generated_ldr_lum = self.G.predict(hdr_lum).squeeze()
            hdr_lum = hdr_lum.squeeze()
            generated_ldr = np.zeros((256, 256, 3))
            for i in range(3):
                generated_ldr[:, :, i] = hdr[:, :, i] * generated_ldr_lum / hdr_lum
            generated_ldr *= 255
            generated_ldr = np.clip(generated_ldr, 0, 255)
            generated_ldr = generated_ldr.astype(np.uint8)
            cv2.imwrite(('imgs/test/' + test_files_names[idx] + '_gen.jpg'), generated_ldr)

            ldr *= 255
            ldr = ldr.astype(np.uint8)
            cv2.imwrite(('imgs/test/' + test_files_names[idx] + '_gt.jpg'), ldr)
 
    def train_data_loader(self):
        batch_size = self.train_batch_size
        filename = self.train_file_name
        train_data = np.load(filename, allow_pickle=True)
        hdr, ldr = train_data['arr_0'], train_data['arr_1']
        size = len(hdr)
        ids = list(range(size))    
        batchs = size//batch_size
        while True:
            np.random.shuffle(ids)
            for i in range(batchs):
                ids_this_batch = ids[i*batch_size:(i+1)*batch_size]
                hdr_this_batch = [hdr[idx] for idx in ids_this_batch]
                ldr_this_batch = [ldr[idx] for idx in ids_this_batch]
                hdr_this_batch = np.reshape(hdr_this_batch, (batch_size, self.image_row, self.image_col, self.channel))
                ldr_this_batch = np.reshape(ldr_this_batch, (batch_size, self.image_row, self.image_col, self.channel))

                yield hdr_this_batch, ldr_this_batch
        
    def test_data_loader(self, mode='train'):
        batch_size = self.test_batch_size
        if mode == 'train':
            filename = self.test_file_name
        else:
            filename = self.test_file_name_color
        train_data = np.load(filename, allow_pickle=True)
        hdr, ldr = train_data['arr_0'], train_data['arr_1']
        size = len(hdr)
        ids = list(range(size))    
        batchs = size//batch_size
        while True:
            for i in range(batchs):
                ids_this_batch = ids[i*batch_size:(i+1)*batch_size]
                # print(ids_this_batch)
                hdr_this_batch = [hdr[idx] for idx in ids_this_batch]
                ldr_this_batch = [ldr[idx] for idx in ids_this_batch]
                if mode == 'train':
                    hdr_this_batch = np.reshape(hdr_this_batch, (batch_size, self.image_row, self.image_col, self.channel))
                    ldr_this_batch = np.reshape(ldr_this_batch, (batch_size, self.image_row, self.image_col, self.channel))
                else:
                    hdr_this_batch = np.reshape(hdr_this_batch, (self.image_row, self.image_col, 3))
                    ldr_this_batch = np.reshape(ldr_this_batch, (self.image_row, self.image_col, 3))
                yield hdr_this_batch, ldr_this_batch

    
    def save_model(self):
        
        def freeze(model):
            for layer in model.layers:
                layer.trainable = False
            
            if isinstance(layer, models.Model):
                freeze(layer)
                
        G = self.G
        
        
        D = self.D
        freeze(G)
        freeze(D)
        
        G.save('G_model')
        print('G saved')
        D.save('D_model')
        print('D saved')

gan = GAN_model()
gan.G = tf.keras.models.load_model('G_model')
gan.D = tf.keras.models.load_model('D_model')
gan.inference()
# gan.train()
# gan.save_model()