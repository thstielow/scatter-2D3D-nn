from tensorflow.keras.layers import *
from tensorflow.keras.models import *

#Basic layer definitions with integrated regularization
def clayer2D(x, n_fil, dropRate=0.2, keInit='GolorotUniform'):
    x = Conv2D(n_fil, 3, padding='same',activation=None, kernel_initializer=keInit)(x)
    x = Dropout(dropRate)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x

def clayer3D(x, n_fil, dropRate=0.2, keInit='GolorotUniform'):
    x = Conv3D(n_fil, 3, padding='same',activation=None, kernel_initializer=keInit)(x)
    x = Dropout(dropRate)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x


#Neural network constructor:
#The depth of the network is determined by the dimensions of the dataset and was not modularized.
def res2D3D(nBase, dropRate=0.2, keInit='he_normal'):
    inputs = Input((128,128,1))
    #input layer
    conv0 = Conv2D(nBase,7,strides=(2,2),padding='same', kernel_initializer=keInit)(inputs)
    lReLU0 = LeakyReLU()(conv0)
    bn0 = BatchNormalization()(lReLU0)
    pool0 = MaxPooling2D(pool_size=(2, 2))(bn0)

    
    conv1a = clayer2D(pool0, nBase*2, dropRate=dropRate, keInit=keInit)
    conv1b = clayer2D(conv1a, nBase*2, dropRate=dropRate, keInit=keInit)
    bypass1 = Conv2D(nBase*2,1, padding='same', kernel_initializer=keInit)(pool0)
    sum1 = Add()([bypass1, conv1b])
    pool1 = MaxPooling2D(pool_size=(2, 2))(sum1)

    conv2a = clayer2D(pool1, nBase*2**2, dropRate=dropRate, keInit=keInit)
    conv2b = clayer2D(conv2a, nBase*2**2, dropRate=dropRate, keInit=keInit)
    bypass2 = Conv2D(nBase*2**2,1, padding='same', kernel_initializer=keInit)(pool1)
    sum2 = Add()([bypass2, conv2b])
    pool2 = MaxPooling2D(pool_size=(2, 2))(sum2)

    conv3a = clayer2D(pool2, nBase*2**3, dropRate=dropRate, keInit=keInit)
    conv3b = clayer2D(conv3a, nBase*2**3, dropRate=dropRate, keInit=keInit)
    bypass3 = Conv2D(nBase*2**3,1, padding='same', kernel_initializer=keInit)(pool2)
    sum3 = Add()([bypass3, conv3b])
    pool3 = MaxPooling2D(pool_size=(2, 2))(sum3)

    conv4a = clayer2D(pool3, nBase*2**4, dropRate=dropRate, keInit=keInit)
    conv4b = clayer2D(conv4a, nBase*2**4, dropRate=dropRate, keInit=keInit)
    bypass4 = Conv2D(nBase*2**4,1, padding='same', kernel_initializer=keInit)(pool3)
    sum4 = Add()([bypass4, conv4b])
    pool4 = MaxPooling2D(pool_size=(2, 2))(sum4)

    conv5a = clayer2D(pool4, nBase*2**5, dropRate=dropRate, keInit=keInit)
    conv5b = clayer2D(conv5a, nBase*2**5, dropRate=dropRate, keInit=keInit)
    bypass5 = Conv2D(nBase*2**5,1, padding='same', kernel_initializer=keInit)(pool4)
    sum5 = Add()([bypass5, conv5b])
    pool5 = MaxPooling2D(pool_size=(2, 2))(sum5)

    #central vector
    flatten = Flatten()(pool5)
    dense = Dense(nBase*2**5)(flatten)
    denseAct = LeakyReLU()(dense)
    central = Reshape((1,1,1,nBase*2**5))(denseAct)

    #up-convolution
    upPool0 = UpSampling3D(size=(2,2,2))(central)
    upConv0a = clayer3D(upPool0, nBase*2**4, dropRate=dropRate, keInit=keInit)
    upConv0b = clayer3D(upConv0a, nBase*2**4, dropRate=dropRate, keInit=keInit)

    upPool1 = UpSampling3D(size=(2,2,2))(upConv0b)
    upConv1a = clayer3D(upPool1, nBase*2**3, dropRate=dropRate, keInit=keInit)
    upConv1b = clayer3D(upConv1a, nBase*2**3, dropRate=dropRate, keInit=keInit)

    upPool2 = UpSampling3D(size=(2,2,2))(upConv1b)
    upConv2a = clayer3D(upPool2, nBase*2**2, dropRate=dropRate, keInit=keInit)
    upConv2b = clayer3D(upConv2a, nBase*2**2, dropRate=dropRate, keInit=keInit)

    upPool3 = UpSampling3D(size=(2,2,2))(upConv2b)
    upConv3a = clayer3D(upPool3, nBase*2**1, dropRate=dropRate, keInit=keInit)
    upConv3b = clayer3D(upConv3a, nBase*2**1, dropRate=dropRate, keInit=keInit)

    upPool4 = UpSampling3D(size=(2,2,2))(upConv3b)
    upConv4a = clayer3D(upPool4, nBase, dropRate=dropRate, keInit=keInit)
    upConv4b = clayer3D(upConv4a, nBase, dropRate=dropRate, keInit=keInit)

    upPool5 = UpSampling3D(size=(2,2,2))(upConv4b)
    upConv5a = clayer3D(upPool5, nBase, dropRate=dropRate, keInit=keInit)
    upConv5b = clayer3D(upConv5a, nBase, dropRate=dropRate, keInit=keInit)

    #terminal layer
    compress = Conv3D(1,1,activation='sigmoid')(upConv5b)

    return Model(inputs = inputs, outputs = compress)