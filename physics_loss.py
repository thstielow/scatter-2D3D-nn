import numpy as np
import tensorflow as tf
from tensorflow import keras

imgSize=128
densSize=64

#Tensorflow implementation of the MSFT algoritm
@tf.function
def forwardScatter_tf(tens):
    calcRes = tf.constant(170, dtype=tf.int32)
    objLength = tf.constant(318.75, dtype=tf.float32)
    lam = tf.constant(13.5, dtype=tf.float32)
    alph = tf.constant(1/12.5, dtype=tf.float32)
    objRes = tf.constant(densSize, dtype=tf.int32)
    k = 2*np.pi / lam
    rStep = objLength / tf.cast(objRes,'float32')
    qBox = 2 * np.pi / rStep
    QX, QY = tf.meshgrid(tf.linspace(-qBox/2,qBox/2,calcRes), tf.linspace(-qBox/2,qBox/2,calcRes))
    QZ = tf.math.real(tf.cast(k, 'complex64') - tf.sqrt(tf.cast((k**2 - (QX**2 + QY**2)),'complex64')))
    mask = tf.cast((QX**2 + QY**2 < k**2),'complex64')
        
    
    depthSlices = tf.cast(tf.scan(lambda a, x: a + x, tf.squeeze(tens)),'float32')
    effDens = tf.cast(tf.cast(tf.squeeze(tens),'float32') * tf.exp(-alph * rStep * depthSlices),'complex64')
    scatterSlices = tf.signal.fftshift(tf.signal.fft2d(tf.pad(effDens,[[0,0],[(calcRes-objRes)//2,(calcRes-objRes)//2],[(calcRes-objRes)//2,(calcRes-objRes)//2]])), axes=[1,2])
    phase=tf.exp(1j * tf.cast(rStep, dtype=tf.complex64) * tf.cast(tf.tensordot(tf.cast(tf.range(objRes), 'float32'), QZ, axes=0),dtype=tf.complex64))
    field =  tf.reverse(tf.math.reduce_sum(scatterSlices * phase,axis=0) * mask, axis=[0,1])[calcRes//8:7*calcRes//8,calcRes//8:7*calcRes//8]
    
    intensity = tf.math.log(tf.cast(tf.abs(field),'float32')**2+ 1.0*10**2)
    intensity = intensity - tf.math.reduce_min(intensity, axis=[-2,-1],keepdims=True)
    intensity = intensity / tf.math.reduce_max(intensity, axis=[-2,-1],keepdims=True)
    return intensity


#evaluates on whole batch -> NN Ready!
@tf.function
def forwardScatter_batch(obj_batch):
    return tf.map_fn(forwardScatter_tf, obj_batch, dtype='float32')


#construct the scatter loss with MSFT function
@tf.function
def scatterLoss(y_true, y_pred):
    return keras.losses.mean_squared_error(forwardScatter_batch(y_true), forwardScatter_batch(y_pred + tf.random.uniform(tf.shape(y_pred),maxval=0.00001)))

#definition of the binary loss regularization function
@tf.function
def binaryLoss(y_true, y_pred):
    return y_pred**2 * (y_pred-tf.constant(1.0))**2

#definition of the physics informed loss function
@tf.function
def physicsLoss(y_true, y_pred):
    return tf.reduce_mean(binaryLoss(y_true ,y_pred))*tf.constant(0.1) + tf.reduce_mean(scatterLoss(y_true, y_pred))