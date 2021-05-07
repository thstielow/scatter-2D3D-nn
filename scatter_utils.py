#Utility functions

import imageio
import numpy as np


import scipy.ndimage

#Read-in functions for png images of scattering patterns and real-space objects
#read scatter images
def read_image(path):
    raw_image = imageio.imread(path)
    img_shape = raw_image.shape
    if len(raw_image.shape) == 2:
        return raw_image.reshape(img_shape[0],img_shape[1],1)/65535
    else:
        return raw_image[:,:,0].reshape(img_shape[0],img_shape[1],1)/255

#read flattened object densities and re-arrange into 3D tensors
def read_density(img_path):
    rawImg = imageio.imread(img_path)
    if len(rawImg.shape) == 2:
        rawImg = 1 - rawImg/65535
    else:
        rawImg = 1 - rawImg[:,:,0]/255
    pix_len = int(round(np.power(np.prod(rawImg.shape),1/3)))
    (num_y, num_x) = np.array(rawImg.shape)/pix_len
    return np.array([np.hsplit(row, int(num_x)) for row in np.vsplit(rawImg, int(num_y))]).reshape(-1,pix_len,pix_len)

#rescale object tensors from 192x192x192 to 64x64x64
def transform_density(dens, target_size = (64,64,64)):
    dens_shape = dens.shape
    return ((np.round(scipy.ndimage.zoom(dens,(target_size[0]/dens_shape[0],target_size[1]/dens_shape[1],target_size[1]/dens_shape[1])))).reshape(*target_size, 1))


#Compression of threee dimensional object densisties into two dimensions by calculating the distance of the object surface to the tensor boundray in negative z-direction. This can be understood as a depth map when looking at the object in direction of the incoming beam.
def depth_from_dens(dens_tens):
    return np.max(dens_tens*np.flip(np.arange(dens_tens.shape[0])+1).reshape(dens_tens.shape[0],1,1,1),axis=0)/(dens_tens.shape[0])


#Implementation of the MSFT algorithm with material parameters of bulk silver. The parameters were adapted to work with the reduced object resolution of 64x64x64 instead of the 192x192x192 used for dataset creating
from numpy import fft

def scatterSim(tens, calcRes = 171, objLength=318.75, lam=13.5, alph=1/12.5):
    objRes = tens.shape[2]
    k = 2*np.pi / lam
    rStep = objLength / objRes
    qBox = 2 * np.pi / rStep
    QX, QY = np.meshgrid(np.linspace(-qBox/2,qBox/2,calcRes), np.linspace(-qBox/2,qBox/2,calcRes))
    QZ = np.real(k - np.sqrt((k**2 - (QX**2 + QY**2).astype('complex64'))))
    mask = (QX**2 + QY**2 < k**2)
    
    densDepth = np.zeros((tens.shape[1],tens.shape[2])).astype('complex64')
    field = np.zeros((calcRes,calcRes)).astype('complex64')
    for i in range(tens.shape[0]):
        densDepth += tens[i].squeeze()
        effDens = tens[i].squeeze() * np.exp(-alph * rStep*densDepth)
        field += fft.fftshift(fft.fft2(effDens, s=(calcRes,calcRes))) * np.exp(1j * QZ * rStep * i) * mask
    return np.flip(field)[calcRes//8:7*calcRes//8,calcRes//8:7*calcRes//8]