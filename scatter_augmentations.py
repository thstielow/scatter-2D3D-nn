#Image augmentation functions for inpirt scattering patterns
import numpy as np
import scipy
import skimage
from skimage import draw
from skimage import transform
import scipy.ndimage
import tensorflow.keras as keras

def add_nothing(img_data):
    return img_data

def add_noise_uniform(img_data):
    img_shape = img_data.shape[:2]
    return img_data + np.random.rand() * 0.2 * np.random.rand(img_shape[0], img_shape[1], 1)

def add_noise_poisson(img_data):
    aug_data = img_data + np.random.rand() * 0.1 * np.random.poisson(lam=1.0,size=img_data.shape)
    return aug_data / np.max(aug_data)

def add_shot(img_data):
    aug_data = img_data * np.random.poisson(lam=10**(np.random.rand()+1),size=img_data.shape)
    return aug_data / np.max(aug_data)

def add_blur(img_data):
    return scipy.ndimage.filters.gaussian_filter(img_data, 1.0*np.random.rand()) 

def add_shift(img_data):
    return scipy.ndimage.shift(img_data, (5*(np.random.rand()*2-1),5*(np.random.rand()*2-1),0))

def add_rotation(img_data):
    return scipy.ndimage.rotate(img_data,np.random.randint(0,360),axes=(1,0),reshape=False)

def add_hole(img_data):
    holed = img_data.copy()
    rr, cc = draw.circle(img_data.shape[0]//2, img_data.shape[1]//2, radius=(10 * np.random.rand()), shape=img_data.shape)
    holed[rr, cc] = 0
    return holed / np.max(holed)

def add_bar(img_data):
    bar_width = int(np.random.rand()*img_data.shape[1]/20)
    mask = np.ones(img_data.shape)
    rr, cc = draw.rectangle((img_data.shape[1]//2-1-bar_width,0),end=(img_data.shape[1]//2-1+bar_width,img_data.shape[0]-1))
    mask[rr, cc] = 0
    return img_data * mask

def add_noise_snp(img_data):    
    return skimage.util.noise.random_noise(img_data, mode='s&p', amount=0.05*np.random.rand())

def add_rect_crop(img_data):
    w_crop = int(np.random.rand()*img_data.shape[1]/4)
    h_crop = int(np.random.rand()*img_data.shape[0]/4)
    mask = np.zeros(img_data.shape)
    rr, cc = draw.rectangle((h_crop,w_crop),end=(img_data.shape[0]-1-h_crop,img_data.shape[1]-1-w_crop))
    mask[rr, cc] = 1
    return img_data * mask

def add_circle_crop(img_data):
    mask = np.zeros(img_data.shape)
    rr, cc = draw.circle(img_data.shape[0]//2, img_data.shape[1]//2, radius=(img_data.shape[0]*(np.random.rand()+1)/4), shape=img_data.shape)
    mask[rr, cc] = 1
    return mask*img_data

def add_crop(img_data):
    return add_rect_crop(add_circle_crop(img_data))

def add_sat(img_data):
    return np.minimum((1 + 0.5*np.random.rand()) * img_data, 1)

def add_blindspot(img_data):
    x,y=np.meshgrid(np.linspace(-1,1,img_data.shape[1]),np.linspace(-1,1,img_data.shape[0]))
    blinded = img_data * (1 - (np.exp(-((x-(np.random.rand()-0.5))**2 + (y-(np.random.rand()-0.5))**2)/(2*(np.random.rand()/4)**2))).reshape(*img_data.shape))
    return blinded / np.max(blinded)

def sim_exp(img_data):
    return add_shift(add_crop(add_hole(add_sat(add_blindspot(add_shift(add_shot(img_data)))))))


#dictionary of all augmentation filters:
augments = {
    0: add_noise_uniform,
    1: add_noise_poisson,
    2: add_shot,
    3: add_shift,
    4: add_hole,
    5: add_noise_snp,
    6: add_crop,
    7: add_sat,
    8: add_blindspot,
    9: sim_exp
}
num_augments = len(augments)

#apply random augmentation filter to intput image
def random_augmentation(img_data):
    augment = augments.get(np.random.randint(num_augments))
    return augment(img_data)
