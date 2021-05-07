import pandas as pd
import os
from tensorflow import keras

from scatter_utils import *
from scatter_augmentations import *

#keras generator class for reading in files of scattering patterns and object tensors from the harddrive. 
#object tensors can be given as both flattened png images as well as pre-processed *.npy files.
#parameters:
#set_df: pandas dataframe containing columns with paths to both scattering patterns and object tensors relative to the "path" parameter
#path: defines the root path of the dataset
#x_col: column name of the scattering patterns
#y_col: column name of the real space tensors
#batch_size: size of the batches returned
#shuffle: if True, the sequence of the dataset is shuffled before each epoch
#augment: how to augment each scattering pattern.
    #False - no augmentation applied
    #True - apply random augmentation
    #'exp' - always apply the sim_exp filter which gives the closest match to experimental scattering pattern
#x_size: image dimensions of the scattering patterns
#y_size: volumne dimensions of the object tensors
#real_npy: switches the routine how object tensors are read between raw *.png files (False) and pre-processed *.npy files. Has to match the paths given in set_df[y_col]

class ScatterDensityGenerator(keras.utils.Sequence):
    #initialize Generator
    def __init__(self, set_df, path='./', x_col='scatter', y_col='real', batch_size=32, shuffle=False, augment=False, x_size=(128,128),y_size=(64,64,64), real_npy=False):
        self.dataset_df = set_df.copy()
        self.datapath = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.x_col = x_col
        self.y_col = y_col
        self.x_size = x_size
        self.y_size = y_size
        self.real_npy = real_npy
        
        #check if files exist: delete missing files from df
        self.dataset_df = self.dataset_df.iloc[[i for i in self.dataset_df.index if (os.path.exists(self.datapath+self.dataset_df[x_col][i]) and os.path.exists(self.datapath+self.dataset_df[y_col][i]))]]
        self.index_array = np.array(self.dataset_df.index)
        self.n = len(self.dataset_df)
        
        if len(self.dataset_df) != len(set_df):
            print('WARNING: file not found in ' + str(len(set_df) - len(self.dataset_df)) + ' cases!')
        

        print("found " + str(self.n) + " image pairs")
        self.on_epoch_end()
        

        
        
    #return number of batches in dataset
    def __len__(self):
        return int(np.floor(len(self.index_array) / self.batch_size))
    
    
    #on epoch end: re-shuffle dataset
    def on_epoch_end(self):
        self.index_array = np.array(self.dataset_df.index)
        if self.shuffle == True:
            np.random.shuffle(self.index_array)
        #self.classes = np.array(self.dataset_df[self.y_col][self.index_array].map(self.class_indices))
    
    
    
    #return single batch at index position
    def batch_gen(self, index):
        batch_indexes = self.index_array[index*self.batch_size:(index+1)*self.batch_size]
        
        x_batch = np.empty((self.batch_size, *self.x_size, 1))
        y_batch = np.empty((self.batch_size, *self.y_size, 1))
        
        for i, ind in enumerate(batch_indexes):
            x_batch[i] = self.augment_image(self.read_scatter(self.datapath + self.dataset_df[self.x_col].iloc[ind]))
            y_batch[i] = self.read_real(self.datapath + self.dataset_df[self.y_col].iloc[ind])
        
        return x_batch, y_batch
    
    def __getitem__(self, index):
        return self.batch_gen(index)
    
    def augment_image(self, img_tens):
        if self.augment:
            if self.augment == 'exp':
                return sim_exp(img_tens)
            else:
                return random_augmentation(img_tens)
        else:
            return img_tens
        
    def read_scatter(self,impath):
        return (imageio.imread(impath)[:,:,1]/255).reshape(self.x_size[0],self.x_size[1],1)

    def read_real(self,impath):
        if self.real_npy:
            return np.load(impath).astype('float32')
        else:
            return transform_density(read_density(impath), self.y_size)