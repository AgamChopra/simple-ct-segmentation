import numpy as np
import torch
import pandas as pd
import random
from PIL import Image
from tqdm import trange
from matplotlib import pyplot as plt


ROOT_PATH = 'PATH/TO/YOUR/FOLDER/'


def update_path(root_path = ''):
    global ROOT_PATH
    print('Path changed from %s to'%(ROOT_PATH), end=' ')
    ROOT_PATH = root_path
    print(ROOT_PATH)


def load_datapoint(path):
    data = np.array(Image.open(path).resize((140,140)))
    if len(data.shape) < 3:
        data = np.expand_dims(data, axis=-1)
    return data.T


def load_dataset(lookup):
    raw, mask = [],[]
    
    for i in trange(len(lookup)):
        path = lookup[i]
        raw.append(load_datapoint(ROOT_PATH + 'images/images/' + path[0]))
        mask.append(load_datapoint(ROOT_PATH + 'masks/masks/' + path[1]))
    
    raw, mask = torch.from_numpy(np.asanyarray(raw)), torch.from_numpy(np.asanyarray(mask))
    return raw, mask

    
class dataloader(): #load all the data, convert to torch, randomize
    def __init__(self, batch = 32, post = False):
        csv = pd.read_csv(ROOT_PATH + 'train.csv')
        self.lookup = csv.to_numpy()[:15000]
        self.id = 0
        self.batch = batch
        self.idx = None
        self.Flag = True
        self.post = post
        self.raw, self.mask = load_dataset(self.lookup)
        self.info = {"samples" : len(self.lookup),
                     "batch_size" : self.batch,
                     "data_shape" : '[batch,channel,%d,%d]'%(self.raw.shape[-2],self.raw.shape[-1])}
        
    def randomize(self):
        sample_len = len(self.lookup)
        self.idx = random.sample(range(0, sample_len), sample_len)
    
    def load_batch(self, post = False):        
        if self.Flag: #only runs the first time 
            self.randomize()
            self.Flag = False
            
        max_id = len(self.lookup) - 1
        
        if self.id + self.batch > max_id:         
            if self.id < max_id:
                batch_raw, batch_mask = self.raw[self.idx[self.id:]], self.mask[self.idx[self.id:]]
            elif self.id == max_id:
                batch_raw, batch_mask = self.raw[self.idx[self.id:self.id]], self.mask[self.idx[self.id:self.id]]
            self.id = 0
            self.randomize()
            if self.post:
                print('Dataset re-randomized...')
        else:
            batch_raw, batch_mask = self.raw[self.idx[self.id:self.id + self.batch]], self.mask[self.idx[self.id:self.id + self.batch]]
            self.id += self.batch
            
        return batch_raw, batch_mask
    
    def data_info(self):
        return self.info
    
    
class dataloader_val(): #load all the data, convert to torch, randomize
    def __init__(self, batch = 32, post = False):
        csv = pd.read_csv(ROOT_PATH + 'train.csv')
        self.lookup = csv.to_numpy()[15000:]
        self.id = 0
        self.batch = batch
        self.idx = None
        self.Flag = True
        self.post = post
        self.raw, self.mask = load_dataset(self.lookup)
        self.info = {"samples" : len(self.lookup),
                     "batch_size" : self.batch,
                     "data_shape" : '[batch,channel,%d,%d]'%(self.raw.shape[-2],self.raw.shape[-1])}
        
    def randomize(self):
        sample_len = len(self.lookup)
        self.idx = random.sample(range(0, sample_len), sample_len)
    
    def load_batch(self, post = False):        
        if self.Flag: #only runs the first time 
            self.randomize()
            self.Flag = False
            
        max_id = len(self.lookup) - 1
        
        if self.id + self.batch > max_id:         
            if self.id < max_id:
                batch_raw, batch_mask = self.raw[self.idx[self.id:]], self.mask[self.idx[self.id:]]
            elif self.id == max_id:
                batch_raw, batch_mask = self.raw[self.idx[self.id:self.id]], self.mask[self.idx[self.id:self.id]]
            self.id = 0
            self.randomize()
            if self.post:
                print('Dataset re-randomized...')
        else:
            batch_raw, batch_mask = self.raw[self.idx[self.id:self.id + self.batch]], self.mask[self.idx[self.id:self.id + self.batch]]
            self.id += self.batch
            
        return batch_raw, batch_mask
    
    def data_info(self):
        return self.info
        
    
def test():
    update_path('E:/ML/Ilka_segmentation/ct_segment/')
    dt = dataloader(batch = 32, post=True)
    print(dt.data_info())
    for i in range(1000):
        x,y = dt.load_batch()
        if i % 100 == 0:
            print(x.shape,y.shape)
            print(dt.id)
            plt.imshow(x[0].detach().cpu().numpy().T)
            plt.show()
            plt.imshow(y[0].detach().cpu().numpy().T)
            plt.show()
    
    
if __name__ == '__main__':
    test()