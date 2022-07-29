import numpy as np
import torch
from torchvision.transforms import functional as tvf
import pandas as pd
import random
from PIL import Image
from tqdm import trange
from matplotlib import pyplot as plt


ROOT_PATH = 'PATH/TO/YOUR/FOLDER/'


def rand_augment(xi, yi):
    if random.random() > 0.5:
        rotation = random.randint(0, 360)
        xi = tvf.rotate(xi, angle = rotation)
        yi = tvf.rotate(yi, angle = rotation)
        
    if random.random() > 0.5:
        xi = tvf.hflip(xi)
        yi = tvf.hflip(yi)
        
    if random.random() > 0.5:
        xi = tvf.vflip(xi)
        yi = tvf.vflip(yi)
    
    return xi, yi


def augment_batch(x, y, p=0.5):
    new_x = []
    new_y = []
    
    xshape = x.shape
    yshape = y.shape
    
    xtemp = torch.zeros(xshape, dtype=torch.int16)
    ytemp = torch.zeros(yshape, dtype=torch.int16)
    
    for i in range(x.shape[0]):
        if random.random() > p:
            x_, y_ = rand_augment(x[i],y[i])
            new_x.append(x_)
            new_y.append(y_)
            
        else:
            new_x.append(x[i])
            new_y.append(y[i])
            
    x = torch.cat(new_x,dim=0,out=xtemp).view(xshape)
    y = torch.cat(new_y,dim=0,out=ytemp).view(yshape)

    return x, y


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
    def __init__(self, batch = 32, post = False, augment = True):
        csv = pd.read_csv(ROOT_PATH + 'train.csv')
        self.augment = augment
        self.lookup = csv.to_numpy()[:15000]
        self.id = 0
        self.batch = batch
        self.idx = None
        self.Flag = True
        self.post = post
        self.raw, self.mask = load_dataset(self.lookup)
        self.info = {"samples" : len(self.lookup),
                     "batch_size" : self.batch,
                     "data_shape" : '[batch,channel,%d,%d]'%(self.raw.shape[-2],self.raw.shape[-1]),
                     "augment" : self.augment}
        
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
                    
        if self.augment:
            batch_raw, batch_mask = augment_batch(batch_raw, batch_mask)
        
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