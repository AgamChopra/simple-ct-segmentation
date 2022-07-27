import torch
import torch.nn as nn
from tqdm import trange
from matplotlib import pyplot as plt
from numpy import min, max

import dataloader
import model

torch.set_printoptions(precision=10)


def norm(x):
    try:
        return (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    except:
        return (x - min(x)) / (max(x) - min(x))


def train(root_path, epochs = 100, lr = 1E-4, batch = 32, device = 'cpu'):
    
    # correct the path
    dataloader.update_path(root_path)
    
    # load the model
    neural_network = model.UNet().to(device)
    
    # load the optimizer and criterion
    optimizer = torch.optim.Adam(neural_network.parameters(),lr)
    criterion = nn.MSELoss()
    
    # load dataloader
    data = dataloader.dataloader(batch=batch)
    
    data_info = data.data_info()
    data_points = data_info['samples']
    
    data_val = dataloader.dataloader_val(batch=batch)
    
    data_val_info = data_val.data_info()
    data_val_points = data_val_info['samples']
    
    
    # how many times to iterate each epoch
    iterations = int(data_points / batch) + (data_points % batch > 0)
    iterations_val = int(data_val_points / batch) + (data_val_points % batch > 0)
    # store training loss for visualization
    losses = []
    losses_train = []
    losses_temp = []
    losses_val = []
    
    for eps in range(epochs):
        print('Epoch %d:'%(eps))
        
        neural_network.train()
        
        for i in trange(iterations):
            optimizer.zero_grad() 
            
            x, y = data.load_batch()
            x, y = norm(x).to(device), torch.where(norm(y) > 0.5, 1., 0.).to(device) # normalize
            
            yp = neural_network(x,device)     
            
            error = criterion(y,yp) 
            error.backward()
            optimizer.step()
            
            losses.append(error.item())
            
        losses_train.append(sum(losses[-iterations:])/iterations)
        
        neural_network.eval()
        
        for i in trange(iterations_val):
            with torch.no_grad():
                x, y = data_val.load_batch()
                x, y = norm(x).to(device), torch.where(norm(y) > 0.5, 1., 0.).to(device) # normalize
                
                yp = neural_network(x,device)     
                
                error = criterion(y,yp) 
                
                losses_temp.append(error.item())
            
        losses_val.append(sum(losses_temp[-iterations_val:])/iterations_val)
        losses_temp = []
            
        if eps % 2 == 0 or eps == epochs - 1:    
            plt.plot(losses_train,label = 'Training Loss')
            plt.plot(losses_val,label = 'Validation Loss')
            plt.title('MSE Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.legend()
            plt.show()
                  
        print(' Average Train Loss: %.4f, Validation Loss: %.4f'%(losses_train[-1],losses_val[-1]))

    
    torch.save(neural_network.cpu().state_dict(), root_path + 'trained_paramaters_%d_epochs.pt'%(epochs))
            
    return neural_network, data, data_val, losses_train, losses_val


def trn(root_path, epochs = 100, lr = 1E-4, batch = 32, device = 'cpu'):  
    _, _, _,losst, lossv = train(root_path, epochs, lr, batch , device)        
    plt.plot(losst,label = 'Training Loss')
    plt.plot(lossv,label = 'Validation Loss')
    plt.title('MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
    
    

def tst(path,file_name,device = 'cpu'):
    dataloader.update_path(path)
    
    batch = 64
    data = dataloader.dataloader_val(batch=batch)
    
    Net = model.UNet()
    Net.load_state_dict(torch.load(path + file_name,map_location='cpu'))
    
    criterion = nn.MSELoss()
    
    error = 0.
    
    samples = data.data_info()['samples']
    
    iterations = int(samples / batch) + (samples % batch > 0)
    
    with torch.no_grad():
        Net = Net.eval().to(device)
        
        for i in trange(iterations):
            x,y = data.load_batch()
            x = norm(x).to(device)
            y = torch.where(norm(y) > 0.5, 1., 0.).to(device)
            yp = torch.where(norm(Net(x,device)) > 0.5, 1., 0.)
            
            error += criterion(y,yp).item()   
            
            for xi,yi,ypi in zip(x,y,yp):
                r,c = 1,3
                fig = plt.figure(figsize=(15,5),dpi=100)
                fig.add_subplot(r,c,1)
                plt.imshow(xi.detach().cpu().numpy().T,cmap='bone')
                plt.title('Input')
                plt.axis('off')
                fig.add_subplot(r,c,2)
                plt.imshow(ypi.detach().cpu().numpy().T)
                plt.title('Predicted')
                plt.axis('off')
                fig.add_subplot(r,c,3)
                plt.imshow(yi.detach().cpu().numpy().T)
                plt.title('Expected')
                plt.axis('off')
                plt.show() 
                break
                #print('exp->',torch.max(yi),torch.min(yi),' pred->',torch.max(ypi),torch.min(ypi))
        
        error /= iterations
        
        print('\nAverage Mean Square Error = ', error)
        
    return error
    
    
if __name__ == '__main__':
    eps = 10
    #trn('E:/ML/Ilka_segmentation/ct_segment/',eps,5E-5,32,'cuda')
    tst('E:/ML/Ilka_segmentation/ct_segment/', 'trained_paramaters_%d_epochs.pt'%(eps),'cuda')