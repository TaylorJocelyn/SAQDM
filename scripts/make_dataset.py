import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

class TrajectoryDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path, map_location='cpu')
        self.xt_list = data['x']
        self.t_list = data['t']

    def __len__(self):
        return self.xt_list[0].shape[0]
    
    def __getitem__(self, index):

        x_traj = []
        for i in range(len(self.t_list)):
            x_traj.append(self.xt_list[i][index])

        return x_traj ,self.t_list

def main():
    trainset = torchvision.datasets.CIFAR10(
    root='./data',      
    train=True,        
    download=True)

    images = []

    for img, _ in trainset:
        img = np.array(img)
        images.append(img)

    images_array = np.array(images)

    save_dir = 'evaluate_data/ref'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savez(os.path.join(save_dir, 'cifar10.npz'), images_array)

def load_data():
    img = np.load('evaluate_data/ref/cifar10.npz')
    
    for i in range(img['arr_0'].shape[0]):
        x = Image.fromarray(img['arr_0'][i].astype(np.uint8))
        x.save('/home/zq/SAQDM/reproduce/cifar10/data/img_{}.png'.format(i+10))

        if i>15:
            break

def load_input_data():
    from torch.utils.data import DataLoader
    dataset = TrajectoryDataset('/home/zq/SAQDM/reproduce/cifar10/data/cifar10_100step_quad_trajectory.pth')
    l = len(dataset)
    data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

    for index in range(l):
        x_traj = []
        for i in range(len(dataset.t_list)):
            x_traj.append(dataset.xt_list[i][index])

    x = data_loader


if __name__ == '__main__':
    # load_input_data()
    import math 
    x = min(16, 2 ** math.floor(math.log2(100)))
    z = np.linspace(0, 100, 16)
    z = np.floor(z[0]) + np.round(z[1:])
    x = z