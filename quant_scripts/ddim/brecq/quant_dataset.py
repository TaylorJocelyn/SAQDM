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
    
def get_train_samples(train_loader, num_samples):
    image_data, t_data = [], [] 
    steps = 0
    for (image, t) in train_loader:
        image_data.append(image)
        t_data.append(t)
        steps = len(image)

        if len(image_data) * image[0].shape[0] >= num_samples:
            break

    # [shape(bs, c, h, w)]
    x_trajs, t_trajs = [], []
    for i in range(steps):
        batch_trajs = []
        batch_ts = []
        for j in range(len(image_data)):
            batch_trajs.append(image_data[j][i])
            batch_ts.append(t_data[j][i])

        x_trajs.append(torch.cat(batch_trajs, dim=0)[:num_samples])
        t_trajs.append(torch.cat(batch_ts, dim=0)[:num_samples])
    
    return x_trajs, t_trajs



def concat_error(data_path1, data_path2):
    error1 = torch.load(data_path1, map_location='cpu')
    error2 = torch.load(data_path2, map_location='cpu')
    error1.extend(error2)

    return error1


if __name__ == '__main__':
    pass
    