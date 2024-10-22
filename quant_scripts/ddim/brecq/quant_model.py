import torch.nn as nn
# from quantization.brecq_quant_block import specials, BaseQuantBlock
from quant_scripts.ddim.brecq.quant_layer import QuantModule, StraightThrough
from quant_scripts.ddim.brecq.fold_bn import search_fold_and_remove_bn
import numpy as np
import math
import torch

class QuantModel(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        # search_fold_and_remove_bn(model)
        self.model = model
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Linear)) and 'shortcut' not in name and 'op' not in name:  ## keep skip connection full-precision
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, QuantModule):  ## remove BaseQuantBlock
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, image, t ):
        return self.model(image, t)

    def set_first_last_layer_to_8bit(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[0].weight_quantizer.bitwidth_refactor(8)
        module_list[0].act_quantizer.bitwidth_refactor(8)
        # module_list[1].weight_quantizer.bitwidth_refactor(8)
        # module_list[1].act_quantizer.bitwidth_refactor(8)
        module_list[2].weight_quantizer.bitwidth_refactor(8) ## it is a input layer
        module_list[2].act_quantizer.bitwidth_refactor(8)

        module_list[-1].weight_quantizer.bitwidth_refactor(8)
        module_list[-1].act_quantizer.bitwidth_refactor(8)

        # ignore reconstruction of the first layer
        module_list[0].ignore_reconstruction = True
        # module_list[1].ignore_reconstruction = True
        module_list[2].ignore_reconstruction = True
        module_list[-1].ignore_reconstruction = True

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[-1].disable_act_quant = True

def init_quant_model(model, data_loader, batch_size, type, device):
    tot = 64
    step_per_taj = None
    traj_num = 0
    count = 0

    traj_list = []
    t_list = []
    steps = 0
    for (image, t) in data_loader:
        if step_per_taj is None:
            steps = len(image)
            step_per_taj = min(16, 2 ** math.floor(math.log2(len(image))))
            traj_num = tot // step_per_taj

        if count + image[0].shape[0] >= traj_num:
            c = traj_num - count
            traj_list.append([xt[:c] for xt in image]) 
            t_list.append([t_[:c] for t_ in t])  
            break

        traj_list.append(image)
        t_list.append(t)
        count += image[0].shape[0]

    init_trajs = []
    init_t = []
    if type == 'uniform':
        idx = np.linspace(0, steps-1, step_per_taj)
        idx = [int(np.round(num)) if np.round(num) < steps else int(np.floor(num)) for num in idx]

    elif type == 'left_skewed':
        t_mean = 0.6
        t_std = 0.2
        idx = np.random.normal(t_mean, t_std, step_per_taj) * (steps-1)
        idx = np.clip(np.round(idx), 0, steps-1).astype(int)
    else:
        raise NotImplementedError("type not implemented!")
    
    for i in range(len(traj_list)):
        for j in idx:
            init_trajs.append(traj_list[i][j])
            init_t.append(t_list[i][j])

    init_trajs = torch.cat(init_trajs, dim=0)
    init_t = torch.cat(init_t, dim=0)

    _ = model(init_trajs.to(device),init_t.to(device))




