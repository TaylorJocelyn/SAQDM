import argparse, os, gc, glob, datetime, yaml
import logging
import math

import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.cuda import amp
from pytorch_lightning import seed_everything
import sys
sys.path.append('.')
from ddim.models.diffusion import Model, Diffusion
from ddim.datasets import inverse_data_transform
from ddim.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from ddim.functions.ckpt_util import get_ckpt_path
from torch.cuda import amp
import torchvision.utils as tvu

from ddim.models.diffusion import ResnetBlock

from quant_scripts.ddim.brecq.quant_model import QuantModel
from quant_scripts.ddim.brecq.quant_layer import QuantModule
from quant_scripts.ddim.brecq.layer_recon import layer_reconstruction
from quant_scripts.ddim.brecq.block_recon import block_reconstruction_two_input

from tqdm import tqdm
from pytorch_lightning import seed_everything
from ddim.models.diffusion import Model, Diffusion
from ddim.datasets import inverse_data_transform
from ddim.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from ddim.functions.ckpt_util import get_ckpt_path
from torch.cuda import amp
import torchvision.utils as tvu
import argparse
import yaml

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "--cond", action="store_true",
        help="whether to use conditional guidance"
    )
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="quad",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=100, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument(
        "--cali_sample_num",
        type=int,
        default=1024,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--weight_bit",
        type=int,
        default=8,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--act_bit",
        type=int,
        default=8,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--quant_ckpt",
        type=str,
        required=True,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--cali_ckpt_path",
        type=str,
        required=True,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--max_images", type=int, default=50000, help="number of images to sample"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )
    return parser
    
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    # fix random seed
    seed_everything(args.seed)

    device = torch.device('cuda:1') 
    diffusion = Diffusion(args, config, device=device)

    model = Model(config)
    name = 'cifar10'
    ckpt = get_ckpt_path(f"ema_{name}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    
    model.to(device)
    model.eval()

    from quant_scripts.ddim.brecq.quant_dataset import TrajectoryDataset
    from torch.utils.data import DataLoader
    from quant_scripts.ddim.brecq.quant_dataset import get_train_samples

    dataset = TrajectoryDataset(args.cali_ckpt_path)
    data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)
    
    wq_params = {'n_bits': args.weight_bit, 'channel_wise': False, 'scale_method': 'mse'}
    aq_params = {'n_bits': args.act_bit, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
    qnn = QuantModel(model=model, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.to(device)
    qnn.eval()

    print('Setting the first and the last layer to 8-bit')
    qnn.set_first_last_layer_to_8bit()

    device = next(qnn.parameters()).device
    # Initialize weight quantization parameters
    qnn.set_quant_state(True, False)

    qnn.disable_network_output_quantization()

    from quant_scripts.ddim.brecq.quant_model import init_quant_model
    print('First run to init model...')
    with torch.no_grad():
        init_quant_model(qnn, data_loader, batch_size=8, type='left_skewed', device=device)

    # Kwargs for weight rounding calibration
    cali_images, cali_t = get_train_samples(data_loader, num_samples=args.cali_sample_num)
    kwargs = dict(cali_images=cali_images, cali_t=cali_t, iters=8000, weight=0.01, asym=True,
                    b_range=(20, 2), warmup=0.2, act_quant=False, opt_mode='mse', batch_size=8)
        
    # Start calibration
    from quant_scripts.ddim.brecq.adaptive_rounding import AdaRoundQuantizer
    for name, module in qnn.named_modules():
        if isinstance(module, QuantModule) and module.ignore_reconstruction is False:
            module.weight_quantizer.soft_targets = False
            module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode='learned_hard_sigmoid', weight_tensor=module.org_weight.data)

    qnn.load_state_dict(torch.load(args.quant_ckpt))
    qnn.to(device)
    qnn.eval()

    batch_size = 8

    x = diffusion.sample_eval(qnn, batch_size)
    x_samples_ddim = torch.clamp(x, min=0.0, max=1.0)
        
    x_samples_ddim = (x_samples_ddim * 255.0).clamp(0, 255).to(torch.uint8)
    x_samples_ddim = x_samples_ddim.permute(0, 2, 3, 1)
    samples = x_samples_ddim.contiguous()

    save_dir = 'reproduce/cifar10/sample/quant'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    from PIL import Image
    for i in range(samples.shape[0]):
        img = Image.fromarray(np.array(samples[i]))
        img.save(os.path.join(save_dir, "image_{}.png".format(i)))

