import sys
sys.path.append(".")
sys.path.append('./taming-transformers')
import os
import logging
os.environ['CUDA_VISIBLE_DEVICES']='0, 1'

import torch
import torch.nn as nn
from omegaconf import OmegaConf
import time
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

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
        "--quant_act", action="store_true", 
        help="if to quantize activations when ptq==True"
    )
    parser.add_argument(
        "--weight_bit",
        type=int,
        default=8,
        help="int bit for weight quantization",
    )
    parser.add_argument(
        "--act_bit",
        type=int,
        default=8,
        help="int bit for weight quantization",
    )
    parser.add_argument(
        "--cali_ckpt_path",
        type=str,
        required=True,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--cali_samples_num",
        type=int,
        default=64,
        help="int bit for activation quantization",
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

if __name__ == '__main__':
    
    parser = get_parser()
    args = parser.parse_args()

    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    # fix random seed
    seed_everything(args.seed)

    # Load model:
    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    # setup logger
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = os.path.join(args.logdir, "quant_weight_traj_align", now)
    os.makedirs(logdir)
    args.logdir = logdir
    log_path = os.path.join(logdir, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    device = torch.device("cuda:1")
    diffusion = Diffusion(args, config, device)

    model = Model(config)
    name = "cifar10"

    from ddim.functions.ckpt_util import get_ckpt_path
    ckpt = get_ckpt_path(f"ema_{name}")
    logger.info("Loading checkpoint {}".format(ckpt))
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

    from quant_scripts.ddim.brecq.quant_model import init_quant_model
    print('First run to init model...')
    with torch.no_grad():
        init_quant_model(qnn, data_loader, batch_size=8, type='left_skewed', device=device)

    # Kwargs for weight rounding calibration
    cali_images, cali_t = get_train_samples(data_loader, num_samples=args.cali_samples_num)
    kwargs = dict(cali_x=cali_images, cali_t=cali_t, iters=8000, weight=0.01, asym=True,
                    b_range=(20, 2), warmup=0.2, act_quant=False, opt_mode='traj_align', batch_size=8, device=device)

    pass_block = 0
    # cnt = 0
    def recon_model(model: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        # global cnt
        global pass_block
        for name, module in model.named_children():
            if isinstance(module, (QuantModule)):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    # cnt += 1
                    layer_reconstruction(qnn, module, **kwargs)

            elif isinstance(module, ResnetBlock):
                pass_block -= 1
                if pass_block < 0 :
                    print('Reconstruction for ResBlock {}'.format(name))
                    # cnt += 1
                    block_reconstruction_two_input(qnn, module, **kwargs)
            else:
                recon_model(module)
        
    # Start calibration
    print('Start calibration')
    t1 = time.time()
    recon_model(qnn)
    t2 = time.time()
    print(f"Recon model took {(t2-t1):6f} s")
    qnn.set_quant_state(weight_quant=True, act_quant=False)
    save_dir = "reproduce/cifar10/weight"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # print(f"recon {cnt} times")
    torch.save(qnn.state_dict(), os.path.join(save_dir, 'quantw{}_ldm_brecq.pth'.format(args.weight_bit)))