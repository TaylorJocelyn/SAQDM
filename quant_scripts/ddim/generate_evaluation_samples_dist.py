"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import sys, time
sys.path.append(".")
sys.path.append('./taming-transformers')
import argparse
from PIL import Image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import time
import logging
from torch.cuda import amp
import numpy as np
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
import torch
# from quant_scripts.ddim.brecq.quant_dataset import DiffusionInputDataset, get_calibration_set
from quant_scripts.ddim.brecq.quant_model import QuantModel
from quant_scripts.ddim.brecq.quant_layer import QuantModule
from quant_scripts.ddim.brecq.adaptive_rounding import AdaRoundQuantizer
from ddim.models.diffusion import Model, Diffusion, logger
import torch.nn as nn
# import pytorch_fid

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def get_train_samples(train_loader, num_samples):
    image_data, t_data, y_data = [], [], []
    for (image, t, y) in train_loader:
        image_data.append(image)
        t_data.append(t)
        y_data.append(y)
        if len(image_data) >= num_samples:
            break
    return torch.cat(image_data, dim=0)[:num_samples], torch.cat(t_data, dim=0)[:num_samples], torch.cat(y_data, dim=0)[:num_samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default='configs/ddim/cifar10.yml', help="Path to the config file"
    )
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
    parser.add_argument(
        "--quant_ckpt",
        type=str,
        required=True,
        help="eta used to control the variances of sigma",
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
        "--cali_sample_num",
        type=int,
        default=1024,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--cali_ckpt_path",
        type=str,
        required=True,
        help="int bit for activation quantization",
    )
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--batch_size', default=25, type=int)
    parser.add_argument('--image_size', default=32, type=int)
    parser.add_argument('--image_type', default='fp32')
    parser.add_argument('--local-rank', help="local device id on current node", type=int)
    parser.add_argument('--nproc_per_node', default=2, type=int)
    args = parser.parse_args()
    print(args)

    print('image type: ', args.image_type)

    os.makedirs('evaluate_data/cifar', exist_ok=True)

    # init ddp
    local_rank = args.local_rank
    device = torch.device("cuda", local_rank)
    print('device: ', device)
    
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.nproc_per_node, rank=local_rank)
    rank = torch.distributed.get_rank()
    
    # seed = int(time.time())
    seed = 100
    torch.manual_seed(seed + rank)
    torch.cuda.set_device(local_rank)
    torch.set_grad_enabled(False)

    # Load model:
    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    diffusion = Diffusion(args, config, device)

    model = Model(config)
    name = "cifar10"

    from ddim.functions.ckpt_util import get_ckpt_path
    ckpt = get_ckpt_path(f"ema_{name}")
    logger.info("Loading checkpoint {}".format(ckpt))
    model.load_state_dict(torch.load(ckpt, map_location=device))
    
    # model.to(device)
    model.cuda(rank)
    model.eval()

    # self.sample_fid(model)

    
    if args.image_type != 'fp32':
        from quant_scripts.ddim.brecq.quant_dataset import TrajectoryDataset
        from torch.utils.data import DataLoader
        from quant_scripts.ddim.brecq.quant_dataset import get_train_samples

        dataset = TrajectoryDataset(args.cali_ckpt_path)
        data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)
        
        wq_params = {'n_bits': args.weight_bit, 'channel_wise': False, 'scale_method': 'mse'}
        aq_params = {'n_bits': args.act_bit, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
        qnn = QuantModel(model=model, weight_quant_params=wq_params, act_quant_params=aq_params)
        qnn.cuda(rank)
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
        qnn.cuda(rank)
        qnn.eval()
        model = qnn

    model=nn.parallel.DistributedDataParallel(model.cuda(rank), device_ids=[rank])

    out_path = os.path.join('evaluate_data/cifar', f"{args.image_type}_{args.skip_type}_{args.timesteps}.npz")
    print("out_path ", out_path)
    logging.info("sampling...")

    generated_num = torch.tensor(0, device=device)
    if rank == 0:
        all_images = []
        
        generated_num = torch.tensor(0, device=device)
    dist.barrier()
    dist.broadcast(generated_num, 0)

    batch_size = args.batch_size

    while generated_num.item() < args.num_samples:
        t0 = time.time()
        samples_ddim = diffusion.sample_eval(model.module, batch_size).to(device)
        # print('---- samples_ddim -------')
        # print('samples_ddim shape, ', samples_ddim.shape)
        # print('min ', torch.min(samples_ddim))
        # print('max ', torch.max(samples_ddim))

        # x_samples_ddim = torch.clamp((samples_ddim+1.0)/2.0, 
        #                             min=0.0, max=1.0)

        x_samples_ddim = torch.clamp(samples_ddim, min=0.0, max=1.0)
        
        x_samples_ddim = (x_samples_ddim * 255.0).clamp(0, 255).to(torch.uint8)

        # print('x_samples_ddim min: ', torch.min(x_samples_ddim))
        # print('x_samples_ddim max: ', torch.max(x_samples_ddim))
        x_samples_ddim = x_samples_ddim.permute(0, 2, 3, 1)
        samples = x_samples_ddim.contiguous()

        t1 = time.time()
        print('throughput : {}'.format((t1 - t0) / x_samples_ddim.shape[0]))
        
        # print('world_size: ', dist.get_world_size())
        gathered_samples = [torch.zeros_like(samples) for _ in range(dist.get_world_size())]
        # print('gathered_samples[0] shape: ', gathered_samples[0].shape)
        # print('gathered_samples len: ', len(gathered_samples))
        dist.all_gather(gathered_samples, samples)  

        if rank == 0:
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            # print('all_images len ', len(all_images))
            # print("all_images [0] shape ", all_images[0].shape)
            logging.info(f"created {len(all_images) * batch_size} samples")
            generated_num = torch.tensor(len(all_images) * batch_size, device=device)
            print('generated_num: ', generated_num)
            
        torch.distributed.barrier()
        dist.broadcast(generated_num, 0)

    if rank == 0:
        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]

        print('arr shape: ', arr.shape)
        logging.info(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logging.info("sampling complete")

if __name__ == "__main__":
    main()