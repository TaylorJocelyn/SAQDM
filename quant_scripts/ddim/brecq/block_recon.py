import torch
from quant_scripts.ddim.brecq.quant_layer import QuantModule, StraightThrough, lp_loss
from quant_scripts.ddim.brecq.quant_model import QuantModel
from quant_scripts.ddim.brecq.adaptive_rounding import AdaRoundQuantizer
from quant_scripts.ddim.brecq.data_utils import save_grad_data, save_inp_oup_data_block, save_inp_oup_data

import torch.nn.functional as F

def block_reconstruction_two_input(model: QuantModel, block, cali_x, cali_t,
                         batch_size: int = 16, iters: int = 20000, weight: float = 0.01, opt_mode: str = 'mse',
                         asym: bool = False, include_act_func: bool = True, b_range: tuple = (20, 2),
                         warmup: float = 0.0, act_quant: bool = False, lr: float = 4e-5, p: float = 2.0,
                         multi_gpu: bool = False, device: torch.device = torch.device('cuda:0')):
    """
    Block reconstruction to optimize the output from each block.

    :param model: QuantModel
    :param block: BaseQuantBlock that needs to be optimized
    :param cali_data: data for calibration, typically 1024 training images, as described in AdaRound
    :param batch_size: mini-batch size for reconstruction
    :param iters: optimization iterations for reconstruction,
    :param weight: the weight of rounding regularization term
    :param opt_mode: optimization mode
    :param asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
    :param include_act_func: optimize the output after activation function
    :param b_range: temperature range
    :param warmup: proportion of iterations that no scheduling for temperature
    :param act_quant: use activation quantization or not.
    :param lr: learning rate for act delta learning
    :param p: L_p norm minimization
    :param multi_gpu: use multi-GPU or not, if enabled, we should sync the gradients
    """
    model.set_quant_state(False, False)
    round_mode = 'learned_hard_sigmoid'
    for m in block.modules():
        if isinstance(m, QuantModule):
            m.set_quant_state(True, act_quant)

    if not include_act_func:
        org_act_func = block.activation_function
        block.activation_function = StraightThrough()

    if not act_quant:
        # Replace weight quantizer to AdaRoundQuantizer
        for name, module in block.named_modules():
            if isinstance(module, QuantModule):
                module.set_quant_state(True, act_quant)
                module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode=round_mode,
                                                            weight_tensor=module.org_weight.data)
                module.weight_quantizer.soft_targets = True

        # Set up optimizer
        opt_params = []
        for name, module in block.named_modules():
            if isinstance(module, QuantModule):
                opt_params += [module.weight_quantizer.alpha]
        optimizer = torch.optim.Adam(opt_params)
        scheduler = None
    else:
        # Use UniformAffineQuantizer to learn delta
        opt_params = []
        for name, module in block.named_modules():
            if isinstance(module, QuantModule):
                if module.act_quantizer.delta is not None:
                    opt_params += [module.act_quantizer.delta]
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)

    loss_mode = 'none' if act_quant else 'relaxation'
    rec_loss = opt_mode

    loss_func = LossFunction(block, round_loss=loss_mode, weight=weight, max_count=iters, rec_loss=rec_loss,
                             b_range=b_range, decay_start=0, warmup=warmup, p=p, device=device)

    # Save data before optimizing the rounding
    cached_inps0_trajs, cached_inps1_trajs, cached_outs_trajs, cached_grads_trajs = [], [], [], []
    time_steps = len(cali_x)
    cali_samples_num = cali_x[0].shape[0]
    for i in range(time_steps):
        cached_inps0, cached_inps1, cached_outs = save_inp_oup_data_block(model, block, cali_x[i], cali_t[i], asym, act_quant, batch_size)
        cached_inps0_trajs.append(cached_inps0)
        cached_inps1_trajs.append(cached_inps1)
        cached_outs_trajs.append(cached_outs)
    
    if opt_mode != 'mse' and opt_mode != 'traj_align':
        for i in range(time_steps): 
            cached_grads = save_grad_data(model, block, cali_x[i], cali_t[i], act_quant, batch_size=batch_size)
            cached_grads_trajs.append(cached_grads)
    else:
        cached_grads = None
    for i in range(iters):
        optimizer.zero_grad()
        idx = torch.randperm(cali_samples_num)[:batch_size]
        cur_out_trajs, cur_grad_trajs, out_quant_trajs = [], [], []
        for j in range(time_steps):
            cur_inp0 = cached_inps0_trajs[j][idx].to(device)
            cur_inp1 = cached_inps1_trajs[j][idx].to(device)
            cur_out = cached_outs_trajs[j][idx].to(device)
            cur_grad = cached_grads_trajs[j][idx].to(device) if opt_mode != 'mse' and opt_mode != 'traj_align' else None
            out_quant = block(cur_inp0, cur_inp1)

            cur_out_trajs.append(cur_out)
            out_quant_trajs.append(out_quant)
            cur_grad_trajs.append(cur_grad)

        err = loss_func(out_quant, cur_out, cur_grad)
        err.backward(retain_graph=True)
        if multi_gpu:
            for p in opt_params:
                link.allreduce(p.grad)
        optimizer.step()
        if scheduler:
            scheduler.step()

    torch.cuda.empty_cache()

    # Finish optimization, use hard rounding.
    for name, module in block.named_modules():
        if isinstance(module, QuantModule):
            module.weight_quantizer.soft_targets = False

    # Reset original activation function
    if not include_act_func:
        block.activation_function = org_act_func


# def block_reconstruction_single_input(model: QuantModel, block, cali_images:torch.Tensor, cali_t:torch.Tensor,
#                          batch_size: int = 8, iters: int = 20000, weight: float = 0.01, opt_mode: str = 'mse',
#                          asym: bool = False, include_act_func: bool = True, b_range: tuple = (20, 2),
#                          warmup: float = 0.0, act_quant: bool = False, lr: float = 4e-5, p: float = 2.0,
#                          multi_gpu: bool = False):
#     """
#     Block reconstruction to optimize the output from each block.

#     :param model: QuantModel
#     :param block: BaseQuantBlock that needs to be optimized
#     :param cali_data: data for calibration, typically 1024 training images, as described in AdaRound
#     :param batch_size: mini-batch size for reconstruction
#     :param iters: optimization iterations for reconstruction,
#     :param weight: the weight of rounding regularization term
#     :param opt_mode: optimization mode
#     :param asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
#     :param include_act_func: optimize the output after activation function
#     :param b_range: temperature range
#     :param warmup: proportion of iterations that no scheduling for temperature
#     :param act_quant: use activation quantization or not.
#     :param lr: learning rate for act delta learning
#     :param p: L_p norm minimization
#     :param multi_gpu: use multi-GPU or not, if enabled, we should sync the gradients
#     """
#     model.set_quant_state(False, False)
#     round_mode = 'learned_hard_sigmoid'
#     for m in block.modules():
#         if isinstance(m, QuantModule):
#             m.set_quant_state(True, act_quant)
            
#     if not include_act_func:
#         org_act_func = block.activation_function
#         block.activation_function = StraightThrough()

#     if not act_quant:
#         # Replace weight quantizer to AdaRoundQuantizer
#         for name, module in block.named_modules():
#             if isinstance(module, QuantModule):
#                 module.set_quant_state(True, act_quant)
#                 module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode=round_mode,
#                                                             weight_tensor=module.org_weight.data)
#                 module.weight_quantizer.soft_targets = True

#         # Set up optimizer
#         opt_params = []
#         for name, module in block.named_modules():
#             if isinstance(module, QuantModule):
#                 opt_params += [module.weight_quantizer.alpha]
#         optimizer = torch.optim.Adam(opt_params)
#         scheduler = None
#     else:
#         # Use UniformAffineQuantizer to learn delta
#         if hasattr(block.act_quantizer, 'delta'):
#             opt_params = [block.act_quantizer.delta]
#         else:
#             opt_params = []
#         for name, module in block.named_modules():
#             if isinstance(module, QuantModule):
#                 if module.act_quantizer.delta is not None:
#                     opt_params += [module.act_quantizer.delta]
#         optimizer = torch.optim.Adam(opt_params, lr=lr)
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)

#     loss_mode = 'none' if act_quant else 'relaxation'
#     rec_loss = opt_mode

#     loss_func = LossFunction(block, round_loss=loss_mode, weight=weight, max_count=iters, rec_loss=rec_loss,
#                              b_range=b_range, decay_start=0, warmup=warmup, p=p)

#     # Save data before optimizing the rounding
#     cached_inps, cached_outs = save_inp_oup_data(model, block, cali_images, cali_t, asym, act_quant, batch_size)
#     if opt_mode != 'mse':
#         cached_grads = save_grad_data(model, block, cali_images, cali_t, act_quant, batch_size=batch_size)
#     else:
#         cached_grads = None
#     device = 'cuda'
#     for i in range(iters):
#         idx = torch.randperm(cached_inps.size(0))[:batch_size]
#         cur_inp = cached_inps[idx].to(device)
#         cur_out = cached_outs[idx].to(device)
#         cur_grad = cached_grads[idx].to(device) if opt_mode != 'mse' else None

#         optimizer.zero_grad()
#         out_quant = block(cur_inp)

#         err = loss_func(out_quant, cur_out, cur_grad)
#         err.backward(retain_graph=True)
#         if multi_gpu:
#             for p in opt_params:
#                 link.allreduce(p.grad)
#         optimizer.step()
#         if scheduler:
#             scheduler.step()

#     torch.cuda.empty_cache()

#     # Finish optimization, use hard rounding.
#     for name, module in block.named_modules():
#         if isinstance(module, QuantModule):
#             module.weight_quantizer.soft_targets = False

#     # Reset original activation function
#     if not include_act_func:
#         block.activation_function = org_act_func

class LossFunction:
    def __init__(self,
                 block,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.,
                 device: torch.device = torch.device('cuda:0')):

        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.device = device

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred_trajs, tgt_trajs, grad_trajs=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred_trajs, tgt_trajs, p=self.p)
        elif self.rec_loss == 'fisher_diag':
            rec_loss = torch.tensor(0.0, device=self.device)
            for i in range(len(pred_trajs)):
                rec_loss += ((pred_trajs[i] - tgt_trajs[i]).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == 'fisher_full':
            rec_loss = torch.tensor(0.0, device=self.device)
            for i in range(len(pred_trajs)):
                a = (pred_trajs[i] - tgt_trajs[i]).abs()
                grad = grad_trajs[i].abs()
                batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
                rec_loss += (batch_dotprod * a * grad).mean() / 100
        elif self.rec_loss == 'traj_align':
            rec_loss = torch.tensor(0.0, device=self.device)
            alpha1, alpha2 = 0.6, 0.4 
            for i in range(len(pred_trajs)):
                mse_loss = (pred_trajs[i]-tgt_trajs[i]).abs().pow(self.p).sum(1).mean()
                cosine_sim = F.cosine_similarity(pred_trajs[i], tgt_trajs[i], dim=1)
                cos_loss = 1 - cosine_sim.mean()
                rec_loss += alpha1 * mse_loss + alpha2 * cos_loss
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            for name, module in self.block.named_modules():
                if isinstance(module, QuantModule):
                    round_vals = module.weight_quantizer.get_soft_targets()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count % 400 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                  float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
