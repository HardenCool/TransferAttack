import random

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TFF

from ..utils import *
from ..attack import Attack


# ---------------------------------------------------------------------------
# Primitive spatial operators
# ---------------------------------------------------------------------------

def _vertical_shift(x):
    step = np.random.randint(low=0, high=x.shape[2], dtype=np.int32)
    return x.roll(step, dims=2)

def _horizontal_shift(x):
    step = np.random.randint(low=0, high=x.shape[3], dtype=np.int32)
    return x.roll(step, dims=3)

def _vertical_flip(x):
    return x.flip(dims=(2,))

def _horizontal_flip(x):
    return x.flip(dims=(3,))

def _identity(x):
    return x

class _Scaling:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        return x / self.scale

class _DIM:
    """Mini-DIM operator used inside OPS operator sampling."""
    def __init__(self, resize_rate=1.1):
        self.resize_rate = resize_rate

    def __call__(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)
        rnd = torch.randint(
            low=min(img_size, img_resize),
            high=max(img_size, img_resize),
            size=(1,),
            dtype=torch.int32,
        )
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left
        padded = F.pad(
            rescaled,
            [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()],
            value=0,
        )
        return F.interpolate(padded, size=[img_size, img_size], mode='bilinear', align_corners=False)

class _Rotate:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TFF.rotate(img=x, angle=self.angle)


# ---------------------------------------------------------------------------
# OPS attack
# ---------------------------------------------------------------------------

class OPS(Attack):
    """
    OPS Attack
    'Operator-Perturbation-based Stochastic optimization'
    https://github.com/the-full/OPS  (the-full/OPS)

    Improves transferability by averaging gradients across a neighbourhood of
    perturbed inputs (perturbation sampling) and a pool of randomly-composed
    spatial operators (operator sampling), then updating with MI-FGSM momentum.

    Arguments:
        model_name (str): surrogate model name.
        epsilon (float): L∞ perturbation budget.
        beta (float): neighbourhood radius scale factor (radius = beta * epsilon * ratio).
        num_iter (int): number of attack iterations.
        num_sample_neighbor (int): neighbours sampled per iteration (0 = off).
        num_sample_operator (int): operators sampled per iteration (0 = off).
        sample_levels (iterable): composition depths for operator sampling.
        sample_ratios (iterable): per-sample radius multipliers.
        decay (float): momentum decay factor.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): random δ initialisation.
        norm (str): 'linfty' or 'l2'.
        loss (str): loss function name.
        device (torch.device): computation device.

    Official arguments:
        epsilon=16/255, beta=2., num_iter=10,
        num_sample_neighbor=10, num_sample_operator=20,
        sample_levels=range(2,5), sample_ratios=np.arange(0.,1.5,0.25)+0.25

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/ops/resnet18 --attack ops --model=resnet18
        python main.py --input_dir ./path/to/data --output_dir adv_data/ops/resnet18 --eval
    """

    def __init__(
        self,
        model_name,
        epsilon=16 / 255,
        beta=2.0,
        num_iter=10,
        num_sample_neighbor=10,
        num_sample_operator=20,
        sample_levels=range(2, 5),
        sample_ratios=np.arange(0.0, 1.5, 0.25) + 0.25,
        decay=1.0,
        targeted=False,
        random_start=False,
        norm='linfty',
        loss='crossentropy',
        device=None,
        attack='OPS',
        **kwargs,
    ):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = epsilon / num_iter
        self.epoch = num_iter
        self.decay = decay

        self.using_sampling = (num_sample_operator * num_sample_neighbor > 0)

        if self.using_sampling:
            self.num_sample_operator = num_sample_operator
            self.basic_ops = [
                _identity,
                _vertical_flip, _horizontal_flip,
                _vertical_shift, _horizontal_shift,
                _Rotate(5), _Rotate(-5), _Rotate(15), _Rotate(-15),
                _Rotate(45), _Rotate(-45), _Rotate(90), _Rotate(-90), _Rotate(180),
                _Scaling(2), _Scaling(3), _Scaling(4), _Scaling(5), _Scaling(6),
                _Scaling(7), _Scaling(8),
                _DIM(1.1), _DIM(1.3), _DIM(1.5), _DIM(1.7), _DIM(1.9),
                _DIM(2.1), _DIM(2.3), _DIM(2.5), _DIM(2.7), _DIM(2.9),
            ]
            self.sample_levels = list(sample_levels)
            self.op_list = []
            self.num_extra_ops = len(self.basic_ops)

            self.num_sample_neighbor = num_sample_neighbor
            self.sample_radius = beta * epsilon * np.asarray(list(sample_ratios))
            self.eps_list = []
            self.num_extra_eps = self.num_sample_neighbor

    # ------------------------------------------------------------------
    # Operator pool helpers
    # ------------------------------------------------------------------

    @property
    def op_num(self):
        return len(self.op_list)

    def _get_new_op(self, k=2):
        sel_ops = random.choices(self.basic_ops, k=k)
        def composed_op(x):
            for op in reversed(sel_ops):
                x = op(x)
            return x
        return composed_op

    def _expand_op_list(self, k=2):
        for _ in range(self.num_extra_ops):
            self.op_list.append(self._get_new_op(k=k))

    def _init_op_list(self):
        self.op_list = []
        for level in self.sample_levels:
            if level == 1:
                self.op_list.extend(self.basic_ops)
            else:
                self._expand_op_list(level)

    # ------------------------------------------------------------------
    # Perturbation neighbourhood helpers
    # ------------------------------------------------------------------

    @property
    def eps_num(self):
        return len(self.eps_list)

    def _expand_eps_list(self, delta, radius=1.0):
        shape = (self.num_extra_eps, *delta.shape[1:])
        noise = torch.zeros(shape).uniform_(-radius, radius).to(self.device)
        self.eps_list.extend(noise)

    def _init_eps_list(self, delta):
        self.eps_list = []
        for radius in self.sample_radius:
            self._expand_eps_list(delta, radius)

    # ------------------------------------------------------------------
    # Gradient estimation
    # ------------------------------------------------------------------

    def _get_surrogate_gradient(self, data, delta, label):
        logits = self.get_logits(data + delta)
        loss = self.get_loss(logits, label)
        return self.get_grad(loss, delta)

    def _get_averaged_gradient(self, data, delta, label):
        averaged_gradient = self._get_surrogate_gradient(data, delta, label)
        if not self.using_sampling:
            return averaged_gradient

        selected_eps = random.sample(self.eps_list, min(self.num_sample_neighbor, self.eps_num))
        for eps in selected_eps:
            x_near = data + delta + eps
            self._init_op_list()
            selected_ops = random.sample(self.op_list, min(self.num_sample_operator, self.op_num))
            for op in selected_ops:
                logits = self.get_logits(op(x_near))
                loss = self.get_loss(logits, label)
                grad = self.get_grad(loss, delta)
                averaged_gradient += grad

        return averaged_gradient / (self.num_sample_neighbor * self.num_sample_operator + 1)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        delta = self.init_delta(data)
        if self.using_sampling:
            self._init_eps_list(delta)

        momentum = 0
        for _ in range(self.epoch):
            avg_grad = self._get_averaged_gradient(data, delta, label)
            momentum = self.get_momentum(avg_grad, momentum)
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()
