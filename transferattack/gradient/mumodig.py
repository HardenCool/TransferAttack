import torch
import torch.nn.functional as F
import kornia.augmentation as K

from ..utils import *
from .mifgsm import MIFGSM
from ..lb_quantization import LBQuantization


class MUMODIG(MIFGSM):
    """
    MUMODIG Attack
    'Improving Integrated Gradient-based Transferable Adversarial Examples by
    Refining the Integration Path (RYC-98/MuMoDIG)'
    https://github.com/RYC-98/MuMoDIG

    Multi-baseline Monotone DIG: combines a direct IG term (from a
    lower-bound-quantized baseline) with an expectation-over-transforms IG term
    to improve transferability.

    Arguments:
        model_name (str): surrogate model name.
        epsilon (float): L∞ perturbation budget.
        alpha (float): per-step size.
        epoch (int): number of iterations.
        decay (float): momentum decay factor.
        N_trans (int): number of input transformations for the EIG term.
        N_base (int): number of baselines per call.
        N_intepolate (int): number of interpolation steps along each integration path.
        region_num (int): quantization bins for LBQ baseline.
        lamb (float): fractional offset inside each interpolation interval.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): random δ initialisation.
        norm (str): 'linfty' or 'l2'.
        loss (str): loss function name.
        device (torch.device): computation device.

    Official arguments:
        epsilon=16/255, alpha=1.6/255, epoch=10, decay=1.,
        N_trans=6, N_base=1, N_intepolate=1, region_num=2, lamb=0.65

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/mumodig/resnet18 --attack mumodig --model=resnet18
        python main.py --input_dir ./path/to/data --output_dir adv_data/mumodig/resnet18 --eval
    """

    def __init__(
        self,
        model_name,
        epsilon=16 / 255,
        alpha=1.6 / 255,
        epoch=10,
        decay=1.0,
        N_trans=6,
        N_base=1,
        N_intepolate=1,
        region_num=2,
        lamb=0.65,
        targeted=False,
        random_start=False,
        norm='linfty',
        loss='crossentropy',
        device=None,
        attack='MUMODIG',
        **kwargs,
    ):
        super().__init__(
            model_name, epsilon, alpha, epoch, decay,
            targeted, random_start, norm, loss, device, attack,
        )
        self.N_trans = N_trans
        self.N_base = N_base
        self.N_intepolate = N_intepolate
        self.quant = LBQuantization(region_num)
        self.lamb = lamb

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
        momentum = 0

        for _ in range(self.epoch):
            sole_grad = self._ig(data, delta, label)
            exp_grad = self._exp_ig(data, delta, label)
            ig_grad = sole_grad + exp_grad

            momentum = self.get_momentum(ig_grad, momentum)
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()

    # ------------------------------------------------------------------
    # IG helpers
    # ------------------------------------------------------------------

    def _ig(self, data, delta, label):
        """Direct IG from a lower-bound-quantized baseline."""
        ig = 0
        for i_base in range(self.N_base):
            baseline = self.quant(data + delta).clone().detach().to(self.device)
            path = data + delta - baseline
            acc_grad = 0
            for i_inter in range(self.N_intepolate):
                x_interp = baseline + (i_inter + self.lamb) * path / self.N_intepolate
                logits = self.get_logits(x_interp)
                loss = self.get_loss(logits, label)
                is_last = (i_base + 1 == self.N_base) and (i_inter + 1 == self.N_intepolate)
                grad = self.get_grad(loss, delta) if is_last else self._get_repeat_grad(loss, delta)
                acc_grad += grad
            ig += acc_grad * path
        return ig

    def _exp_ig(self, data, delta, label):
        """Expectation-over-transforms IG."""
        ig = 0
        for _ in range(self.N_trans):
            x_trans = self._select_transform(data + delta)
            for i_base in range(self.N_base):
                baseline = self.quant(x_trans).clone().detach().to(self.device)
                path = x_trans - baseline
                acc_grad = 0
                for i_inter in range(self.N_intepolate):
                    x_interp = baseline + (i_inter + self.lamb) / self.N_intepolate * path
                    logits = self.get_logits(x_interp)
                    loss = self.get_loss(logits, label)
                    is_last = (i_base + 1 == self.N_base) and (i_inter + 1 == self.N_intepolate)
                    grad = self.get_grad(loss, delta) if is_last else self._get_repeat_grad(loss, delta)
                    acc_grad += grad
                ig += acc_grad * path
        return ig

    def _get_repeat_grad(self, loss, delta):
        """Gradient that keeps the graph alive for subsequent uses."""
        return torch.autograd.grad(loss, delta, retain_graph=True, create_graph=False)[0]

    # ------------------------------------------------------------------
    # Spatial transformations
    # ------------------------------------------------------------------

    def _vertical_shift(self, x):
        step = np.random.randint(low=0, high=x.shape[2], dtype=np.int32)
        return x.roll(step, dims=2)

    def _horizontal_shift(self, x):
        step = np.random.randint(low=0, high=x.shape[3], dtype=np.int32)
        return x.roll(step, dims=3)

    def _vertical_flip(self, x):
        return x.flip(dims=(2,))

    def _horizontal_flip(self, x):
        return x.flip(dims=(3,))

    def _random_rotate(self, x):
        return K.RandomRotation(p=1, degrees=45)(x)

    def _random_affine(self, x):
        ops = [
            self._vertical_shift,
            self._horizontal_shift,
            self._vertical_flip,
            self._horizontal_flip,
            self._random_rotate,
        ]
        return ops[torch.randint(0, len(ops), [1]).item()](x)

    def _random_resize_and_pad(self, x, img_large_size=245):
        img_inter_size = torch.randint(
            low=min(x.shape[-1], img_large_size),
            high=max(x.shape[-1], img_large_size),
            size=(1,),
            dtype=torch.int32,
        )
        img_inter = F.interpolate(
            x, size=[img_inter_size, img_inter_size], mode='bilinear', align_corners=False
        )
        res_space = img_large_size - img_inter_size
        res_top = torch.randint(low=0, high=res_space.item(), size=(1,), dtype=torch.int32)
        res_bottom = res_space - res_top
        res_left = torch.randint(low=0, high=res_space.item(), size=(1,), dtype=torch.int32)
        res_right = res_space - res_left
        padded = F.pad(
            img_inter,
            [res_left.item(), res_right.item(), res_top.item(), res_bottom.item()],
            value=0,
        )
        return F.interpolate(padded, size=[x.shape[-1], x.shape[-1]], mode='bilinear', align_corners=False)

    def _select_transform(self, x):
        ops = [self._random_affine, self._random_resize_and_pad]
        return ops[torch.randint(0, len(ops), [1]).item()](x)
