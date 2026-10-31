"""Run patch-embedding Conv3d (kernel == stride, no padding, one patch per sample) as the equivalent matmul.

Under torch 2.9.1 / cuDNN 9.10 a bf16 Conv3d over Qwen3-VL's 9408 patches of 4 frames takes ~70 s (cudnn on or off);
F.linear on the flattened patch takes 0.2 ms and agrees to one bf16 ulp. Loaded at interpreter start by
envs/simplestream-qwen3/.../zz_kfs_conv3d_fix.pth (see envs/setup_envs.sh), so it covers our scripts and
SimpleStream's official Qwen3 scripts alike; other convolutions fall through to the original forward.
"""

import torch
import torch.nn.functional as F

_conv3d_forward = torch.nn.Conv3d.forward


def _forward(self, x):
    k = tuple(self.kernel_size)
    if (tuple(self.stride) == k and tuple(self.padding) == (0, 0, 0) and tuple(self.dilation) == (1, 1, 1)
            and self.groups == 1 and x.dim() == 5 and tuple(x.shape[2:]) == k):
        y = F.linear(x.reshape(x.shape[0], -1), self.weight.reshape(self.out_channels, -1), self.bias)
        return y.view(x.shape[0], self.out_channels, 1, 1, 1)
    return _conv3d_forward(self, x)


torch.nn.Conv3d.forward = _forward
