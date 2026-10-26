import torch
import torch.nn as nn
import torch.nn.functional as F

class ReferenceRayKnightOperator(nn.Module):

    def __init__(self, in_channels: int=256, out_channels: int=256):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.conv_rank = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.conv_file = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.ray_dist_weights = nn.Parameter(torch.randn(8, 7) * 0.05)
        self.knight_weights = nn.Parameter(torch.randn(8) * 0.05)
        self.knight_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.conv_out = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.silu(self.conv_in(x))
        ray_ortho = self.conv_rank(z) + self.conv_file(z)
        ray_diag = torch.zeros_like(z)
        for dist in range(1, 8):
            w_ne = self.ray_dist_weights[4, dist - 1]
            w_nw = self.ray_dist_weights[5, dist - 1]
            w_se = self.ray_dist_weights[6, dist - 1]
            w_sw = self.ray_dist_weights[7, dist - 1]
            ne_shift = F.pad(z[:, :, :-dist, :-dist], (dist, 0, dist, 0))
            nw_shift = F.pad(z[:, :, :-dist, dist:], (0, dist, dist, 0))
            se_shift = F.pad(z[:, :, dist:, :-dist], (dist, 0, 0, dist))
            sw_shift = F.pad(z[:, :, dist:, dist:], (0, dist, 0, dist))
            ray_diag = ray_diag + w_ne * ne_shift + w_nw * nw_shift + w_se * se_shift + w_sw * sw_shift
        ray_out = ray_ortho + ray_diag
        knight_shifts = [F.pad(z[:, :, :-2, :-1], (1, 0, 2, 0)), F.pad(z[:, :, :-2, 1:], (0, 1, 2, 0)), F.pad(z[:, :, 2:, :-1], (1, 0, 0, 2)), F.pad(z[:, :, 2:, 1:], (0, 1, 0, 2)), F.pad(z[:, :, :-1, :-2], (2, 0, 1, 0)), F.pad(z[:, :, :-1, 2:], (0, 2, 1, 0)), F.pad(z[:, :, 1:, :-2], (2, 0, 0, 1)), F.pad(z[:, :, 1:, 2:], (0, 2, 0, 1))]
        knight_out = torch.zeros_like(z)
        for i, s in enumerate(knight_shifts):
            knight_out = knight_out + self.knight_weights[i] * s
        knight_out = self.knight_proj(knight_out)
        combined = torch.cat([ray_out, knight_out], dim=1)
        return F.silu(self.conv_out(combined))
