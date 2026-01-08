import math
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fno import TFNO


class MGTFNO(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        levels: int,
        kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        assert levels >= 0

        self.C_in = in_channels
        self.C_out = out_channels
        self.levels = levels

        self.num_sub_axis = 2 ** self.levels
        self.num_tiles = self.num_sub_axis * self.num_sub_axis

        self.multi_C_in = self.C_in * (self.levels + 1)

        base_kwargs = dict(
            in_channels=self.multi_C_in,
            out_channels=self.C_out,
        )
        if kwargs:
            base_kwargs.update(kwargs)

        self.model_tiles = nn.ModuleList([
            TFNO(**base_kwargs) for _ in range(self.num_tiles)
        ])

    @staticmethod
    def _is_power_of_two(n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    def _build_levels_for_tile(
        self,
        x: torch.Tensor,
        s_log2: int,
        sub_size: int,
        i: int,
        j: int,
    ) -> torch.Tensor:
        B, C_in, H, W = x.shape
        assert H == W
        top0 = i * sub_size
        left0 = j * sub_size
        bottom0 = top0 + sub_size
        right0 = left0 + sub_size

        levels: List[torch.Tensor] = []
        for k in range(self.levels + 1):
            size_k = 2 ** (s_log2 - self.levels + k)
            pad_needed = (size_k - sub_size) // 2

            top_k = top0 - pad_needed
            left_k = left0 - pad_needed
            bottom_k = bottom0 + pad_needed
            right_k = right0 + pad_needed

            pad_top = max(0, -top_k)
            pad_left = max(0, -left_k)
            pad_bottom = max(0, bottom_k - H)
            pad_right = max(0, right_k - W)

            top_k = max(top_k, 0)
            left_k = max(left_k, 0)
            bottom_k = min(bottom_k, H)
            right_k = min(right_k, W)

            crop = x[:, :, top_k:bottom_k, left_k:right_k]  # [B, C_in, h_k, w_k]
            if pad_top or pad_left or pad_bottom or pad_right:
                crop = F.pad(crop, (pad_left, pad_right, pad_top, pad_bottom),
                             mode="constant", value=0.0)

            # Downsample
            if k > 0:
                stride = 2 ** k
                crop = crop[:, :, ::stride, ::stride]

            assert crop.shape[2] == sub_size and crop.shape[3] == sub_size, f"level {k} crop shape {crop.shape} != ({sub_size},{sub_size})"

            levels.append(crop)  # [B, C_in, sub, sub]

        return torch.cat(levels, dim=1)  # [B, C_in*(L+1), sub, sub]

    def _tile_inputs(self, x: torch.Tensor):
        B, C_in, H, W = x.shape
        assert H == W, "H==W"
        assert self._is_power_of_two(H), f"H={H} must be power of two."

        s_log2 = int(math.log2(H))
        sub_size = 2 ** (s_log2 - self.levels)
        assert H % sub_size == 0 and W % sub_size == 0

        tiles = []
        for i in range(self.num_sub_axis):
            for j in range(self.num_sub_axis):
                tile_ij = self._build_levels_for_tile(x, s_log2, sub_size, i, j)
                tiles.append(tile_ij)  # [B, multi_C_in, sub, sub]
        return tiles, sub_size, H, W

    def _stitch(self, outputs: List[torch.Tensor], sub_size: int, H: int, W: int):
        """
        outputs: List of [B, C_out, sub, sub], length = num_tiles
        return: y [B, C_out, H, W]
        """
        assert len(outputs) == self.num_tiles
        B = outputs[0].shape[0]
        device = outputs[0].device
        dtype = outputs[0].dtype

        y = torch.zeros((B, self.C_out, H, W), device=device, dtype=dtype)
        idx = 0
        for i in range(self.num_sub_axis):
            for j in range(self.num_sub_axis):
                top = i * sub_size
                left = j * sub_size
                bottom = top + sub_size
                right = left + sub_size
                y[:, :, top:bottom, left:right] = outputs[idx]
                idx += 1
        return y

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        x: [B, C_in, H, W]  →  y: [B, C_out, H, W]
        """
        assert x.dim() == 4, "x must be [B, C_in, H, W]"
        assert x.shape[1] == self.C_in, f"in_channels mismatch: expected {self.C_in}, got {x.shape[1]}"

        # 1) multigrid crop for all tiles
        tiles, sub_size, H, W = self._tile_inputs(x)  # list of [B, multi_C_in, sub, sub]

        # 2) per-tile independent TFNO
        outputs: List[torch.Tensor] = []
        for idx, tile_in in enumerate(tiles):
            out_ij = self.model_tiles[idx](tile_in)  # [B, C_out, sub, sub]
            outputs.append(out_ij)

        # 3) stitch back
        y = self._stitch(outputs, sub_size, H, W)
        return y
