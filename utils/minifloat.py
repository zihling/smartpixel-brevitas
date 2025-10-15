from typing import Optional

import array_api_compat
import brevitas.nn as qnn
import torch.nn as nn
from brevitas.inject import ExtendedInjector
from brevitas.quant.experimental.float_base import (
    FloatActBase,
    FloatBase,
    FloatWeightBase,
)
from numpy import ndarray
from torch import FloatTensor, IntTensor

__all__ = ["mf_to_raw", "raw_to_mf", "fp_mixin_factory", "MinifloatQuantizer"]


# Helper for converting quantized floats to raw binary
def mf_to_raw(
    x: FloatTensor | ndarray,
    exponent_bit_width: int,
    mantissa_bit_width: int,
    exponent_bias: int,
    eps: float,
) -> IntTensor | ndarray:
    xp = array_api_compat.array_namespace(x)

    emin = -exponent_bias + 1

    sign = xp.astype(xp.signbit(x), int)
    exp = xp.astype(xp.floor(xp.log2(xp.abs(x) + eps)), int)
    exp = xp.clip(exp, min=emin)
    man = xp.abs(x) / xp.exp2(exp)

    exp_bits = exp - xp.astype((man < 1), int) + exponent_bias  # Denorm
    man_bits = xp.astype((man * (1 << mantissa_bit_width)), int)
    man_bits = man_bits & ((1 << mantissa_bit_width) - 1)

    out = (
        (sign << (exponent_bit_width + mantissa_bit_width))
        | (exp_bits << mantissa_bit_width)
        | man_bits
    )

    format_width = 1 + exponent_bit_width + mantissa_bit_width
    # Cast to unsigned int for interfacing w/ Arbolta
    out_dtype = (
        xp.uint8
        if format_width <= 8
        else xp.uint16
        if format_width <= 16
        else xp.uint32
        if format_width <= 32
        else xp.uint64
    )

    return xp.astype(out, out_dtype)


# Helper for converting raw binary minifloats to floats
def raw_to_mf(
    x: IntTensor | ndarray,
    exponent_bit_width: int,
    mantissa_bit_width: int,
    exponent_bias: int,
) -> FloatTensor | ndarray:
    xp = array_api_compat.array_namespace(x)
    x = xp.astype(x, int)  # Cast to signed int to allow for bit shifting

    emin = -exponent_bias + 1

    sign_mask = 1 << (exponent_bit_width + mantissa_bit_width)
    exp_mask = ((1 << exponent_bit_width) - 1) << mantissa_bit_width
    man_mask = (1 << mantissa_bit_width) - 1

    sign = xp.where((x & sign_mask) == 0, 1.0, -1.0)
    exp_denorm = xp.astype((x & exp_mask) >> mantissa_bit_width, xp.float32)
    man_denorm = xp.astype(x & man_mask, xp.float32) / (2**mantissa_bit_width)

    exp = xp.where(exp_denorm == 0, emin, exp_denorm - exponent_bias)
    man = xp.where(exp_denorm == 0, man_denorm, 1.0 + man_denorm)

    return sign * man * xp.exp2(exp)


# Helper for creating custom minifloat quantizers
def fp_mixin_factory(
    exponent_bit_width: int, mantissa_bit_width: int, base_class: FloatBase
):
    bit_width = 1 + exponent_bit_width + mantissa_bit_width
    name = f"Fp{bit_width}e{exponent_bit_width}m{mantissa_bit_width}"
    mixin = type(
        name + "Mixin",
        (ExtendedInjector,),
        {
            # Add sign bit
            "bit_width": bit_width,
            "exponent_bit_width": exponent_bit_width,
            "mantissa_bit_width": mantissa_bit_width,
            "saturating": True,
        },
    )

    if base_class is FloatActBase:
        class_name = name + "Act"
    elif base_class is FloatWeightBase:
        class_name = name + "Weight"
    else:
        raise TypeError("Unsupported Float Base")

    return type(class_name, (mixin, base_class), {})


class MinifloatQuantizer(nn.Module):
    def __init__(
        self,
        exponent_bit_width: int,
        mantissa_bit_width: int,
        exponent_bias: Optional[int] = None,
        eps: float = 1e-9,
    ):
        super(MinifloatQuantizer, self).__init__()
        if exponent_bias is None:
            exponent_bias = (2 ** (exponent_bit_width - 1)) - 1

        self.exponent_bit_width = exponent_bit_width
        self.mantissa_bit_width = mantissa_bit_width
        self.exponent_bias = exponent_bias
        self.eps = eps
        self.ident = qnn.QuantIdentity(
            act_quant=fp_mixin_factory(
                exponent_bit_width, mantissa_bit_width, FloatActBase
            )
        )

    def forward(
        self, x: FloatTensor, return_raw: bool = False
    ) -> IntTensor | FloatTensor:
        out = self.ident(x)

        if return_raw:
            out = mf_to_raw(
                out,
                self.exponent_bit_width,
                self.mantissa_bit_width,
                self.exponent_bias,
                self.eps,
            )

        return out

    def dequantize_raw(self, x: IntTensor) -> FloatTensor:
        out = raw_to_mf(
            x, self.exponent_bit_width, self.mantissa_bit_width, self.exponent_bias
        )
        return out