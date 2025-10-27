from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from brevitas.quant.experimental.float_base import (
    FloatActBase,
    FloatWeightBase,
)
import brevitas.nn as qnn
from torch import Tensor
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Int8Bias
from brevitas.quant_tensor.float_quant_tensor import FloatQuantTensor
from brevitas.quant_tensor.int_quant_tensor import IntQuantTensor
from .minifloat import mf_to_raw, fp_mixin_factory

NUM_CLASSES = 12

__all__ = [
    "DenseModel",
    "DenseModelLarge",
    "QuantDenseModel",
    "QuantDenseModelLarge",
    "IntQuantDenseModel",
    "FloatQuantDenseModel",
    "FloatQuantDenseModelLarge",
]

# Original dense model in PyTorch
class DenseModel(nn.Module):
    """
    Original float model, used for small model, medium model and large model2
    Small model: input_shape=13, dense_width=16
    Medium model: input_shape=13, dense_width=58
    Large model2: input_shape=13, dense_width=512
    """
    def __init__(self, in_features, dense_width=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, dense_width),
            nn.BatchNorm1d(dense_width),
            nn.ReLU(),
            nn.Linear(dense_width, NUM_CLASSES)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
    
class DenseModelLarge(nn.Module):
    """
    Original float model, used for large model
    Large model1: input_shape=13, dense_width=32
    """
    def __init__(self, in_features, dense_width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, dense_width),
            nn.BatchNorm1d(dense_width),
            nn.ReLU(),
            nn.Linear(dense_width, dense_width),
            nn.BatchNorm1d(dense_width),
            nn.ReLU(),
            nn.Linear(dense_width, dense_width),
            nn.BatchNorm1d(dense_width),
            nn.ReLU(),
            nn.Linear(dense_width, NUM_CLASSES)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

# TODO: add more quantized model for mini-float supported model 
# Quantized model using Brevitas
class QuantDenseModel(nn.Module, metaclass=ABCMeta):
    """
    Abstract base for quantized dense models.
    Child classes (IntQuantDenseModel, FloatQuantDenseModel)
    define quantization behavior.
    """

    def __init__(
        self,
        in_features: int,
        dense_width: int = 58,
        num_classes: int = 12,
        **kwargs,
    ):
        super(QuantDenseModel, self).__init__()

    # !: Define all layers in child classes

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

    @abstractmethod
    def quant_weight(self) -> tuple[Tensor, Tensor]:
        """Get quantized model weights."""
        pass

    @abstractmethod
    def quant_input(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize input tensor."""
        pass

class QuantDenseModelLarge(nn.Module):
    def __init__(
        self,
        in_features: int,
        dense_width: int = 58,
        num_classes: int = 12,
        **kwargs,
    ):
        super().__init__()

        @abstractmethod
        def build_layers(self):
            """Child classes define quantization blocks and output layer."""
            pass

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.output(x)
        return x
    
    @abstractmethod
    def quant_weight(self) -> tuple[Tensor, Tensor]:
        """Get quantized model weights."""
        pass

    @abstractmethod
    def quant_input(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize input tensor."""
        pass

# Integer-quantized model using Brevitas
class IntQuantDenseModel(QuantDenseModel):
    def __init__(
        self,
        in_features: int,
        dense_width: int = 58,
        logit_total_bits: int = 8,
        activation_total_bits: int = 8,
        num_classes: int = 12,
    ):
        super(IntQuantDenseModel, self).__init__(
            in_features=in_features,
            dense_width=dense_width,
            num_classes=num_classes,
        )

        self.fc1 = qnn.QuantLinear(
            in_features=in_features,
            out_features=dense_width,
            bias=False,
            weight_quant=Int8WeightPerTensorFloat,
            weight_bit_width=logit_total_bits,
            input_quant=Int8ActPerTensorFloat,
            input_bit_width=activation_total_bits,
        )
        self.bn1 = nn.BatchNorm1d(dense_width)
        self.act1 = qnn.QuantReLU(bit_width=activation_total_bits)
        self.fc2 = qnn.QuantLinear(
            in_features=dense_width,
            out_features=num_classes,
            bias=True,
            bias_quant=Int8Bias,
            weight_quant=Int8WeightPerTensorFloat,
            weight_bit_width=logit_total_bits,
            input_quant=Int8ActPerTensorFloat,
            input_bit_width=activation_total_bits,
        )

    def quant_weight(self) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            w: IntQuantTensor = self.fc1.quant_weight()
        return w.int(), w.scale

    def quant_input(self, x: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            inp: IntQuantTensor = self.fc1.input_quant(x)
        return inp.int(), inp.scale
    
class IntQuantDenseModelLarge(QuantDenseModelLarge):
    def __init__(
        self,
        in_features: int,
        dense_width: int = 58,
        logit_total_bits: int = 4,
        activation_total_bits: int = 8,
        num_classes: int = 12,
    ):
        super(IntQuantDenseModelLarge, self).__init__(
            in_features=in_features,
            dense_width=dense_width,
            num_classes=num_classes,
        )

        def quant_block(in_f, out_f):
            return nn.Sequential(
                qnn.QuantLinear(
                    in_features=in_f,
                    out_features=out_f,
                    bias=False,
                    weight_quant=Int8WeightPerTensorFloat,
                    weight_bit_width=logit_total_bits,
                    input_quant=Int8ActPerTensorFloat,
                    input_bit_width=activation_total_bits,
                ),
                nn.BatchNorm1d(out_f),
                qnn.QuantReLU(bit_width=activation_total_bits)
            )

        self.block1 = quant_block(in_features, dense_width)
        self.block2 = quant_block(dense_width, dense_width)
        self.block3 = quant_block(dense_width, dense_width)
        self.output = qnn.QuantLinear(
            in_features=dense_width,
            out_features=NUM_CLASSES,
            bias=True,
            bias_quant=Int8Bias,
            weight_quant=Int8WeightPerTensorFloat,
            weight_bit_width=logit_total_bits,
            input_quant=Int8ActPerTensorFloat,
            input_bit_width=activation_total_bits,
        )

    def quant_weight(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        quant_weights = {}
        model_scales = {}
        with torch.no_grad():
            for name, module in self.named_modules():
                if isinstance(module, qnn.QuantLinear):
                    w = module.quant_weight()
                    quant_weights[name] = w.int()
                    model_scales[name] = w.scale
        return quant_weights, model_scales

    def quant_input(self, x: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            first_fc = self.block1[0]
            inp: IntQuantTensor = first_fc.input_quant(x)
        return inp.int(), inp.scale


# Floating-point-quantized model using Brevitas    
class FloatQuantDenseModel(QuantDenseModel):
    """
    Dense model with floating-point (minifloat) quantization.
    Uses fp_mixin_factory to create FloatActBase / FloatWeightBase quantizers.
    """
    def __init__(
        self,
        in_features: int,
        dense_width: int = 58,
        num_classes: int = 12,
        input_exponent_bit_width: int = 4,
        input_mantissa_bit_width: int = 3,
        weight_exponent_bit_width: int = 4,
        weight_mantissa_bit_width: int = 3,
    ):
        activation_total_bits = 1 + input_exponent_bit_width + input_mantissa_bit_width
        logit_total_bits = 1 + weight_exponent_bit_width + weight_mantissa_bit_width
        # Build custom minifloat quantizers
        input_quant = fp_mixin_factory(
            input_exponent_bit_width, input_mantissa_bit_width, FloatActBase
        )
        weight_quant = fp_mixin_factory(
            weight_exponent_bit_width, weight_mantissa_bit_width, FloatWeightBase
        )
        bias_quant = fp_mixin_factory(
            input_exponent_bit_width, input_mantissa_bit_width, FloatActBase, bias=True
        )

        super(FloatQuantDenseModel, self).__init__(
            in_features=in_features,
            dense_width=dense_width,
            num_classes=num_classes,
            input_quant=input_quant,
            weight_quant=weight_quant,
        )

        self.fc1 = qnn.QuantLinear(
            in_features=in_features,
            out_features=dense_width,
            bias=False,
            weight_quant=weight_quant,
            weight_bit_width=logit_total_bits,
            input_quant=input_quant,
            input_bit_width=activation_total_bits,
        )
        self.bn1 = nn.BatchNorm1d(dense_width)
        self.act1 = qnn.QuantReLU(bit_width=activation_total_bits)
        self.fc2 = qnn.QuantLinear(
            in_features=dense_width,
            out_features=num_classes,
            bias=False,
            # bias_quant=bias_quant,
            weight_quant=weight_quant,
            weight_bit_width=logit_total_bits,
            input_quant=input_quant,
            input_bit_width=activation_total_bits,
        )

    def quant_weight(self, return_raw: bool = False) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            weights: FloatQuantTensor = self.fc1.quant_weight()
        out = weights.minifloat()

        if return_raw:
            out = mf_to_raw(
                out,
                weights.exponent_bit_width.int(),
                weights.mantissa_bit_width.int(),
                weights.exponent_bias.int(),
                weights.eps,
            ).type(torch.uint8)

        return out, weights.scale

    def quant_input(self, x: Tensor, return_raw: bool = False) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            inputs: FloatQuantTensor = self.fc1.input_quant(x)
        out = inputs.minifloat()

        if return_raw:
            out = mf_to_raw(
                out,
                inputs.exponent_bit_width.int(),
                inputs.mantissa_bit_width.int(),
                inputs.exponent_bias.int(),
                inputs.eps,
            ).type(torch.uint8)

        return out, inputs.scale
    
class FloatQuantDenseModelLarge(QuantDenseModelLarge):
    def __init__(
        self,
        in_features: int,
        dense_width: int = 58,
        num_classes: int = 12,
        input_exponent_bit_width: int = 4,
        input_mantissa_bit_width: int = 3,
        weight_exponent_bit_width: int = 4,
        weight_mantissa_bit_width: int = 3,
    ):
        activation_total_bits = 1 + input_exponent_bit_width + input_mantissa_bit_width
        logit_total_bits = 1 + weight_exponent_bit_width + weight_mantissa_bit_width
        # Build custom minifloat quantizers
        input_quant = fp_mixin_factory(
            input_exponent_bit_width, input_mantissa_bit_width, FloatActBase
        )
        weight_quant = fp_mixin_factory(
            weight_exponent_bit_width, weight_mantissa_bit_width, FloatWeightBase
        )
        bias_quant = fp_mixin_factory(
            input_exponent_bit_width, input_mantissa_bit_width, FloatActBase, bias=True
        )

        super(FloatQuantDenseModelLarge, self).__init__(
            in_features=in_features,
            dense_width=dense_width,
            num_classes=num_classes,
        )

        def quant_block(in_f, out_f):
            return nn.Sequential(
                qnn.QuantLinear(
                    in_features=in_f,
                    out_features=out_f,
                    bias=False,
                    weight_quant=weight_quant,
                    weight_bit_width=logit_total_bits,
                    input_quant=input_quant,
                    input_bit_width=activation_total_bits,
                ),
                nn.BatchNorm1d(out_f),
                qnn.QuantReLU(bit_width=activation_total_bits)
            )

        self.block1 = quant_block(in_features, dense_width)
        self.block2 = quant_block(dense_width, dense_width)
        self.block3 = quant_block(dense_width, dense_width)
        self.output = qnn.QuantLinear(
            in_features=dense_width,
            out_features=NUM_CLASSES,
            bias=False,
            # bias_quant=Int8Bias,
            weight_quant=weight_quant,
            weight_bit_width=logit_total_bits,
            input_quant=input_quant,
            input_bit_width=activation_total_bits,
        )

    def quant_weight(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        quant_weights = {}
        model_scales = {}
        with torch.no_grad():
            for name, module in self.named_modules():
                if isinstance(module, qnn.QuantLinear):
                    w = module.quant_weight()
                    quant_weights[name] = w.int()
                    model_scales[name] = w.scale
        return quant_weights, model_scales

    def quant_input(self, x: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            first_fc = self.block1[0]
            inp: IntQuantTensor = first_fc.input_quant(x)
        return inp.int(), inp.scale