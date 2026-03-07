import os
import sys

import torch.nn as nn
import copy

# Ensure the directory (networks/) is on sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from brevitas.graph.quantize import preprocess_for_quantize, layerwise_quantize, LAYERWISE_COMPUTE_LAYER_MAP
from example_ParticleTransformer import get_model as get_float_model

import torch.fx as fx
import torch


def fx_trace_leaf_pair_embed(model: torch.nn.Module) -> fx.GraphModule:
    class Tracer(fx.Tracer):
        def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
            if m.__class__.__name__ == "PairEmbed":
                return True
            if module_qualified_name.endswith("pair_embed") or ".pair_embed" in module_qualified_name:
                return True
            return super().is_leaf_module(m, module_qualified_name)

    tracer = Tracer()
    graph = tracer.trace(model)
    return fx.GraphModule(model, graph)


def get_model(data_config, **kwargs):
    float_model, model_info = get_float_model(data_config, **kwargs)

    custom_layer_map = copy.deepcopy(LAYERWISE_COMPUTE_LAYER_MAP)
    custom_layer_map[nn.MultiheadAttention] = None

    # Build quantized structure exactly like PTQ script
    gm = fx_trace_leaf_pair_embed(float_model)
    fx_model = preprocess_for_quantize(gm, trace_model=False)

    # Use layerwise_quantize (quantize() failed on residual alignment)
    qmodel = layerwise_quantize(fx_model, compute_layer_map=custom_layer_map)

    return qmodel, model_info