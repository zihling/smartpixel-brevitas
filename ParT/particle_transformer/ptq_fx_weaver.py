import os
import glob
import argparse
import torch
import torch.nn as nn
import copy

from weaver.train import test_load
from weaver.utils.nn.tools import evaluate_classification as evaluate
from networks.example_ParticleTransformer import get_model

from brevitas.graph.quantize import preprocess_for_quantize, layerwise_quantize, LAYERWISE_COMPUTE_LAYER_MAP
import brevitas.nn as qnn

# arguments
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-config",
        default="./data/JetClass/JetClass_full.yaml",
    )

    parser.add_argument(
        "--val-glob",
        default="./datasets/JetClass/Pythia/val_5M/*.root",
    )

    parser.add_argument(
        "--test-glob",
        default="./datasets/JetClass/Pythia/test_20M/*.root",
    )

    parser.add_argument(
        "--ckpt",
        default="./training/JetClass/Pythia/full/ParT/20260309-194205_example_ParticleTransformer_ranger_lr0.001_batch512/net_best_epoch_state.pt",
    )

    parser.add_argument(
        "--out",
        default="./models/ParT_full_w8a8_ptq.pt",
    )

    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--calib-batches", type=int, default=512)

    parser.add_argument("--gpus", default="1, 2, 3")

    return parser.parse_args()
    

# ckpt loading helpers
def normalize_state_dict(sd):
    if isinstance(sd, dict):
        for k in ["state_dict", "model", "net"]:
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

def add_prefix(sd, prefix):
    return {prefix + k: v for k, v in sd.items()}

def maybe_add_mod_prefix(sd, model):
    model_keys = list(model.state_dict().keys())
    needs_mod = any(k.startswith("mod.") for k in model_keys)
    has_mod = any(k.startswith("mod.") for k in sd.keys())
    if needs_mod and not has_mod:
        sd = add_prefix(sd, "mod.")
    return sd

def load_checkpoint(model, ckpt_path, strict=True):
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = normalize_state_dict(sd)
    sd = maybe_add_mod_prefix(sd, model)
    model.load_state_dict(sd, strict=strict)
    return model

# args wrapper
def build_loader(data_config, val_glob, batch_size):

    class Args:
        pass

    args = Args()

    args.data_config = data_config
    args.data_test = [f"VAL:{val_glob}"]

    args.batch_size = batch_size
    args.num_workers = 0

    args.fetch_by_files = True
    args.fetch_step = 1.0
    args.data_fraction = 1.0

    args.predict = True
    args.gpus = "0"

    args.steps_per_epoch = None
    args.steps_per_epoch_val = None
    args.samples_per_epoch = None
    args.samples_per_epoch_val = None
    args.extra_test_selection = None

    test_loader, data_config = test_load(args)

    print("test_loader type:", type(test_loader))
    print("test_loader keys:", test_loader.keys())

    # test_loader is a dict: name -> loader factory
    loader_factory = list(test_loader.values())[0]
    loader = loader_factory()

    return loader, data_config

    # return test_loader, data_config

def unwrap_batch(batch):
    if isinstance(batch, dict):
        return batch
    if isinstance(batch, (list, tuple)) and len(batch) > 0 and isinstance(batch[0], dict):
        return batch[0]
    raise TypeError(f"Unexpected batch type: {type(batch)}")

def prepare_inputs(data, device):
    points = data["pf_points"].to(device)
    feats  = data["pf_features"].to(device)
    vecs   = data["pf_vectors"].to(device)
    mask   = data["pf_mask"].to(device).bool()
    return points, feats, vecs, mask

def count_modules(m):
    n_linear  = sum(1 for _ in m.modules() if isinstance(_, torch.nn.Linear))
    n_qlinear = sum(1 for _ in m.modules() if isinstance(_, qnn.QuantLinear))
    n_qid     = sum(1 for _ in m.modules() if isinstance(_, qnn.QuantIdentity))
    return n_linear, n_qlinear, n_qid

import torch.fx as fx

def fx_trace_leaf_pair_embed(model: torch.nn.Module):
    class Tracer(fx.Tracer):
        def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
            # Only make PairEmbed a leaf to avoid tril_indices(Proxy, Proxy)
            if m.__class__.__name__ == "PairEmbed":
                return True
            # Also catch it by attribute name, in case class name differs
            if module_qualified_name.endswith("pair_embed") or ".pair_embed" in module_qualified_name:
                return True
            return super().is_leaf_module(m, module_qualified_name)

    tracer = Tracer()
    graph = tracer.trace(model)
    return fx.GraphModule(model, graph)

# TODO: fix the underlying issue with quantizing MultiheadAttention and residual alignment, rather than just skipping quantization for it. 
def disable_packed_in_proj(m):
    # Brevitas QuantMultiheadAttention class has an optimization where it packs the in_proj weights into a single matrix. This is not compatible with FX graph transformations, so we need to disable it.
    for mod in m.modules():
        if hasattr(mod, "packed_in_proj"):
            mod.packed_in_proj = False
        if hasattr(mod, "multi_head_attention") and hasattr(mod.multi_head_attention, "packed_in_proj"):
            mod.multi_head_attention.packed_in_proj = False

def build_ffn_only_blacklist(model):
    """
    Only keep the FFN layers quantized, since those are the most important for performance and the most robust to quantization. This allows us to get good accuracy with PTQ.
    """
    blacklist = []

    for name, mod in model.named_modules():
        # Only allow transformer FFN's fc1/fc2 to be quantized
        keep = (
            name.endswith(".fc1") or
            name.endswith(".fc2")
        )

        # These types are the objects that layerwise_quantize might replace
        quantizable = isinstance(mod, (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.MultiheadAttention,
        ))

        if quantizable and not keep:
            blacklist.append(name)

    return blacklist


@torch.no_grad()
def calibrate(qmodel, loader, device, max_batches=512):
    qmodel.eval()
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        data = unwrap_batch(batch)
        points, feats, vecs, mask = prepare_inputs(data, device)
        _ = qmodel(points, feats, vecs, mask)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    val_files = sorted(glob.glob(args.val_glob))

    if len(val_files) == 0:
        raise RuntimeError("No validation files found")

    val_loader, dc = build_loader(
        args.data_config,
        args.val_glob,
        args.batch_size,
    )

    test_loader, _ = build_loader(
        args.data_config,
        args.test_glob,
        args.batch_size,
    )

    print("building model")
    model, _ = get_model(dc)
    print(model)
    model = load_checkpoint(model, args.ckpt)
    model = model.to(device).eval()
    fp_model = copy.deepcopy(model)
    print("float model built")

    # Brevitas PTQ
    print("running FX preprocess")
    gm = fx_trace_leaf_pair_embed(model)
    fx_model = preprocess_for_quantize(gm, trace_model=False)

    print("running quantize")
    custom_layer_map = copy.deepcopy(LAYERWISE_COMPUTE_LAYER_MAP)
    custom_layer_map[nn.MultiheadAttention] = None # TODO: Don't quantize MultiheadAttention, as it causes issues with residual alignment in this model. This is a temporary workaround until we can fix the underlying issue.
    name_blacklist = build_ffn_only_blacklist(fx_model)
    print("num blacklisted modules:", len(name_blacklist))
    for n in name_blacklist:
        print("blacklist:", n)
    qmodel = layerwise_quantize(fx_model, compute_layer_map=custom_layer_map, name_blacklist=name_blacklist).to(device).eval()

    print("=== sanity check qmodel types ===")
    for name, mod in qmodel.named_modules():
        if any(key in name for key in [
            "embed.embed",
            "pair_embed.embed",
            "blocks.0.fc1",
            "blocks.0.fc2",
            "blocks.0.attn",
            "cls_blocks.0.fc1",
            "cls_blocks.0.fc2",
            "fc.0",
        ]):
            print(name, "->", type(mod))
    print("module counts:")
    print("float :", count_modules(model))
    print("quant :", count_modules(qmodel))

    print("calibrating")
    calibrate(
        qmodel,
        val_loader,
        device,
        args.calib_batches,
    )

    # print("evaluating float model")
    # test_acc_fp = evaluate(
    #     fp_model, test_loader, device, epoch=0
    # )

    print("evaluating quant model")
    test_acc_ptq = evaluate(
        qmodel, test_loader, device, epoch=0
    )

    # print("FP32 accuracy:", test_acc_fp)
    print("PTQ accuracy:", test_acc_ptq)
    # print("accuracy drop:", test_acc_fp - test_acc_ptq)

    print("saving model")
    sd = qmodel.state_dict()
    print("num state_dict keys:", len(sd))
    torch.save(sd, args.out)
    print("saved:", args.out)    

if __name__ == "__main__":
    main()