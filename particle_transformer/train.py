import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch_optimizer as optim
from tqdm import tqdm
import awkward as ak
import numpy as np
import uproot
from particle_transformer.model import ParTModel


class ParticleDataset(Dataset): # 16 input features from particle data, turned into 17 derived features
    def __init__(self, data_dir, in_features = 17, T=128, num_classes=10, tree_name="tree"):
        self.data_dir = Path(data_dir)
        self.in_features = in_features
        self.T = T
        self.num_classes = num_classes
        self.tree_name = tree_name

        # collect ROOT files
        self.files = sorted(self.data_dir.glob("*.root"))
        assert len(self.files) > 0, f"No .root files found in {data_dir}"

        # count events per file
        self.file_event_counts = []
        for f in self.files:
            with uproot.open(f) as file:
                tree = file[self.tree_name]
                self.file_event_counts.append(tree.num_entries)

        self.cum_events = np.cumsum(self.file_event_counts)

        # Label order
        self.label_names = [
            "label_QCD",
            "label_Hbb",
            "label_Hcc",
            "label_Hgg",
            "label_H4q",
            "label_Hqql",
            "label_Zqq",
            "label_Wqq",
            "label_Tbqq",
            "label_Tbl",
        ]

    def __len__(self):
        return int(self.cum_events[-1])
    
    def _locate(self, idx):
        file_idx = np.searchsorted(self.cum_events, idx, side="right")
        local_idx = idx if file_idx == 0 else idx - self.cum_events[file_idx - 1]
        return file_idx, local_idx

    def __getitem__(self, idx):
        file_idx, local_idx = self._locate(idx)
        file_path = self.files[file_idx]

        with uproot.open(file_path) as file:
            tree = file[self.tree_name]

            # ---- load particle branches (jagged) ----
            arr = tree.arrays(
                [
                    # kinematics
                    "part_px",
                    "part_py",
                    "part_pz",
                    "part_energy",
                    "part_deta",
                    "part_dphi",

                    # displacement
                    "part_d0val",
                    "part_d0err",
                    "part_dzval",
                    "part_dzerr",

                    # charge
                    "part_charge",

                    # PID (already one-hot)
                    "part_isChargedHadron",
                    "part_isNeutralHadron",
                    "part_isPhoton",
                    "part_isElectron",
                    "part_isMuon",

                    # jet-level
                    "jet_pt",
                    "jet_energy",
                ],
                entry_start=local_idx,
                entry_stop=local_idx + 1,
            )

            # ---- load label ----
            labels = tree.arrays(
                self.label_names,
                entry_start=local_idx,
                entry_stop=local_idx + 1,
            )[0]

        # ---- particle arrays ----
        px = arr["part_px"][0]
        py = arr["part_py"][0]
        pz = arr["part_pz"][0]
        E  = arr["part_energy"][0]
        deta = arr["part_deta"][0]
        dphi = arr["part_dphi"][0]

        # ---- derived features ----
        pt = np.sqrt(px**2 + py**2) + 1e-8
        log_pt = np.log(pt)
        log_E = np.log(E + 1e-8)

        jet_pt = arr["jet_pt"][0]
        jet_E  = arr["jet_energy"][0]

        log_pt_rel = np.log(pt / (jet_pt + 1e-8))
        log_E_rel  = np.log(E / (jet_E + 1e-8))

        deltaR = np.sqrt(deta**2 + dphi**2)


        # stack particle features
        feats = np.stack(
            [
                ak.to_numpy(deta),
                ak.to_numpy(dphi),
                ak.to_numpy(log_pt),
                ak.to_numpy(log_E),
                ak.to_numpy(log_pt_rel),
                ak.to_numpy(log_E_rel),
                ak.to_numpy(deltaR),
                ak.to_numpy(arr["part_d0val"][0]),
                ak.to_numpy(arr["part_d0err"][0]),
                ak.to_numpy(arr["part_dzval"][0]),
                ak.to_numpy(arr["part_dzerr"][0]),
                ak.to_numpy(arr["part_charge"][0]),
                ak.to_numpy(arr["part_isChargedHadron"][0]),
                ak.to_numpy(arr["part_isNeutralHadron"][0]),
                ak.to_numpy(arr["part_isPhoton"][0]),
                ak.to_numpy(arr["part_isElectron"][0]),
                ak.to_numpy(arr["part_isMuon"][0]),
            ],
            axis=1,
        )

        # feats = ak.to_numpy(feats)
        n_particles = min(len(feats), self.T)

        # ---- build tensors ----
        x = torch.zeros(self.T, self.in_features, dtype=torch.float32)
        x[:n_particles] = torch.from_numpy(feats[:n_particles]).float()

        assert x.dim() == 2
        assert x.shape[-1] == self.in_features   # 17

        mask = torch.zeros(self.T, dtype=torch.bool)
        mask[:n_particles] = True


        # pairwise interaction tensor U (T, T, 4)
        U = torch.zeros(self.T, self.T, 4, dtype=torch.float32) # 4 features per pair

        for i in range(n_particles):
            for j in range(n_particles):
                delta_eta = deta[i] - deta[j]
                delta_phi = dphi[i] - dphi[j]
                delta = np.sqrt(delta_eta**2 + delta_phi**2)

                kT = min(pt[i], pt[j]) * delta
                z = min(pt[i], pt[j]) / (pt[i] + pt[j] + 1e-8)
                m2 = (E[i] + E[j])**2 - ((px[i] + px[j])**2 + (py[i] + py[j])**2 + (pz[i] + pz[j])**2)

                U[i, j, 0] = float(delta)
                U[i, j, 1] = float(kT)
                U[i, j, 2] = float(z)
                U[i, j, 3] = float(m2)

        # ---- build single class label ----
        label_vals = [labels[name] for name in self.label_names]

        # QCD is float, others are bool/int
        if label_vals[0] > 0:
            y = 0
        else:
            y = int(np.argmax(label_vals[1:]) + 1)


        return x, U, mask, y
    
# Training and evaluation functions
from itertools import cycle


def train_one_iter(model, batch, optimizer, device):
    model.train()

    x, U, mask, y = batch
    x = x.to(device)
    U = U.to(device)
    mask = mask.to(device)
    mask_bool = mask
    attn_mask = (~mask_bool).float() * (-1e9)
    attn_mask = attn_mask[:, None, None, :]   # [B, 1, 1, N]
    y = y.to(device)

    optimizer.zero_grad()


    logits = model(x, U, attn_mask)
    if hasattr(logits, "value"):   # unwrap Brevitas QuantTensor to a torch.Tensor
        logits = logits.value 

    logits = logits.squeeze(1)     # [B, 1, 10] -> [B, 10]
    loss = F.cross_entropy(logits, y)

    loss.backward()
    optimizer.step()
    
    # print_ram() # Debugging: print RAM usage after each iteration

    return loss.item()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for x, U, mask, y in loader:
        x = x.to(device)
        U = U.to(device)
        mask = mask.to(device)
        y = y.to(device)

        attn_mask = (~mask).float() * (-1e9)
        attn_mask = attn_mask[:, None, None, :]

        logits = model(x, U, attn_mask)
        if hasattr(logits, "value"):
            logits = logits.value
        logits = logits.squeeze(1)
        preds = logits.argmax(dim=-1)

        correct += (preds == y).sum().item()
        total += y.numel()

    return correct / total



def main():
    # Hyperparameters
    batch_size = 256 # 512 is the original setting
    total_iters = 1000000
    eval_interval = 20000
    initial_lr = 1e-3
    num_classes = 10


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ParticleDataset(data_dir="particle_transformer/data/JetClass_Pythia_train_100M", num_classes=num_classes)
    val_ds   = ParticleDataset(data_dir="particle_transformer/data/JetClass_Pythia_val_5M/val_5M",  num_classes=num_classes)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)

    model = ParTModel(
        in_features=17,
        d_model=128,
        num_heads=8,
        num_classes=num_classes,
        w_bit_width=8,
        a_bit_width=8,
        pab_num=8,
        cab_num=2,
    ).to(device)

    # Optimizer
    base_optimizer = optim.RAdam(
        model.parameters(),
        lr=initial_lr,
        betas=(0.95, 0.999),
        eps=1e-5,
        weight_decay=0.0
    )

    optimizer = optim.Lookahead(
        base_optimizer,
        k=6,
        alpha=0.5
    )

    def lr_lambda(step):
        warm = int(0.7 * total_iters)
        if step < warm:
            return 1.0
        decay_steps = (step - warm) // 20000
        return 0.99 ** decay_steps  # decays to ~1% by end

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_lambda
    )

    # Training loop

    best_val_acc = 0.0
    global_step = 0

    train_iter = cycle(train_loader)

    for step in range(1, total_iters + 1):
        batch = next(train_iter)

        loss = train_one_iter(
            model=model,
            batch=batch,
            optimizer=optimizer,
            device=device,
        )

        scheduler.step()

        # Evaluation & checkpoint
        if step % eval_interval == 0:
            val_acc = evaluate(model, val_loader, device)
            lr = scheduler.get_last_lr()[0]

            print(
                f"Iter {step:7d} | "
                f"Loss {loss:.4f} | "
                f"Val Acc {val_acc:.4f} | "
                f"LR {lr:.2e}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "results/best_part_model.pth")


    # Final checkpoint
    torch.save(model.state_dict(), "results/part_model_final.pth")

    # Test

    test_ds = ParticleDataset(
        data_dir="data/test_20M",
        num_classes=num_classes
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=512,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    test_acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()