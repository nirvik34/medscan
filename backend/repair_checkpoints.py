"""
Utility to remap model checkpoint tensors into a repaired checkpoint matching the SimpleSeq model
used by the Gradio app. This script attempts deterministic mapping by shape and order and writes
`<origname>_fixed.pth` with {'model_state_dict': repaired_state, 'num_classes': ...}.

Run:
    python repair_checkpoints.py

"""
import os
import torch
from collections import OrderedDict

BASE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE, 'models')
CHECKPOINTS = [
    ('ct_model.pth', 'ct_model_fixed.pth'),
    ('cnn_chestxray.pth', 'cnn_chestxray_fixed.pth'),
    ('ultrasound_model.pth', 'ultrasound_model_fixed.pth'),
]


def make_simple_seq_state_dict(num_classes=2, pool=(4, 4)):
    """Create an empty SimpleSeq model and return its state_dict skeleton (with zero tensors).
    We don't need actual values, only keys and shapes."""
    class SimpleSeq(torch.nn.Module):
        def __init__(self, num_classes=2, pool=(4,4)):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(16, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
                torch.nn.AdaptiveAvgPool2d(pool),
                torch.nn.Flatten(),
                torch.nn.Linear(32 * pool[0] * pool[1], 128), torch.nn.ReLU(),
                torch.nn.Linear(128, num_classes)
            )
        def forward(self, x):
            return self.net(x)
    m = SimpleSeq(num_classes=num_classes, pool=pool)
    sd = m.state_dict()
    return sd


def infer_pool_from_state(state):
    # find linear weight with largest in_features
    linear_in = None
    for k, v in state.items():
        if k.endswith('.weight') and isinstance(v, torch.Tensor) and v.dim() == 2:
            in_f = v.shape[1]
            if linear_in is None or in_f > linear_in:
                linear_in = int(in_f)
    if linear_in is not None and linear_in % 32 == 0:
        area = linear_in // 32
        side = int((area ** 0.5))
        if side * side == area:
            return (side, side)
    return (4,4)


def repair_checkpoint(orig_path, out_path):
    full = os.path.join(MODELS_DIR, orig_path)
    if not os.path.exists(full):
        print(f"Skipping missing {orig_path}")
        return
    print(f"Processing {orig_path}...")
    ckpt = torch.load(full, map_location='cpu')
    if isinstance(ckpt, dict) and ('model_state_dict' in ckpt or 'state_dict' in ckpt):
        state = ckpt.get('model_state_dict', ckpt.get('state_dict'))
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
    else:
        print(f"Checkpoint {orig_path} appears to be a full model object; skipping (can't remap)")
        return

    pool = infer_pool_from_state(state)
    num_classes = int(ckpt.get('num_classes', 2)) if isinstance(ckpt, dict) else 2
    target_sd = make_simple_seq_state_dict(num_classes=num_classes, pool=pool)

    # Build list of checkpoint tensors
    ck_items = [(k, v) for k, v in state.items() if isinstance(v, torch.Tensor)]

    repaired = OrderedDict()
    used = [False] * len(ck_items)

    # First pass: exact key match
    for k in target_sd.keys():
        if k in state and isinstance(state[k], torch.Tensor) and tuple(state[k].shape) == tuple(target_sd[k].shape):
            repaired[k] = state[k].clone()
            # mark used
            for i, (ckk, _) in enumerate(ck_items):
                if ckk == k:
                    used[i] = True
                    break

    # Second pass: match by shape (deterministic order)
    for k in target_sd.keys():
        if k in repaired:
            continue
        tshape = tuple(target_sd[k].shape)
        found = False
        for i, (ckk, ckv) in enumerate(ck_items):
            if not used[i] and tuple(ckv.shape) == tshape:
                repaired[k] = ckv.clone()
                used[i] = True
                found = True
                break
        if not found:
            # keep original target (zeros)
            repaired[k] = target_sd[k].clone()

    # report
    total_keys = len(target_sd)
    restored = sum(1 for k in target_sd.keys() if k in repaired and not repaired[k].sum().item() == 0)
    print(f"Repaired: {restored}/{total_keys} parameters assigned (by exact or shape match)")

    out_ckpt = {'model_state_dict': repaired, 'num_classes': num_classes}
    out_full = os.path.join(MODELS_DIR, out_path)
    torch.save(out_ckpt, out_full)
    print(f"Saved repaired checkpoint to {out_full}\n")


if __name__ == '__main__':
    for orig, fixed in CHECKPOINTS:
        repair_checkpoint(orig, fixed)
    print('Done.')
