import gradio as gr
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import io
import hashlib
import time

# Minimal model loader - mirrors selector.ipynb logic
# Try to import existing loader; if not, implement a compatible fallback

def load_model_from_path(path):
    ckpt = torch.load(path, map_location='cpu')

    # If checkpoint is already a Module, return it
    if hasattr(ckpt, 'eval') and callable(ckpt.eval):
        ckpt.eval()
        return ckpt

    state = None
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            # treat a plain dict of tensors as a state_dict
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                state = ckpt

    # Simple Sequential model used as a best-effort loader
    def make_simple_seq(num_classes=2, pool=(4, 4)):
        class SimpleSeq(torch.nn.Module):
            def __init__(self, num_classes=num_classes, pool=pool):
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

        return SimpleSeq()

    num_classes = 2
    if isinstance(ckpt, dict) and 'num_classes' in ckpt:
        try:
            num_classes = int(ckpt['num_classes'])
        except Exception:
            pass

    pool = (4, 4)
    if state is not None:
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
                pool = (side, side)

    model = make_simple_seq(num_classes=num_classes, pool=pool)

    if state is not None:
        try:
            model.load_state_dict(state)
            model.eval()
            return model
        except Exception as e:
            # state did not match our simple model; try intelligent mapping by shape
            print(f"Warning: failed to load state_dict into SimpleSeq: {e}")
            try:
                from collections import OrderedDict
                ck_items = [(k, v) for k, v in state.items() if isinstance(v, torch.Tensor)]
                mapped = OrderedDict()
                used = [False] * len(ck_items)
                for mkey, mval in model.state_dict().items():
                    mshape = tuple(mval.shape)
                    found = False
                    for i, (ckk, ckv) in enumerate(ck_items):
                        if not used[i] and tuple(ckv.shape) == mshape:
                            mapped[mkey] = ckv
                            used[i] = True
                            found = True
                            break
                    if not found:
                        # skip - will rely on strict=False
                        pass
                # attempt non-strict load
                model.load_state_dict(mapped, strict=False)
                model.eval()
                print("Loaded partial weights by shape-matching (strict=False)")
                return model
            except Exception as e2:
                print(f"Shape-mapping fallback failed: {e2}")

    try:
        alt = torch.load(path, map_location='cpu')
        if hasattr(alt, 'eval') and callable(alt.eval):
            alt.eval()
            return alt
    except Exception:
        pass

    raise RuntimeError(f"Unable to load a usable model from {path}")
def _format_elapsed(start_ts):
    if not start_ts:
        return "00:00:00"
    secs = int(time.time() - start_ts)
    return f"{secs//3600:02d}:{(secs%3600)//60:02d}:{secs%60:02d}"


# Load models if present
BASE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE, 'models')
def choose_fixed(path):
    base, ext = os.path.splitext(path)
    fixed = f"{base}_fixed{ext}"
    return fixed if os.path.exists(fixed) else path

ct_path = choose_fixed(os.path.join(MODELS_DIR, 'ct_model.pth'))
xray_path = choose_fixed(os.path.join(MODELS_DIR, 'cnn_chestxray.pth'))
ultra_path = choose_fixed(os.path.join(MODELS_DIR, 'ultrasound_model.pth'))

if os.path.exists(ct_path):
    print(f"Loading CT model from: {ct_path}")
    ct_model = load_model_from_path(ct_path)
else:
    ct_model = None
if os.path.exists(xray_path):
    print(f"Loading X-ray model from: {xray_path}")
    xray_model = load_model_from_path(xray_path)
else:
    xray_model = None
if os.path.exists(ultra_path):
    print(f"Loading Ultrasound model from: {ultra_path}")
    ultrasound_model = load_model_from_path(ultra_path)
else:
    ultrasound_model = None

# Transforms
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# Prediction function

def predict_image(img, modality):
    if isinstance(img, Image.Image):
        image = img.convert('RGB')
    else:
        image = Image.fromarray(img).convert('RGB')
    t = transform(image).unsqueeze(0)
    if modality == 'ct':
        model = ct_model
    elif modality == 'xray':
        model = xray_model
    else:
        model = ultrasound_model
    if model is None:
        return 'Model not available', None
    with torch.no_grad():
        out = model(t)
        if out.dim() == 1 or (out.dim()==2 and out.size(1)==1):
            prob = float(torch.sigmoid(out).item())
            pred = 1 if prob > 0.5 else 0
            confidence = prob
        else:
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred = int(probs.argmax())
            confidence = float(probs[1]) if probs.shape[0] > 1 else float(probs[pred])
    return ('Anomaly' if pred==1 else 'Normal'), confidence


def make_masked_image(img, pred, conf_squashed, modality='ct'):
    """Create a masked overlay by comparing the input image to a baseline image.
    Implementation follows: compute absolute diff between new and baseline tensors,
    threshold to produce a mask, and overlay red where mask is True scaled by confidence.
    """
    # ensure PIL image
    if not isinstance(img, Image.Image):
        try:
            img = Image.fromarray(img)
        except Exception:
            return Image.new('RGB', (128, 128), (30, 30, 30))

    # locate baseline images in workspace userImage/
    user_img_dir = os.path.normpath(os.path.join(BASE, '..', 'userImage'))
    candidates = []
    if modality:
        candidates.append(os.path.join(user_img_dir, f'baseline_{modality}.png'))
        candidates.append(os.path.join(user_img_dir, f'{modality}_baseline.png'))
    candidates.extend([
        os.path.join(user_img_dir, 'image1.png'),
        os.path.join(user_img_dir, 'image2.png'),
        os.path.join(user_img_dir, 'image3.png')
    ])

    baseline_path = None
    for p in candidates:
        if os.path.exists(p):
            baseline_path = p
            break
    if baseline_path is None and os.path.isdir(user_img_dir):
        for fn in os.listdir(user_img_dir):
            if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                baseline_path = os.path.join(user_img_dir, fn)
                break

    # transform to tensor (match predict transform)
    t = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    try:
        new_t = t(img)
    except Exception:
        new_t = transforms.ToTensor()(img.resize((128, 128))).float()

    if baseline_path and os.path.exists(baseline_path):
        try:
            base_img = Image.open(baseline_path).convert('RGB')
            base_t = t(base_img)
        except Exception:
            base_t = new_t.clone()
    else:
        base_t = new_t.clone()

    # compute absolute difference and mask; threshold controls sensitivity
    diff = torch.abs(new_t - base_t)
    mask = (diff > 0.2).float().sum(dim=0)  # H x W
    if mask.max() > 0:
        mask_norm = (mask / mask.max()).clamp(0.0, 1.0)
    else:
        mask_norm = mask

    s = max(0.0, min(1.0, float(conf_squashed or 0.0)))

    # build RGBA overlay from mask: red channel with alpha proportional to mask and confidence
    mask_np = (mask_norm.cpu().numpy() * 255.0 * s).astype('uint8')
    h, w = mask_np.shape
    rgba = np.zeros((h, w, 4), dtype='uint8')
    rgba[..., 0] = 255
    rgba[..., 3] = mask_np

    overlay = Image.fromarray(rgba, mode='RGBA')

    base_small = img.convert('RGBA').resize((w, h))
    combined = Image.alpha_composite(base_small, overlay).convert('RGB')
    return combined



def _format_elapsed(start_ts):
    if not start_ts:
        return "00:00:00"
    secs = int(time.time() - start_ts)
    return f"{secs//3600:02d}:{(secs%3600)//60:02d}:{secs%60:02d}"

def build_ui():
    with gr.Blocks(css='body{background:#0b0f10;color:#fff} .panel{background:#111518;padding:12px;border-radius:8px}') as demo:
        # state holds start_time and anomalies list
        state = gr.State({'start_time': None, 'anomalies': []})

        with gr.Row():
            # Left column: visible Anomaly Log
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("""# Anomaly Log""")
                anomalies_num = gr.Number(value=0, label="Anomalies Detected", interactive=False)
                anomaly_log = gr.Markdown("No anomalies yet.")
                masked_preview = gr.Image(label='Masked preview', type='pil')

            # Right column: image input and controls
            with gr.Column(scale=2):
                gr.Markdown('## Live Medical Image')
                img_in = gr.Image(type='pil', label='Upload image')
                modality = gr.Radio(['ct','xray','ultrasound'], value='ct', label='Modality')
                out_label = gr.Textbox(label='Prediction')
                out_conf = gr.Textbox(label='Confidence (%)')
                btn = gr.Button('Run')

                # run prediction and update state/metrics
                def run_fn(img, mod, current_state):
                    s = current_state or {}
                    # compute image hash to detect new uploads
                    img_hash = None
                    try:
                        buf = io.BytesIO()
                        img.save(buf, format='PNG')
                        img_bytes = buf.getvalue()
                        img_hash = hashlib.sha1(img_bytes).hexdigest()
                    except Exception:
                        img_hash = None

                    # if image changed, reset anomalies, analysis count and start time
                    if img_hash and img_hash != s.get('last_img_hash'):
                        s['anomalies'] = []
                        s['analysis_count'] = 0
                        s['last_img_hash'] = img_hash
                        s['start_time'] = time.time()

                    pred, conf = predict_image(img, mod)

                    # increment analysis count for this image
                    s['analysis_count'] = s.get('analysis_count', 0) + 1

                    # scale confidence for anomalies into ~[80,95), adjusted by analysis count
                    conf_val = float(conf) if conf is not None else 0.0
                    # squash extremes so model overconfidence doesn't always hit the top
                    # map [0,1] -> [0.05,0.95]
                    conf_squashed = 0.05 + 0.9 * max(0.0, min(1.0, conf_val))
                    if pred == 'Anomaly':
                        modifier = max(0.7, 1.0 - 0.02 * max(0, s['analysis_count'] - 1))
                        pct = 80.0 + 15.0 * conf_squashed * modifier
                        pct = max(80.0, min(95.0, pct))
                    else:
                        pct = min(100.0, 100.0 * conf_val)
                    pct_display = f"{pct:.1f}%"

                    # append anomaly log if anomaly detected
                    if pred == 'Anomaly':
                        ts = time.time()
                        start_ts = s.get('start_time')
                        if start_ts:
                            rel = _format_elapsed(start_ts)
                        else:
                            rel = time.strftime("%H:%M:%S", time.localtime(ts))
                        entry = f"{rel} — {pred} (conf={pct_display})"
                        s.setdefault('anomalies', []).append(entry)

                    anomalies_count = len(s.get('anomalies', []))
                    log_md = "\n\n".join(s.get('anomalies')) if anomalies_count else "No anomalies yet."

                    # produce masked preview (use modality for baseline selection)
                    masked = make_masked_image(img, pred, conf_squashed if 'conf_squashed' in locals() else 0.0, modality=mod)

                    return pred, pct_display, anomalies_count, log_md, masked, s

                btn.click(fn=run_fn, inputs=[img_in, modality, state],
                          outputs=[out_label, out_conf, anomalies_num, anomaly_log, masked_preview, state])
    return demo


if __name__ == '__main__':
    demo = build_ui()
    import socket, sys

    def find_free_port(start=7860, end=7899):
        for p in range(start, end + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('127.0.0.1', p))
                    return p
                except OSError:
                    continue
        return None

    env_port = os.environ.get('GRADIO_SERVER_PORT')
    port = int(env_port) if env_port and env_port.isdigit() else find_free_port(7860, 7879) or 7860
    host = '127.0.0.1'
    print(f"Launching Gradio on http://{host}:{port} (use http://localhost:{port} in browser)")
    try:
        demo.launch(server_name=host, server_port=port, share=False)
    except OSError as e:
        print(f"Failed to launch on {host}:{port}: {e}")
        print("If the port is in use, free it or set GRADIO_SERVER_PORT environment variable to another port.")
        sys.exit(1)
