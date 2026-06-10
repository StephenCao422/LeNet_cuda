#!/usr/bin/env python3
"""Generate a self-contained HTML visualization of CNN feature maps.

Usage: python3 visualize.py feature_maps/
"""

import sys
import os
import struct
import zlib
import base64
import numpy as np

# Layer index -> (name, shape) mapping matching ece408net.cc
LAYER_INFO = [
    ("Conv1",     (4, 80, 80)),
    ("ReLU1",     (4, 80, 80)),
    ("MaxPool1",  (4, 40, 40)),
    ("Conv2",     (16, 34, 34)),
    ("ReLU2",     (16, 34, 34)),
    ("MaxPool2",  (16, 9, 9)),
    ("FC1",       (32,)),
    ("ReLU3",     (32,)),
    ("FC2",       (10,)),
    ("Softmax",   (10,)),
]

INPUT_SHAPE = (1, 86, 86)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def read_bin(path):
    with open(path, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.float32)


def make_png_bytes(pixels_u8):
    """Encode a 2D uint8 numpy array as a grayscale PNG (stdlib only)."""
    h, w = pixels_u8.shape
    # Build raw image data with filter byte (0 = None) per row
    raw = b""
    for row in range(h):
        raw += b"\x00" + pixels_u8[row].tobytes()
    compressed = zlib.compress(raw)

    def chunk(ctype, data):
        c = ctype + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)  # 8-bit grayscale
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")


def to_base64_img(pixels_u8, scale=1):
    """Convert uint8 2D array to base64 inline PNG <img> tag."""
    png = make_png_bytes(pixels_u8)
    b64 = base64.b64encode(png).decode("ascii")
    h, w = pixels_u8.shape
    sw, sh = w * scale, h * scale
    return f'<img src="data:image/png;base64,{b64}" width="{sw}" height="{sh}" style="image-rendering:pixelated;">'


def normalize_to_u8(arr):
    """Normalize float array to 0-255 uint8."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


def render_feature_maps(name, data, shape):
    """Render a convolutional/pooling layer as a grid of images."""
    c, h, w = shape
    maps = data.reshape(c, h, w)
    scale = max(1, 160 // max(h, w))  # scale small maps up
    html = f'<div class="layer"><h3>{name} <span class="dim">{c}x{h}x{w}</span></h3><div class="maps">'
    for i in range(c):
        u8 = normalize_to_u8(maps[i])
        html += f'<div class="map"><div class="label">ch {i}</div>{to_base64_img(u8, scale)}</div>'
    html += "</div></div>"
    return html


def render_fc(name, data, n):
    """Render a fully-connected layer as horizontal bars."""
    vals = data[:n]
    mn, mx = vals.min(), vals.max()
    rng = mx - mn if mx - mn > 1e-8 else 1.0
    html = f'<div class="layer"><h3>{name} <span class="dim">{n} values</span></h3><div class="bars">'
    for i in range(n):
        pct = (vals[i] - mn) / rng * 100
        html += f'<div class="bar-row"><span class="bar-idx">{i}</span>'
        html += f'<div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>'
        html += f'<span class="bar-val">{vals[i]:.4f}</span></div>'
    html += "</div></div>"
    return html


def render_softmax(data, true_label, predicted):
    """Render softmax probabilities with class names."""
    probs = data[:10]
    html = '<div class="layer"><h3>Softmax <span class="dim">10 classes</span></h3><div class="softmax">'
    for i in range(10):
        pct = probs[i] * 100
        cls = "highlight-true" if i == true_label else ""
        cls += " highlight-pred" if i == predicted else ""
        html += f'<div class="sm-row {cls}">'
        html += f'<span class="sm-name">{CLASS_NAMES[i]}</span>'
        html += f'<div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>'
        html += f'<span class="bar-val">{probs[i]:.4f}</span></div>'
    html += "</div></div>"
    return html


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 visualize.py <feature_maps_dir>")
        sys.exit(1)

    fdir = sys.argv[1]

    # Read metadata
    true_label, predicted = -1, -1
    meta_path = os.path.join(fdir, "metadata.txt")
    if os.path.exists(meta_path):
        for line in open(meta_path):
            k, v = line.strip().split("=")
            if k == "true_label":
                true_label = int(v)
            elif k == "predicted":
                predicted = int(v)

    sections = []

    # Input image
    inp = read_bin(os.path.join(fdir, "input.bin"))
    inp_img = inp.reshape(INPUT_SHAPE[1], INPUT_SHAPE[2])
    u8 = normalize_to_u8(inp_img)
    sections.append(
        f'<div class="layer"><h3>Input <span class="dim">86x86</span></h3>'
        f'<div class="maps"><div class="map">{to_base64_img(u8, 2)}</div></div></div>'
    )

    # Each layer
    for i, (name, shape) in enumerate(LAYER_INFO):
        path = os.path.join(fdir, f"layer_{i}.bin")
        data = read_bin(path)
        if len(shape) == 3:
            sections.append(render_feature_maps(name, data, shape))
        elif name == "Softmax":
            sections.append(render_softmax(data, true_label, predicted))
        else:
            sections.append(render_fc(name, data, shape[0]))

    # Build verdict
    correct = true_label == predicted
    verdict_cls = "correct" if correct else "wrong"
    true_name = CLASS_NAMES[true_label] if 0 <= true_label < 10 else "?"
    pred_name = CLASS_NAMES[predicted] if 0 <= predicted < 10 else "?"
    verdict = (
        f'<div class="verdict {verdict_cls}">'
        f'Predicted: <b>{pred_name}</b> | Ground Truth: <b>{true_name}</b>'
        f' {"&#10004;" if correct else "&#10008;"}</div>'
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Feature Map Visualizer</title>
<style>
body {{ background: #0f172a; color: #e5e7eb; font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; padding: 20px; }}
h1 {{ text-align: center; color: #7c9cff; }}
h3 {{ margin: 8px 0; color: #93c5fd; }}
.dim {{ color: #9ca3af; font-weight: normal; font-size: 0.85em; }}
.layer {{ background: #111827; border-radius: 8px; padding: 16px; margin: 16px 0; }}
.maps {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.map {{ text-align: center; }}
.map .label {{ font-size: 11px; color: #9ca3af; }}
.bars, .softmax {{ max-width: 600px; }}
.bar-row, .sm-row {{ display: flex; align-items: center; gap: 8px; margin: 2px 0; }}
.bar-idx {{ width: 24px; text-align: right; font-size: 12px; color: #9ca3af; }}
.sm-name {{ width: 100px; text-align: right; font-size: 13px; }}
.bar-bg {{ flex: 1; height: 16px; background: #1f2937; border-radius: 3px; overflow: hidden; }}
.bar-fill {{ height: 100%; background: #60a5fa; }}
.bar-val {{ width: 70px; font-size: 12px; text-align: right; font-family: monospace; }}
.highlight-true {{ background: rgba(34,197,94,0.12); border-radius: 4px; }}
.highlight-pred .bar-fill {{ background: #818cf8; }}
.verdict {{ text-align: center; font-size: 1.3em; padding: 16px; margin: 16px 0; border-radius: 8px; }}
.verdict.correct {{ background: rgba(34,197,94,0.2); color: #4ade80; }}
.verdict.wrong {{ background: rgba(248,113,113,0.2); color: #f87171; }}
</style></head><body>
<h1>Feature Map Visualizer</h1>
{verdict}
{"".join(sections)}
</body></html>"""

    out_path = os.path.join(os.path.dirname(fdir.rstrip("/")), "feature_maps.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Generated {out_path}")
    print(f"Predicted: {pred_name} | Ground Truth: {true_name} | {'Correct' if correct else 'Wrong'}")


if __name__ == "__main__":
    main()
