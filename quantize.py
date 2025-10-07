# quant_nbit.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# ---------- core helpers (uniform quant) ----------
def qparams_from_minmax(xmin, xmax, n_bits=8, unsigned=False, eps=1e-12):
    """
    Returns (scale, zero_point, qmin, qmax) for uniform quant.
    - unsigned=True  -> [0, 2^b - 1]
    - unsigned=False -> symmetric int range [-2^(b-1)+1, 2^(b-1)-1]
    """
    if unsigned:
        qmin, qmax = 0, (1 << n_bits) - 1
        # (common for post-ReLU) ensure non-negative min for tighter range
        xmin = torch.zeros_like(xmin)
        scale = (xmax - xmin).clamp_min(eps) / float(qmax - qmin)
        zp = torch.round(-xmin / scale).clamp(qmin, qmax)
    else:
        qmax = (1 << (n_bits - 1)) - 1
        qmin = -qmax
        max_abs = torch.max(xmin.abs(), xmax.abs()).clamp_min(eps)
        scale = max_abs / float(qmax)
        zp = torch.zeros_like(scale)
    return scale, zp, int(qmin), int(qmax)

def quantize(x, scale, zp, qmin, qmax):
    q = torch.round(x / scale + zp)
    return q.clamp(qmin, qmax)

def dequantize(q, scale, zp):
    return (q - zp) * scale

# ---------- activation fake-quant (with calibration then freeze) ----------
class ActFakeQuant(nn.Module):
    """
    Per-tensor activation fake-quant with configurable bits.
    Intended to be placed AFTER ReLU -> use unsigned=True.
    """
    def __init__(self, n_bits=8, unsigned=True):
        super().__init__()
        self.n_bits = n_bits
        self.unsigned = unsigned
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zp", torch.tensor(0.0))
        self.frozen = False
        self.qmin, self.qmax = None, None

    @torch.no_grad()
    def observe(self, x):
        self.min_val = torch.minimum(self.min_val, x.min())
        self.max_val = torch.maximum(self.max_val, x.max())

    @torch.no_grad()
    def freeze(self):
        scale, zp, qmin, qmax = qparams_from_minmax(
            self.min_val, self.max_val, n_bits=self.n_bits, unsigned=self.unsigned
        )
        self.scale.copy_(scale)
        self.zp.copy_(zp)
        self.qmin, self.qmax = qmin, qmax
        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            self.observe(x)
            return x
        q = quantize(x, self.scale, self.zp, self.qmin, self.qmax)
        return dequantize(q, self.scale, self.zp)

# ---------- weight fake-quant wrappers (freeze-from-weights) ----------
class QuantConv2d(nn.Conv2d):
    """
    Per-tensor symmetric int quantization for weights with configurable bits.
    We compute/freeze params once from trained weights (PTQ).
    """
    def __init__(self, *args, weight_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.register_buffer("w_scale", torch.tensor(1.0))
        self.register_buffer("w_zp", torch.tensor(0.0))
        self.frozen = False
        self.qmin = None
        self.qmax = None

    @torch.no_grad()
    def freeze(self):
        w = self.weight.detach().cpu()
        w_min, w_max = w.min(), w.max()
        scale, zp, qmin, qmax = qparams_from_minmax(
            w_min, w_max, n_bits=self.weight_bits, unsigned=False
        )
        self.w_scale.copy_(scale)
        self.w_zp.copy_(zp)
        self.qmin, self.qmax = qmin, qmax
        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        q = quantize(self.weight, self.w_scale, self.w_zp, self.qmin, self.qmax)
        w_dq = dequantize(q, self.w_scale, self.w_zp)
        return F.conv2d(x, w_dq, self.bias, self.stride, self.padding, self.dilation, self.groups)

class QuantLinear(nn.Linear):
    def __init__(self, *args, weight_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.register_buffer("w_scale", torch.tensor(1.0))
        self.register_buffer("w_zp", torch.tensor(0.0))
        self.frozen = False
        self.qmin = None
        self.qmax = None

    @torch.no_grad()
    def freeze(self):
        w = self.weight.detach().cpu()
        w_min, w_max = w.min(), w.max()
        scale, zp, qmin, qmax = qparams_from_minmax(
            w_min, w_max, n_bits=self.weight_bits, unsigned=False
        )
        self.w_scale.copy_(scale)
        self.w_zp.copy_(zp)
        self.qmin, self.qmax = qmin, qmax
        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            return F.linear(x, self.weight, self.bias)
        q = quantize(self.weight, self.w_scale, self.w_zp, self.qmin, self.qmax)
        w_dq = dequantize(q, self.w_scale, self.w_zp)
        return F.linear(x, w_dq, self.bias)

# ---------- model surgery with user-selected bits ----------
def swap_to_quant_modules(model, weight_bits=8, act_bits=8, activations_unsigned=True):
    """
    - Replace every Conv2d/Linear with Quant* using weight_bits.
    - Replace every ReLU with Sequential(ReLU, ActFakeQuant(act_bits)).
    """
    for name, m in list(model.named_children()):
        swap_to_quant_modules(m, weight_bits, act_bits, activations_unsigned)

        if isinstance(m, nn.Conv2d):
            q = QuantConv2d(
                m.in_channels, m.out_channels, m.kernel_size,
                stride=m.stride, padding=m.padding, dilation=m.dilation,
                groups=m.groups, bias=(m.bias is not None),
                weight_bits=weight_bits
            )
            q.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                q.bias.data.copy_(m.bias.data)
            setattr(model, name, q)

        elif isinstance(m, nn.Linear):
            q = QuantLinear(m.in_features, m.out_features, bias=(m.bias is not None), weight_bits=weight_bits)
            q.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                q.bias.data.copy_(m.bias.data)
            setattr(model, name, q)

        elif isinstance(m, nn.ReLU):
            seq = nn.Sequential(OrderedDict([
                ("relu", nn.ReLU(inplace=getattr(m, "inplace", False))),
                ("aq", ActFakeQuant(n_bits=act_bits, unsigned=activations_unsigned)),
            ]))
            setattr(model, name, seq)

def freeze_all_quant(model):
    """
    Freeze weights and activations (finalize scales/ZPs) after calibration.
    """
    for mod in model.modules():
        if isinstance(mod, (QuantConv2d, QuantLinear)):
            mod.freeze()
        if isinstance(mod, nn.Sequential):
            for sub in mod.modules():
                if isinstance(sub, ActFakeQuant):
                    sub.freeze()

def model_size_bytes_fp32(model):
    """Total size of all parameters if stored as FP32 (4 bytes each)."""
    total = 0
    for p in model.parameters():
        total += p.numel() * 4
    return total

def model_size_bytes_quant(model, weight_bits=8):
    """Total size if all weights were stored as intN, biases stay FP32."""
    total = 0
    for name, p in model.named_parameters():
        if "weight" in name:
            total += p.numel() * weight_bits // 8  # intN
        elif "bias" in name:
            total += p.numel() * 4                 # keep biases FP32
    return total

def print_compression(model, weight_bits=8):
    fp32_size = model_size_bytes_fp32(model)
    quant_size = model_size_bytes_quant(model, weight_bits)
    
    # Calculate weights sizes
    fp32_weights_size = sum(p.numel() * p.element_size() for p in model.parameters())
    quant_weights_size = sum(p.numel() for p in model.parameters()) * (weight_bits / 8)
    
    # Approximate activations sizes as remainder
    fp32_activations_size = fp32_size - fp32_weights_size
    quant_activations_size = quant_size - quant_weights_size
    
    # Compute ratios
    overall_ratio = fp32_size / max(quant_size, 1)
    weights_ratio = fp32_weights_size / max(quant_weights_size, 1)
    activations_ratio = fp32_activations_size / max(quant_activations_size, 1)
    
    print("=== Compression Summary ===")
    print(f"FP32 model size:          {fp32_size / 1024 / 1024:.2f} MB")
    print(f"Quantized model size:     {quant_size / 1024 / 1024:.2f} MB (weights={weight_bits}-bit)")
    print(f"Compression ratio overall: {overall_ratio:.2f}x")
    print(f"Compression ratio weights: {weights_ratio:.2f}x")
    print(f"Compression ratio activations: {activations_ratio:.2f}x")
    print("===========================")
