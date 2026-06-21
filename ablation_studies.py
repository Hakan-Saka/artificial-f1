"""
Ablation Studies: F1.1 Component Isolation
=============================================
Isolates the contribution of three F1.1 components to accuracy AND robustness:
  1. Safety margin m       (F1.0 -> F1.1 extension)
  2. EXP-F built-in filter (EMA + Hysteresis + Variance norm)
  3. Gate sparsity k        (compute budget)

Reports BOTH accuracy contribution and robustness contribution
(noise-collapse threshold shift) for each component, in table form.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)

# ── Data ───────────────────────────────────────────────────────────────────
SIGNAL_DIM  = 5
DIM         = 100
TRAIN_NOISE = 0.5

def generate_data(batch_size=64, dim=DIM, noise_level=TRAIN_NOISE):
    signal = torch.randn(batch_size, SIGNAL_DIM) * 5
    noise  = torch.randn(batch_size, dim - SIGNAL_DIM) * noise_level
    x = torch.cat([signal, noise], dim=1)
    y = (signal.sum(dim=1) > 0).long()
    return x, y

# ── F2 ─────────────────────────────────────────────────────────────────────
class F2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, x): return self.net(x)

# ── Built-in filter (same as EXP-F) ────────────────────────────────────────
class PreSelectionFilter(nn.Module):
    def __init__(self, dim, ema_alpha=0.8, hyst_gap=0.05, var_norm=True, use_filter=True):
        super().__init__()
        self.dim = dim; self.ema_alpha = ema_alpha; self.hyst_gap = hyst_gap
        self.var_norm = var_norm; self.use_filter = use_filter
        self.register_buffer('ema_state', torch.zeros(dim))
        self.register_buffer('prev_selected', torch.zeros(dim))
        self.register_buffer('initialized', torch.tensor(False))

    def reset(self):
        self.ema_state.zero_(); self.prev_selected.zero_(); self.initialized.fill_(False)

    def forward(self, sal):
        if not self.use_filter or self.training:
            return sal
        sal_mean = sal.mean(dim=0)
        if not self.initialized:
            self.ema_state.copy_(sal_mean); self.initialized.fill_(True)
        self.ema_state = (self.ema_alpha * sal_mean + (1 - self.ema_alpha) * self.ema_state).detach()
        sal_smoothed = self.ema_alpha * sal + (1 - self.ema_alpha) * self.ema_state.unsqueeze(0)
        if self.var_norm:
            std = sal_smoothed.std(dim=1, keepdim=True).clamp(min=1e-6)
            sal_smoothed = sal_smoothed / std
        sal_range = (sal_smoothed.max(1, keepdim=True).values - sal_smoothed.min(1, keepdim=True).values).clamp(min=1e-6)
        hyst_boost = self.prev_selected.unsqueeze(0) * sal_range * self.hyst_gap
        return sal_smoothed + hyst_boost

    def update_selected(self, gate):
        if self.use_filter:
            self.prev_selected = (gate > 0.05).float().mean(dim=0).detach()

# ── F1Gate: ablatable (m and filter can be independently switched) ────────
class F1GateAblation(nn.Module):
    def __init__(self, dim, k=5, m=0, margin_weight=0.15, use_filter=False,
                 ema_alpha=0.8, hyst_gap=0.05, var_norm=True):
        super().__init__()
        self.k = k; self.m = m; self.margin_weight = margin_weight
        self.score = nn.Parameter(torch.zeros(dim))
        self.bias  = nn.Parameter(torch.zeros(dim))
        self.filter = PreSelectionFilter(dim, ema_alpha, hyst_gap, var_norm, use_filter)

    def get_salience(self, x):
        return x * torch.sigmoid(self.score) + self.bias

    def forward(self, x):
        sal_raw = self.get_salience(x)
        sal = self.filter(sal_raw)

        topk_vals, _ = torch.topk(sal, self.k, dim=1)
        main_threshold = topk_vals[:, -1:].detach()
        gate_hard = (sal >= main_threshold).float()
        gate_soft = torch.sigmoid((sal - main_threshold) / 0.05)
        gate = gate_hard.detach() - gate_soft.detach() + gate_soft

        if self.m > 0 and self.margin_weight > 0:
            topkm_vals, _ = torch.topk(sal, self.k + self.m, dim=1)
            margin_threshold = topkm_vals[:, -1:].detach()
            in_margin = ((sal >= margin_threshold) & (sal < main_threshold)).float()
            gate = gate + in_margin * self.margin_weight

        self.filter.update_selected(gate)
        return x * gate, gate

class F1ModelAblation(nn.Module):
    def __init__(self, dim, k=5, m=0, use_filter=False):
        super().__init__()
        self.k = k; self.m = m
        self.f1 = F1GateAblation(dim, k=k, m=m, use_filter=use_filter)
        self.f2 = F2(dim)
    def forward(self, x):
        xg, gate = self.f1(x)
        return self.f2(xg), gate
    def active_features(self): return self.k + self.m

# ── Training ───────────────────────────────────────────────────────────────
def train(model, steps=2000, noise=TRAIN_NOISE):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(steps):
        x, y = generate_data(noise_level=noise)
        logits = model(x)[0]
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def clean_accuracy(model, repeats=30):
    model.eval()
    with torch.no_grad():
        accs = []
        for _ in range(repeats):
            x, y = generate_data(noise_level=TRAIN_NOISE)
            logits = model(x)[0]
            accs.append((logits.argmax(1) == y).float().mean().item())
    model.train()
    return np.mean(accs)

def collapse_threshold(model, sigma_range=None, repeats=15, threshold=0.6):
    """
    Finds the noise level sigma at which accuracy drops below `threshold`.
    This is the empirical noise-collapse threshold (robustness metric).
    Higher = more robust.
    """
    if sigma_range is None:
        sigma_range = np.linspace(0.1, 6.0, 40)
    model.eval()
    if hasattr(model, 'f1') and hasattr(model.f1, 'filter'):
        model.f1.filter.reset()
    with torch.no_grad():
        for sigma in sigma_range:
            accs = []
            for _ in range(repeats):
                x, y = generate_data(noise_level=float(sigma))
                logits = model(x)[0]
                accs.append((logits.argmax(1) == y).float().mean().item())
            if np.mean(accs) < threshold:
                model.train()
                return float(sigma)
    model.train()
    return float(sigma_range[-1])  # never collapsed within range

# ── Ablation grid ────────────────────────────────────────────────────────
print("=== Ablation Studies: F1.1 Component Isolation ===\n")

K_BASE = 5

configs = [
    {"name": "F1.0 (base)",                  "k": K_BASE, "m": 0, "use_filter": False},
    {"name": "F1.0 + filter only",           "k": K_BASE, "m": 0, "use_filter": True},
    {"name": "F1.1 (m=3) only",              "k": K_BASE, "m": 3, "use_filter": False},
    {"name": "F1.1 (m=3) + filter [full]",   "k": K_BASE, "m": 3, "use_filter": True},
    {"name": "F1.1 (m=5) only",              "k": K_BASE, "m": 5, "use_filter": False},
    {"name": "Sparsity ablation: k=3, m=0",  "k": 3,      "m": 0, "use_filter": False},
    {"name": "Sparsity ablation: k=10, m=0", "k": 10,     "m": 0, "use_filter": False},
]

results = []
for cfg in configs:
    print(f"Training: {cfg['name']} ...")
    model = F1ModelAblation(DIM, k=cfg['k'], m=cfg['m'], use_filter=cfg['use_filter'])
    model = train(model)
    acc = clean_accuracy(model)
    sigma_c = collapse_threshold(model)
    compute = model.active_features()
    results.append({
        'name': cfg['name'], 'k': cfg['k'], 'm': cfg['m'],
        'filter': cfg['use_filter'], 'compute': compute,
        'accuracy': acc, 'collapse_sigma': sigma_c
    })
    print(f"    accuracy={acc:.3f}  collapse_sigma={sigma_c:.2f}  compute={compute}")

# ── Marginal contribution analysis ─────────────────────────────────────────
print("\n=== Marginal Contribution Analysis ===\n")

base       = next(r for r in results if r['name'] == "F1.0 (base)")
filt_only  = next(r for r in results if r['name'] == "F1.0 + filter only")
m_only     = next(r for r in results if r['name'] == "F1.1 (m=3) only")
full       = next(r for r in results if r['name'] == "F1.1 (m=3) + filter [full]")

# Marginal contribution of m (safety margin): F1.1(m=3) - F1.0
delta_m_acc   = m_only['accuracy'] - base['accuracy']
delta_m_sigma = m_only['collapse_sigma'] - base['collapse_sigma']

# Marginal contribution of filter alone: F1.0+filter - F1.0
delta_filter_acc   = filt_only['accuracy'] - base['accuracy']
delta_filter_sigma = filt_only['collapse_sigma'] - base['collapse_sigma']

# Combined (full F1.1+filter) vs base
delta_full_acc   = full['accuracy'] - base['accuracy']
delta_full_sigma = full['collapse_sigma'] - base['collapse_sigma']

# Interaction term: does combining give more/less than sum of parts?
additive_expected_acc   = delta_m_acc + delta_filter_acc
additive_expected_sigma = delta_m_sigma + delta_filter_sigma
interaction_acc   = delta_full_acc - additive_expected_acc
interaction_sigma = delta_full_sigma - additive_expected_sigma

print(f"Component: Safety margin m (F1.0 -> F1.1, m=3)")
print(f"  Δ accuracy        = {delta_m_acc:+.3f}")
print(f"  Δ collapse_sigma   = {delta_m_sigma:+.3f}")

print(f"\nComponent: Built-in filter alone (F1.0 -> F1.0+filter)")
print(f"  Δ accuracy        = {delta_filter_acc:+.3f}")
print(f"  Δ collapse_sigma   = {delta_filter_sigma:+.3f}")

print(f"\nComponent: Both combined (F1.0 -> F1.1+filter)")
print(f"  Δ accuracy        = {delta_full_acc:+.3f}")
print(f"  Δ collapse_sigma   = {delta_full_sigma:+.3f}")

print(f"\nInteraction term (combined - sum of individual marginal effects):")
print(f"  Δ accuracy interaction      = {interaction_acc:+.3f}")
print(f"  Δ collapse_sigma interaction = {interaction_sigma:+.3f}")
print(f"  (near zero = additive/independent contributions;")
print(f"   positive = synergistic; negative = redundant/conflicting)")

# ── Output table ─────────────────────────────────────────────────────────
print("\n=== Full Ablation Table ===\n")
header = f"{'Configuration':<32} {'k':>3} {'m':>3} {'filter':>7} {'compute':>8} {'accuracy':>9} {'collapse_σ':>11}"
print(header)
print("-" * len(header))
for r in results:
    print(f"{r['name']:<32} {r['k']:>3} {r['m']:>3} {str(r['filter']):>7} "
          f"{r['compute']:>8} {r['accuracy']:>9.3f} {r['collapse_sigma']:>11.2f}")

print("\n=== Done ===")
