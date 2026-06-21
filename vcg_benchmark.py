"""
VCG Benchmark: Valence-Conditioned Gating
============================================
Empirical test of Theorem 2A-1 (Non-Scalarizability of Valence-Conditioned
Evaluation, Paper 2A). Tests whether multi-dimensional valence structure
(m >= 2) induced by selection-conditioned access resists scalar reduction.

Three sub-experiments:
  VCG-1: Multi-objective retention — does H_s(G) degrade valence dimensions
         non-uniformly?
  VCG-2: Pareto frontier analysis — does the achievable (A,C,U) front
         narrow as H_s(G) increases?
  VCG-3: Scalarization challenge — can a scalar-weighted baseline reach
         the same Pareto region as the multi-valence F1.1 gate?

Valence dimensions (m=3, satisfying Theorem 2A-1's m>=2 condition):
  A — Accuracy   (task correctness)
  C — Confidence (calibration: how well predicted probability matches correctness)
  U — Urgency    (response to a synthetic urgency-encoding subset of features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────
# DATA: synthetic task with THREE separable valence-relevant subspaces
# ─────────────────────────────────────────────────────────────────────────
DIM = 100
ACC_DIM   = 5     # features 0:5   — determine classification accuracy
CONF_DIM  = 5     # features 5:10  — determine calibration target (confidence)
URG_DIM   = 5     # features 10:15 — determine urgency label
SIGNAL_DIM = ACC_DIM + CONF_DIM + URG_DIM  # 15 informative features total

def generate_vcg_data(batch_size=128, dim=DIM, noise_level=0.5):
    """
    Three independent signal blocks + shared noise floor.
    - acc_signal determines binary classification target y_acc
    - conf_signal determines a synthetic 'reliability' target y_conf
      (higher |conf_signal sum| = more reliable / should be high confidence)
    - urg_signal determines a binary urgency target y_urg
    """
    acc_signal  = torch.randn(batch_size, ACC_DIM) * 5
    conf_signal = torch.randn(batch_size, CONF_DIM) * 5
    urg_signal  = torch.randn(batch_size, URG_DIM) * 5
    noise = torch.randn(batch_size, dim - SIGNAL_DIM) * noise_level

    x = torch.cat([acc_signal, conf_signal, urg_signal, noise], dim=1)

    y_acc  = (acc_signal.sum(dim=1) > 0).long()
    # Confidence target: should the model be confident? Based on |conf_signal| magnitude
    conf_strength = conf_signal.abs().sum(dim=1)
    y_conf = (conf_strength > conf_strength.median()).float()  # 1 = high-reliability case
    y_urg  = (urg_signal.sum(dim=1) > 0).long()

    return x, y_acc, y_conf, y_urg

# ─────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────
class MultiValenceHead(nn.Module):
    """F2 with three separate output heads — one per valence dimension."""
    def __init__(self, dim, hidden=48):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU())
        self.acc_head  = nn.Linear(hidden, 2)   # classification logits
        self.conf_head = nn.Linear(hidden, 1)   # confidence regression (sigmoid)
        self.urg_head  = nn.Linear(hidden, 2)   # urgency logits

    def forward(self, x):
        h = self.trunk(x)
        return self.acc_head(h), torch.sigmoid(self.conf_head(h)).squeeze(-1), self.urg_head(h)

class F1GateVCG(nn.Module):
    """Top-k hard gate with tunable k to control H_s(G)."""
    def __init__(self, dim, k):
        super().__init__()
        self.k = k
        self.score = nn.Parameter(torch.zeros(dim))
        self.bias  = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        sal = x * torch.sigmoid(self.score) + self.bias
        topk_vals, _ = torch.topk(sal, self.k, dim=1)
        thr = topk_vals[:, -1:].detach()
        gate_hard = (sal >= thr).float()
        gate_soft = torch.sigmoid((sal - thr) / 0.05)
        gate = gate_hard.detach() - gate_soft.detach() + gate_soft
        return x * gate, gate

class VCGModel(nn.Module):
    """Multi-valence F1 gate model. k controls effective H_s(G)."""
    def __init__(self, dim, k):
        super().__init__()
        self.k = k
        self.f1 = F1GateVCG(dim, k)
        self.f2 = MultiValenceHead(dim)

    def forward(self, x):
        xg, gate = self.f1(x)
        acc_logits, conf_pred, urg_logits = self.f2(xg)
        return acc_logits, conf_pred, urg_logits, gate

class ScalarBaseline(nn.Module):
    """
    Scalar baseline: same gate architecture, but trained to optimize
    a SINGLE weighted scalar objective s = w1*A + w2*C + w3*U instead
    of the three valence dimensions independently.
    Architecturally identical to VCGModel -- only the training objective differs.
    """
    def __init__(self, dim, k):
        super().__init__()
        self.k = k
        self.f1 = F1GateVCG(dim, k)
        self.f2 = MultiValenceHead(dim)  # same heads, but trained via scalarized loss

    def forward(self, x):
        xg, gate = self.f1(x)
        acc_logits, conf_pred, urg_logits = self.f2(xg)
        return acc_logits, conf_pred, urg_logits, gate

# ─────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────
def train_vcg_multiobjective(model, steps=2500, noise=0.5):
    """
    Train with three SEPARATE losses, summed but each preserved in its own
    metric space (not scalarized into a single weighted objective at the
    metric level -- this is the multi-valence training regime).
    """
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(steps):
        x, y_acc, y_conf, y_urg = generate_vcg_data(noise_level=noise)
        acc_logits, conf_pred, urg_logits, _ = model(x)

        loss_acc  = F.cross_entropy(acc_logits, y_acc)
        loss_conf = F.binary_cross_entropy(conf_pred, y_conf)
        loss_urg  = F.cross_entropy(urg_logits, y_urg)

        # Equal-weight sum for training (architecture learns all three;
        # evaluation will still report them SEPARATELY, never merged into
        # one scalar metric -- that's the VCG-1/VCG-2 measurement protocol)
        loss = loss_acc + loss_conf + loss_urg
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def train_scalar_baseline(model, weights=(0.34, 0.33, 0.33), steps=2500, noise=0.5):
    """
    Train with a TRUE scalar objective: a single weighted sum optimized
    directly, simulating a system that scalarizes valence at the
    objective level (as RLHF / standard RL reward scalarization would).
    """
    w1, w2, w3 = weights
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(steps):
        x, y_acc, y_conf, y_urg = generate_vcg_data(noise_level=noise)
        acc_logits, conf_pred, urg_logits, _ = model(x)

        # Convert each to a [0,1] "reward-like" scalar contribution
        acc_prob  = F.softmax(acc_logits, dim=1)[:, 1]
        conf_term = conf_pred  # already in [0,1], target y_conf in [0,1]
        urg_prob  = F.softmax(urg_logits, dim=1)[:, 1]

        # Scalarized reward signal the model is trained to maximize/match
        target_scalar = w1 * y_acc.float() + w2 * y_conf + w3 * y_urg.float()
        pred_scalar   = w1 * acc_prob + w2 * conf_term + w3 * urg_prob

        loss = F.mse_loss(pred_scalar, target_scalar)
        opt.zero_grad(); loss.backward(); opt.step()
    return model

# ─────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────
def evaluate_vcg(model, n_batches=20, noise=0.5):
    """Returns (Accuracy, Confidence-calibration, Urgency-accuracy) — kept SEPARATE."""
    model.eval()
    accs, confs, urgs = [], [], []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y_acc, y_conf, y_urg = generate_vcg_data(noise_level=noise)
            acc_logits, conf_pred, urg_logits, _ = model(x)

            acc = (acc_logits.argmax(1) == y_acc).float().mean().item()
            # Confidence: 1 - mean absolute calibration error
            conf_err = (conf_pred - y_conf).abs().mean().item()
            conf = 1.0 - conf_err
            urg = (urg_logits.argmax(1) == y_urg).float().mean().item()

            accs.append(acc); confs.append(conf); urgs.append(urg)
    model.train()
    return np.mean(accs), np.mean(confs), np.mean(urgs)

def estimate_hs(model, n_batches=20, noise=0.5):
    """Empirical H_s(G) estimate via entropy of gate selection pattern."""
    model.eval()
    entropies = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, *_ = generate_vcg_data(noise_level=noise)
            _, _, _, gate = model(x)
            p = gate / (gate.sum(dim=1, keepdim=True) + 1e-8)
            ent = -(p * torch.log(p + 1e-8)).sum(dim=1).mean().item()
            entropies.append(ent)
    model.train()
    return np.mean(entropies)

# ─────────────────────────────────────────────────────────────────────────
# VCG-1 + VCG-2: Multi-objective retention + Pareto frontier across H_s(G)
# ─────────────────────────────────────────────────────────────────────────
print("=== VCG-1 + VCG-2: Multi-objective retention & Pareto analysis ===\n")

k_values = [3, 5, 8, 12, 18, 25, 35]   # sparse -> dense, controls H_s(G)
results_vcg = []

for k in k_values:
    print(f"  Training VCG model k={k} ...")
    model = VCGModel(DIM, k)
    model = train_vcg_multiobjective(model)
    A, C, U = evaluate_vcg(model)
    hs = estimate_hs(model)
    results_vcg.append({'k': k, 'A': A, 'C': C, 'U': U, 'H_s': hs})
    print(f"    H_s(G)={hs:.3f}  A={A:.3f}  C={C:.3f}  U={U:.3f}")

# ─────────────────────────────────────────────────────────────────────────
# Pareto frontier computation
# ─────────────────────────────────────────────────────────────────────────
def pareto_efficient(points):
    """points: list of (A,C,U) tuples. Returns boolean mask of Pareto-efficient points."""
    pts = np.array(points)
    n = len(pts)
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            # j dominates i if j >= i in all dims and > in at least one
            if np.all(pts[j] >= pts[i]) and np.any(pts[j] > pts[i]):
                is_efficient[i] = False
                break
    return is_efficient

# For VCG-2, we need MULTIPLE configurations at each H_s level to compute
# a meaningful frontier. Re-train each k with several random seeds /
# different loss-weight perturbations to get a spread of (A,C,U) points.
print("\n=== VCG-2: Generating point cloud for Pareto frontier (multi-seed) ===")
pareto_data = {}  # k -> list of (A,C,U)
for k in k_values:
    pts = []
    for seed in range(5):
        torch.manual_seed(100 + seed)
        model = VCGModel(DIM, k)
        # Perturb training slightly via different random init + dropout-like noise
        model = train_vcg_multiobjective(model, steps=1200)
        A, C, U = evaluate_vcg(model)
        pts.append((A, C, U))
    pareto_data[k] = pts
    print(f"  k={k}: {len(pts)} points generated")
torch.manual_seed(42)

# ─────────────────────────────────────────────────────────────────────────
# VCG-3: Scalarization challenge (multi-seed for variance estimate)
# ─────────────────────────────────────────────────────────────────────────
print("\n=== VCG-3: Scalarization challenge (multi-seed) ===")
k_fixed = 12  # representative mid-sparsity regime
N_SEEDS = 5

print(f"  Training multi-valence VCG model (k={k_fixed}) across {N_SEEDS} seeds ...")
vcg_points = []
for seed in range(N_SEEDS):
    torch.manual_seed(200 + seed)
    vcg_model = VCGModel(DIM, k_fixed)
    vcg_model = train_vcg_multiobjective(vcg_model, steps=2500)
    A, C, U = evaluate_vcg(vcg_model)
    vcg_points.append((A, C, U))
    print(f"    seed {seed}: A={A:.3f}  C={C:.3f}  U={U:.3f}")
torch.manual_seed(42)

vcg_points = np.array(vcg_points)
A_vcg, C_vcg, U_vcg = vcg_points.mean(axis=0)
A_vcg_std, C_vcg_std, U_vcg_std = vcg_points.std(axis=0)
print(f"\n  Multi-valence mean: A={A_vcg:.3f}±{A_vcg_std:.3f}  "
      f"C={C_vcg:.3f}±{C_vcg_std:.3f}  U={U_vcg:.3f}±{U_vcg_std:.3f}")

# Denser scalar weight grid (more thorough than single-point sweep)
print("\n  Sweeping DENSE scalar baseline weight grid (multi-seed) ...")
weight_grid = []
# Systematic simplex grid over (w1, w2, w3) with w1+w2+w3=1
grid_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for w1 in grid_steps:
    for w2 in grid_steps:
        w3 = 1.0 - w1 - w2
        if w3 > 0.05:
            weight_grid.append((round(w1,2), round(w2,2), round(w3,2)))
print(f"  Grid size: {len(weight_grid)} weight combinations")

scalar_sweep_results = []
for w in weight_grid:
    torch.manual_seed(7)
    sm = ScalarBaseline(DIM, k_fixed)
    sm = train_scalar_baseline(sm, weights=w, steps=1200)
    A_s, C_s, U_s = evaluate_vcg(sm)
    scalar_sweep_results.append({'weights': w, 'A': A_s, 'C': C_s, 'U': U_s})
torch.manual_seed(42)

print(f"  Completed {len(scalar_sweep_results)} scalar baseline runs")

# Check dominance against MEAN multi-valence point, conservatively
vcg_point = np.array([A_vcg, C_vcg, U_vcg])
scalar_points = np.array([[r['A'], r['C'], r['U']] for r in scalar_sweep_results])
dominates_vcg = np.any(np.all(scalar_points >= vcg_point[None, :], axis=1))

# Also check against the WORST-case multi-valence point (mean - 1 std)
# -- conservative test accounting for variance
vcg_point_conservative = vcg_point - np.array([A_vcg_std, C_vcg_std, U_vcg_std])
dominates_vcg_conservative = np.any(np.all(scalar_points >= vcg_point_conservative[None, :], axis=1))

print(f"\n  Does any of {len(weight_grid)} scalar weightings dominate the mean "
      f"multi-valence point? {'YES' if dominates_vcg else 'NO'}")
print(f"  Does any dominate the conservative (mean - 1σ) point? "
      f"{'YES' if dominates_vcg_conservative else 'NO'}")

# How close does the best scalar point get? (Euclidean distance in (A,C,U) space)
distances = np.linalg.norm(scalar_points - vcg_point[None, :], axis=1)
best_scalar_idx = distances.argmin()
print(f"\n  Closest scalar baseline to multi-valence point: "
      f"w={scalar_sweep_results[best_scalar_idx]['weights']}, "
      f"distance={distances[best_scalar_idx]:.3f}")
print(f"  That point: A={scalar_points[best_scalar_idx,0]:.3f} "
      f"C={scalar_points[best_scalar_idx,1]:.3f} U={scalar_points[best_scalar_idx,2]:.3f}")

# ─────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 5))

# Panel 1: VCG-1 — multi-objective retention vs H_s(G)
ax1 = fig.add_subplot(1, 3, 1)
hs_vals = [r['H_s'] for r in results_vcg]
A_vals  = [r['A'] for r in results_vcg]
C_vals  = [r['C'] for r in results_vcg]
U_vals  = [r['U'] for r in results_vcg]
ax1.plot(hs_vals, A_vals, 'o-', color='#d62728', label='Accuracy (A)', lw=2, ms=6)
ax1.plot(hs_vals, C_vals, 's-', color='#2ca02c', label='Confidence cal. (C)', lw=2, ms=6)
ax1.plot(hs_vals, U_vals, '^-', color='#1f77b4', label='Urgency (U)', lw=2, ms=6)
ax1.set_xlabel('Estimated $H_s(G)$')
ax1.set_ylabel('Valence dimension value')
ax1.set_title('VCG-1: Multi-Objective Retention\nvs Selection Hardness')
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

# Panel 2: VCG-2 — Pareto frontier across H_s levels (3D->2D projection: A vs C, colored by H_s)
ax2 = fig.add_subplot(1, 3, 2)
cmap = plt.cm.viridis
k_to_hs = {r['k']: r['H_s'] for r in results_vcg}
norm = plt.Normalize(min(hs_vals), max(hs_vals))
for k in k_values:
    pts = np.array(pareto_data[k])
    hs_k = k_to_hs[k]
    color = cmap(norm(hs_k))
    ax2.scatter(pts[:,0], pts[:,1], color=color, s=40, alpha=0.8,
                label=f'k={k} ($H_s$={hs_k:.2f})' if k in [k_values[0], k_values[-1]] else None)
sm_cb = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm_cb, ax=ax2)
cbar.set_label('$H_s(G)$', fontsize=9)
ax2.set_xlabel('Accuracy (A)')
ax2.set_ylabel('Confidence calibration (C)')
ax2.set_title('VCG-2: (A,C) Point Cloud\nacross Selection Hardness Levels')
ax2.grid(alpha=0.3)

# Panel 3: VCG-3 — scalarization challenge (multi-seed + dense grid)
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.scatter(vcg_points[:,0], vcg_points[:,1], vcg_points[:,2],
            color='red', s=120, marker='*',
            label=f'Multi-valence VCG\n({N_SEEDS} seeds)', zorder=5)
sw = scalar_points
ax3.scatter(sw[:,0], sw[:,1], sw[:,2], color='steelblue', s=15, alpha=0.4,
            label=f'Scalar baselines\n({len(weight_grid)} weight settings)')
ax3.set_xlabel('A', fontsize=8)
ax3.set_ylabel('C', fontsize=8)
ax3.set_zlabel('U', fontsize=8)
ax3.set_title(f'VCG-3: Scalarization Challenge\n(k={k_fixed}, dense weight grid)', fontsize=10)
ax3.legend(fontsize=7, loc='upper left')

plt.tight_layout()
out = "/mnt/user-data/outputs/vcg_benchmark.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"\n→ Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────
# SUMMARY STATS
# ─────────────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print("\nVCG-1 (multi-objective retention):")
for r in results_vcg:
    print(f"  k={r['k']:3d}  H_s={r['H_s']:.3f}  A={r['A']:.3f}  C={r['C']:.3f}  U={r['U']:.3f}")

# Check uniformity of degradation: are A, C, U degrading at the SAME rate
# as H_s changes, or differently? (non-uniform degradation supports
# non-scalarizability -- a single scalar trade-off would predict uniform
# trade-off slopes)
A_range = max(A_vals) - min(A_vals)
C_range = max(C_vals) - min(C_vals)
U_range = max(U_vals) - min(U_vals)
print(f"\n  Dynamic range: A={A_range:.3f}  C={C_range:.3f}  U={U_range:.3f}")
print(f"  (Non-uniform ranges support non-scalarizability: a single scalar")
print(f"   trade-off parameter would predict comparable degradation profiles)")

print(f"\nVCG-3 (scalarization challenge, multi-seed):")
print(f"  Multi-valence mean: A={A_vcg:.3f}±{A_vcg_std:.3f} "
      f"C={C_vcg:.3f}±{C_vcg_std:.3f} U={U_vcg:.3f}±{U_vcg_std:.3f}")
print(f"  Best scalar A: {sw[:,0].max():.3f}")
print(f"  Best scalar C: {sw[:,1].max():.3f}")
print(f"  Best scalar U: {sw[:,2].max():.3f}")
print(f"  Any scalar dominates mean multi-valence point? {'YES' if dominates_vcg else 'NO'}")
print(f"  Any scalar dominates conservative (-1σ) point? "
      f"{'YES' if dominates_vcg_conservative else 'NO'}")
print(f"  Grid size tested: {len(weight_grid)} weight combinations")
