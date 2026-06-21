"""
Fixed-Point Contraction Sanity Check
======================================
Numerically verify L_E * L_Phi < 1 for the actual F1/F2 architecture
used in Paper 2B experiments (f1_1_validation.py).

Theorem 2A-2 requires this condition for fixed-point existence/uniqueness.
Remark C.1 gives a softmax closed-form (L_E = 1/(4*tau)) as a sufficient
example. Here we check the ACTUAL architecture used in experiments.
"""

import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)

DIM = 100
SIGNAL_DIM = 5

# ── Reconstruct actual F1Gate (Phi/encoder+gate) and F2 (evaluation) ──────
class F2(nn.Module):
    """Actual F2 from f1_1_validation.py"""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, x):
        return self.net(x)

class F1Gate(nn.Module):
    """Actual F1Gate from f1_1_validation.py — top-k hard gate"""
    def __init__(self, dim, k=5):
        super().__init__()
        self.k = k
        self.score = nn.Parameter(torch.zeros(dim))
        self.bias  = nn.Parameter(torch.zeros(dim))

    def get_salience(self, x):
        return x * torch.sigmoid(self.score) + self.bias

    def forward(self, x):
        sal = self.get_salience(x)
        topk_vals, _ = torch.topk(sal, self.k, dim=1)
        main_threshold = topk_vals[:, -1:].detach()
        gate_hard = (sal >= main_threshold).float()
        gate_soft = torch.sigmoid((sal - main_threshold) / 0.05)
        gate = gate_hard.detach() - gate_soft.detach() + gate_soft
        return x * gate

# ── Estimate Lipschitz constants empirically via finite differences ──────
def estimate_lipschitz(module, dim, n_samples=2000, eps=1e-3, scale=5.0):
    """
    Empirical Lipschitz estimate:
    L_hat = max over sampled pairs of ||f(x) - f(y)|| / ||x - y||
    Uses random pairs + local perturbation pairs for better coverage.
    """
    module.eval()
    max_ratio = 0.0
    with torch.no_grad():
        # Local perturbation pairs (more informative for Lipschitz estimate)
        for _ in range(n_samples):
            x = torch.randn(1, dim) * scale
            direction = torch.randn(1, dim)
            direction = direction / direction.norm()
            y = x + direction * eps

            fx = module(x)
            fy = module(y)

            num = (fy - fx).flatten().norm().item()
            den = (y - x).flatten().norm().item()
            if den > 1e-12:
                ratio = num / den
                max_ratio = max(max_ratio, ratio)

        # Also sample larger-distance pairs (global behavior)
        for _ in range(n_samples):
            x = torch.randn(1, dim) * scale
            y = torch.randn(1, dim) * scale
            fx = module(x)
            fy = module(y)
            num = (fy - fx).flatten().norm().item()
            den = (y - x).flatten().norm().item()
            if den > 1e-6:
                ratio = num / den
                max_ratio = max(max_ratio, ratio)
    module.train()
    return max_ratio

# ── Build untrained (random init) and trained models ──────────────────────
print("=== Lipschitz constant estimation ===\n")

print("--- Untrained (random init) ---")
phi_untrained = F1Gate(DIM, k=5)
f2_untrained  = F2(DIM)

L_phi_untrained = estimate_lipschitz(phi_untrained, DIM, scale=5.0)
L_e_untrained   = estimate_lipschitz(f2_untrained,  DIM, scale=1.0)  # F2 input is gated x, smaller effective range

print(f"  L_Phi (F1 gate)  ~ {L_phi_untrained:.4f}")
print(f"  L_E   (F2 eval)  ~ {L_e_untrained:.4f}")
print(f"  L_E * L_Phi      ~ {L_e_untrained * L_phi_untrained:.4f}")
print(f"  Contraction (< 1)? {'YES' if L_e_untrained * L_phi_untrained < 1 else 'NO'}")

# ── Now train them briefly on the actual task and re-check ───────────────
print("\n--- After training (2000 steps, actual task) ---")

def generate_data(batch_size=64, dim=DIM, noise_level=0.5):
    signal = torch.randn(batch_size, SIGNAL_DIM) * 5
    noise  = torch.randn(batch_size, dim - SIGNAL_DIM) * noise_level
    x = torch.cat([signal, noise], dim=1)
    y = (signal.sum(dim=1) > 0).long()
    return x, y

phi = F1Gate(DIM, k=5)
f2  = F2(DIM)
opt = torch.optim.Adam(list(phi.parameters()) + list(f2.parameters()), lr=3e-3)

import torch.nn.functional as F
for step in range(2000):
    x, y = generate_data()
    gated = phi(x)
    logits = f2(gated)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad(); loss.backward(); opt.step()

L_phi_trained = estimate_lipschitz(phi, DIM, scale=5.0)
L_e_trained   = estimate_lipschitz(f2,  DIM, scale=1.0)

print(f"  L_Phi (F1 gate)  ~ {L_phi_trained:.4f}")
print(f"  L_E   (F2 eval)  ~ {L_e_trained:.4f}")
print(f"  L_E * L_Phi      ~ {L_e_trained * L_phi_trained:.4f}")
print(f"  Contraction (< 1)? {'YES' if L_e_trained * L_phi_trained < 1 else 'NO'}")

# ── Theoretical check: is F1 gate actually 1-Lipschitz as claimed? ────────
print("\n--- Theoretical claim check: is top-k gate 1-Lipschitz? ---")
print("  Appendix C Step 1 claims: 'zeroing components is non-expansive'")
print("  This means ||G(z) - G(z')|| <= ||z - z'|| for the GATING operation alone.")
print("  But F1Gate here also has trainable score/bias BEFORE the gate.")
print("  The full Phi = gate(score(x)) may NOT be 1-Lipschitz if score/bias")
print("  amplify the input. Let's check the raw gate-only Lipschitz separately.")

class PureGate(nn.Module):
    """Just the top-k zeroing operation, no learned salience"""
    def __init__(self, k=5):
        super().__init__()
        self.k = k
    def forward(self, x):
        topk_vals, _ = torch.topk(x.abs(), self.k, dim=1)
        threshold = topk_vals[:, -1:]
        mask = (x.abs() >= threshold).float()
        return x * mask

pure_gate = PureGate(k=5)
L_pure = estimate_lipschitz(pure_gate, DIM, scale=5.0)
print(f"\n  L(pure top-k zeroing) ~ {L_pure:.4f}  (claim: should be <= 1)")
