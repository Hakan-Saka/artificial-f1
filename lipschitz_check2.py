"""
Diagnose the pure top-k gate Lipschitz issue.
The claim "zeroing is non-expansive" should hold IF the same components
are zeroed for both x and y. The discontinuity happens at the BOUNDARY
where top-k SET CHANGES between x and y (different components selected).
This is the real subtlety Appendix C glosses over.
"""

import torch
import torch.nn as nn

torch.manual_seed(42)

class PureGate(nn.Module):
    def __init__(self, k=5):
        super().__init__()
        self.k = k
    def forward(self, x):
        topk_vals, _ = torch.topk(x.abs(), self.k, dim=1)
        threshold = topk_vals[:, -1:]
        mask = (x.abs() >= threshold).float()
        return x * mask

gate = PureGate(k=5)
DIM = 100

# Case A: same top-k SET for x and y (small perturbation, no rank change)
print("=== Case A: small perturbation, SAME top-k set selected ===")
x = torch.randn(1, DIM) * 5
y = x + torch.randn(1, DIM) * 1e-4   # tiny perturbation
fx, fy = gate(x), gate(y)
num = (fy-fx).norm().item(); den=(y-x).norm().item()
print(f"  ||G(x)-G(y)|| / ||x-y|| = {num/den:.4f}")

# Case B: perturbation that FLIPS which component is in top-k
print("\n=== Case B: perturbation that swaps top-k membership (near boundary) ===")
x = torch.zeros(1, DIM)
x[0, :5] = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.001])  # 5th and 6th close
x[0, 5] = 6.0  # just below threshold — component 5 (idx5) barely excluded
y = x.clone()
y[0, 5] = 6.002  # tiny nudge flips component 5 INTO top-k, kicks out idx4
fx, fy = gate(x), gate(y)
num = (fy-fx).norm().item(); den=(y-x).norm().item()
print(f"  ||G(x)-G(y)|| / ||x-y|| = {num/den:.2f}")
print(f"  x active: {(fx[0]!=0).nonzero().flatten().tolist()}")
print(f"  y active: {(fy[0]!=0).nonzero().flatten().tolist()}")
print(f"  → This is where Lipschitz constant blows up: discrete SET membership")
print(f"    changes discontinuously, even though the gate VALUE per fixed")
print(f"    component is 1-Lipschitz (identity or zero).")

print("\n=== Conclusion ===")
print("The 'non-expansive' claim in Appendix C Step 1 is TRUE only within")
print("a fixed selection pattern (same k-subset active). At ranking-boundary")
print("crossings, G is discontinuous and NOT globally Lipschitz — this is")
print("precisely the mechanism EXP-1/EXP-2 (Paper 2B) call 'collapse'.")
print("The contraction proof (Theorem 2A-2) implicitly assumes a FIXED")
print("selection regime; it does not hold uniformly across ranking transitions.")
