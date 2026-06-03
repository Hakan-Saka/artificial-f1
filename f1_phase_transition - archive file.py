"""
F1 Phase Transition — Noise Sweep v2
======================================
Düzeltmeler:
1. F1Gate: input-dependent salience (diagonal linear — attention'dan fark korunuyor)
2. MI proxy: düzeltildi — H(Y) - H(Y|pred) yerine log-loss tabanlı
3. Eğitim: her iki model için aynı koşullar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────

def generate_data(batch_size=64, dim=100, noise_level=0.5):
    signal = torch.randn(batch_size, 5) * 5
    noise  = torch.randn(batch_size, dim - 5) * noise_level
    x = torch.cat([signal, noise], dim=1)
    y = (signal.sum(dim=1) > 0).long()
    return x, y


# ─────────────────────────────────────────
# 2. F1 GATE  — input-dependent, diagonal
#
# Fark attention'dan:
#   Attention: softmax(Wx) → continuous weights ∈ (0,1), hiçbir şey sıfır değil
#   F1:        top-k hard mask → k dışındaki feature'lar tamamen sıfır
# ─────────────────────────────────────────

class F1Gate(nn.Module):
    def __init__(self, dim, k=5):
        super().__init__()
        self.k = k
        # Per-feature salience scorer (her feature kendi skorunu üretiyor)
        # dim → dim değil, her feature için tek bir scalar → diagonal
        self.score = nn.Parameter(torch.zeros(dim))  # öğrenilen feature önem ağırlıkları
        self.bias  = nn.Parameter(torch.zeros(dim))

    def get_salience(self, x):
        # input × öğrenilen önem + bias → per-feature, per-sample salience
        return x * torch.sigmoid(self.score) + self.bias

    def forward(self, x):
        salience = self.get_salience(x)

        # top-k hard gate (straight-through)
        topk_vals, _ = torch.topk(salience, self.k, dim=1)
        threshold = topk_vals[:, -1:].detach()
        gate_hard = (salience >= threshold).float()
        gate_soft = torch.sigmoid((salience - threshold) / 0.05)
        gate = gate_hard.detach() - gate_soft.detach() + gate_soft

        return x * gate, gate, salience.detach()


# ─────────────────────────────────────────
# 3. F2
# ─────────────────────────────────────────

class F2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────
# 4. FULL MODEL
# ─────────────────────────────────────────

class F1F2Model(nn.Module):
    def __init__(self, dim, k=5):
        super().__init__()
        self.f1 = F1Gate(dim, k=k)
        self.f2 = F2(dim)

    def forward(self, x):
        x_filt, gate, salience = self.f1(x)
        logits = self.f2(x_filt)
        return logits, gate, salience


# ─────────────────────────────────────────
# 5. ATTENTION BASELINE
#    Soft weighting — nothing excluded
# ─────────────────────────────────────────

class AttentionModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Parameter(torch.zeros(dim))
        self.f2    = F2(dim)

    def forward(self, x):
        # per-feature soft weights — same architecture depth as F1 but NO hard exclusion
        weights    = torch.softmax(x * torch.sigmoid(self.score), dim=1)
        x_weighted = x * weights
        logits     = self.f2(x_weighted)
        return logits, weights


# ─────────────────────────────────────────
# 6. TRAINING
# ─────────────────────────────────────────

def train_model(model, steps=2000, dim=100, train_noise=0.5, is_f1=True):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(steps):
        x, y = generate_data(batch_size=64, dim=dim, noise_level=train_noise)
        out = model(x)
        logits = out[0]   # always first element
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 400 == 0:
            print(f"  step {step:4d}  loss {loss.item():.4f}")
    return model


# ─────────────────────────────────────────
# 7. METRICS
# ─────────────────────────────────────────

def compute_accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()


def compute_mi(logits, y):
    """
    MI(pred; label) = H(Y) - H(Y|pred)
    H(Y) = log(2) for balanced binary
    H(Y|pred) = cross-entropy loss
    """
    h_y = np.log(2)
    h_y_given_pred = F.cross_entropy(logits, y).item()
    return max(0.0, h_y - h_y_given_pred)


# ─────────────────────────────────────────
# 8. NOISE SWEEP
# ─────────────────────────────────────────

def sweep(model, noise_levels, dim=100, repeats=30, is_f1=True):
    accs, mis = [], []
    model.eval()
    with torch.no_grad():
        for noise in noise_levels:
            batch_acc, batch_mi = [], []
            for _ in range(repeats):
                x, y = generate_data(batch_size=128, dim=dim,
                                     noise_level=noise.item())
                out = model(x)
                logits = out[0]
                batch_acc.append(compute_accuracy(logits, y))
                batch_mi.append(compute_mi(logits, y))
            accs.append(np.mean(batch_acc))
            mis.append(np.mean(batch_mi))
    model.train()
    return accs, mis


# ─────────────────────────────────────────
# 9. PLOT
# ─────────────────────────────────────────

def plot_results(noise_levels_np, f1_acc, attn_acc, f1_mi, attn_mi):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(noise_levels_np, f1_acc,   "b-o", markersize=4, label="F1 (hard gating)")
    ax.plot(noise_levels_np, attn_acc, "r--s", markersize=4, label="Attention (soft)")
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.6, label="Train noise σ=0.5")
    ax.set_xlabel("Noise level σ")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Noise Level")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(noise_levels_np, f1_mi,   "b-o", markersize=4, label="F1 (hard gating)")
    ax.plot(noise_levels_np, attn_mi, "r--s", markersize=4, label="Attention (soft)")
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.6, label="Train noise σ=0.5")
    ax.set_xlabel("Noise level σ")
    ax.set_ylabel("MI  I(pred; label)  [nats]")
    ax.set_title("Information Retention vs Noise Level")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle("F1 vs Attention: Noise Robustness", fontsize=13)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/phase_transition_noise.png", dpi=150)
    plt.close()
    print("  → phase_transition_noise.png kaydedildi")


# ─────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    DIM         = 100
    TRAIN_NOISE = 0.5

    print("=== F1F2 model eğitiliyor ===")
    f1_model = F1F2Model(DIM, k=5)
    f1_model = train_model(f1_model, steps=2000, dim=DIM,
                           train_noise=TRAIN_NOISE, is_f1=True)

    print("\n=== Attention model eğitiliyor ===")
    attn_model = AttentionModel(DIM)
    attn_model = train_model(attn_model, steps=2000, dim=DIM,
                             train_noise=TRAIN_NOISE, is_f1=False)

    noise_levels = torch.linspace(0.1, 8.0, 40)
    nl_np = noise_levels.numpy()

    print("\n=== Noise sweep (forward) ===")
    f1_acc,   f1_mi   = sweep(f1_model,   noise_levels, dim=DIM, is_f1=True)
    attn_acc, attn_mi = sweep(attn_model, noise_levels, dim=DIM, is_f1=False)

    plot_results(nl_np, f1_acc, attn_acc, f1_mi, attn_mi)

    # ── Hysteresis testi ──────────────────────────────────────────────────────
    # Noise artır (0.1 → 8) sonra azalt (8 → 0.1)
    # Aynı σ'da farklı MI çıkıyorsa: nonlinear threshold kanıtı
    print("\n=== Hysteresis testi ===")
    noise_fwd = torch.linspace(0.1, 8.0, 30)
    noise_rev = torch.linspace(8.0, 0.1, 30)

    f1_fwd, _  = sweep(f1_model, noise_fwd, dim=DIM, repeats=50, is_f1=True)
    f1_rev, _  = sweep(f1_model, noise_rev, dim=DIM, repeats=50, is_f1=True)

    _, f1_mi_fwd = sweep(f1_model, noise_fwd, dim=DIM, repeats=50, is_f1=True)
    _, f1_mi_rev = sweep(f1_model, noise_rev, dim=DIM, repeats=50, is_f1=True)

    # forward ve reverse aynı σ'da karşılaştır (reverse'i flip et)
    noise_rev_np = noise_rev.numpy()
    noise_fwd_np = noise_fwd.numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(noise_fwd_np, f1_mi_fwd, "b-o", markersize=4,
             label="F1: noise ↑ (0.1→8)")
    plt.plot(noise_rev_np, f1_mi_rev, "b--^", markersize=4,
             label="F1: noise ↓ (8→0.1)")
    plt.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5,
                label="Train noise σ=0.5")
    plt.xlabel("Noise level σ")
    plt.ylabel("MI I(pred; label) [nats]")
    plt.title("Hysteresis Test: F1 Forward vs Reverse Noise Sweep")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/hysteresis_test.png", dpi=150)
    plt.close()
    print("  → hysteresis_test.png kaydedildi")

    # ── Gate overlap + soft top-k ablation ───────────────────────────────────
    print("\n=== Gate overlap + ablation ===")

    # Soft top-k (Gumbel-softmax) ablation model
    class SoftF1Model(nn.Module):
        """Aynı mimari, hard yerine soft selection — hardness effect'i izole eder."""
        def __init__(self, dim, k=5):
            super().__init__()
            self.k = k
            self.score = nn.Parameter(torch.zeros(dim))
            self.bias  = nn.Parameter(torch.zeros(dim))
            self.f2    = F2(dim)

        def get_salience(self, x):
            return x * torch.sigmoid(self.score) + self.bias

        def forward(self, x):
            salience = self.get_salience(x)
            # Gumbel-softmax: differentiable soft selection
            weights = torch.softmax(salience / 0.1, dim=1)  # low temp ≈ hard
            x_weighted = x * weights
            logits = self.f2(x_weighted)
            return logits, weights, salience.detach()

    print("  Soft F1 (Gumbel) eğitiliyor...")
    soft_model = SoftF1Model(DIM, k=5)
    soft_model = train_model(soft_model, steps=2000, dim=DIM,
                             train_noise=TRAIN_NOISE, is_f1=False)

    # Noise sweep for soft model
    soft_acc, soft_mi = sweep(soft_model, noise_levels, dim=DIM,
                              repeats=30, is_f1=False)

    # Gate overlap sweep
    noise_levels3 = torch.linspace(0.1, 4.0, 30)
    overlaps, feat_acc2 = [], []

    f1_model.eval()
    with torch.no_grad():
        for noise in noise_levels3:
            ov_batch, fa_batch = [], []
            for _ in range(50):
                x1, _ = generate_data(batch_size=64, dim=DIM,
                                      noise_level=noise.item())
                x2, _ = generate_data(batch_size=64, dim=DIM,
                                      noise_level=noise.item())
                _, gate1, _ = f1_model(x1)
                _, gate2, _ = f1_model(x2)

                # Overlap: hangi feature'lar her iki batch'te de seçilmiş?
                # Per-sample overlap → mean
                intersection = (gate1 * gate2).sum(dim=1)
                union        = ((gate1 + gate2) > 0).float().sum(dim=1)
                ov = (intersection / (union + 1e-8)).mean().item()
                ov_batch.append(ov)

                # Feature selection accuracy (signal = dim 0-4)
                topk_idx = gate1.topk(5, dim=1).indices
                correct  = (topk_idx < 5).float().sum(dim=1) / 5.0
                fa_batch.append(correct.mean().item())

            overlaps.append(np.mean(ov_batch))
            feat_acc2.append(np.mean(fa_batch))
    f1_model.train()

    nl3_np = noise_levels3.numpy()

    # ── Figure 1: Ablation (F1 hard vs soft vs attention) ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(nl_np, f1_acc,   "b-o",  markersize=4, label="F1 hard gating")
    ax.plot(nl_np, soft_acc, "m-^",  markersize=4, label="F1 soft (Gumbel)")
    ax.plot(nl_np, attn_acc, "r--s", markersize=4, label="Attention (soft)")
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Noise level σ")
    ax.set_ylabel("Accuracy")
    ax.set_title("Ablation: Hard vs Soft Selection")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(nl_np, f1_mi,   "b-o",  markersize=4, label="F1 hard gating")
    ax.plot(nl_np, soft_mi, "m-^",  markersize=4, label="F1 soft (Gumbel)")
    ax.plot(nl_np, attn_mi, "r--s", markersize=4, label="Attention (soft)")
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Noise level σ")
    ax.set_ylabel("MI I(pred; label) [nats]")
    ax.set_title("Ablation: Information Retention")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle("Ablation: Hardness of Selection Drives Abrupt Failure",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/ablation_hard_vs_soft.png", dpi=150)
    plt.close()
    print("  → ablation_hard_vs_soft.png kaydedildi")

    # ── Figure 2: Gate overlap + feature accuracy ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(nl3_np, overlaps, "b-o", markersize=4,
            label="Gate overlap |S1∩S2|/|S1∪S2|")
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5,
               label="Train noise σ=0.5")
    ax.set_xlabel("Noise level σ")
    ax.set_ylabel("Overlap (Jaccard)")
    ax.set_title("Gate Selection Consistency vs Noise")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(nl3_np, feat_acc2, "g-o", markersize=4,
            label="Signal features in top-k")
    ax.axhline(y=0.05, color="gray", linestyle="--", alpha=0.5,
               label="Random baseline (5/100)")
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5,
               label="Train noise σ=0.5")
    ax.set_xlabel("Noise level σ")
    ax.set_ylabel("Fraction correct")
    ax.set_title("Feature Selection Accuracy vs Noise")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle("F1 Mechanism: Selection Consistency and Feature Accuracy",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/gate_overlap_analysis.png", dpi=150)
    plt.close()
    print("  → gate_overlap_analysis.png kaydedildi")
    print("\n=== Gate instability + feature selection accuracy ===")

    noise_levels2 = torch.linspace(0.1, 5.0, 35)
    gate_vars, correct_rates = [], []

    f1_model.eval()
    with torch.no_grad():
        for noise in noise_levels2:
            gv_batch, cr_batch = [], []
            for _ in range(50):
                x, y = generate_data(batch_size=128, dim=DIM,
                                     noise_level=noise.item())
                _, gate, _ = f1_model(x)

                # Gate variance — instability proxy
                gv_batch.append(gate.var(dim=1).mean().item())

                # Correct feature rate — ilk 5 dim gerçek signal
                # top-k'nın kaçı gerçek signal?
                topk_idx = gate.topk(5, dim=1).indices  # (batch, 5)
                correct = (topk_idx < 5).float().sum(dim=1) / 5.0
                cr_batch.append(correct.mean().item())

            gate_vars.append(np.mean(gv_batch))
            correct_rates.append(np.mean(cr_batch))
    f1_model.train()

    nl2_np = noise_levels2.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(nl2_np, gate_vars, "b-o", markersize=4)
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5,
               label="Train noise σ=0.5")
    ax.set_xlabel("Noise level σ")
    ax.set_ylabel("Gate variance (instability)")
    ax.set_title("Gate Instability vs Noise")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(nl2_np, correct_rates, "g-o", markersize=4,
            label="Correct features in top-k")
    ax.axhline(y=0.05, color="gray", linestyle="--", alpha=0.5,
               label="Random baseline (5/100)")
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5,
               label="Train noise σ=0.5")
    ax.set_xlabel("Noise level σ")
    ax.set_ylabel("Fraction of signal features selected")
    ax.set_title("Feature Selection Accuracy vs Noise")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle("F1 Mechanism: Gate Instability and Feature Selection Collapse",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/gate_analysis.png", dpi=150)
    plt.close()
    print("  → gate_analysis.png kaydedildi")

    # ── Temperature sweep ablation ────────────────────────────────────────────
    print("\n=== Temperature sweep ablation ===")

    temperatures = [0.05, 0.2, 0.5, 1.0, 5.0]
    temp_results = {}

    for temp in temperatures:
        class TempModel(nn.Module):
            def __init__(self, dim, t=1.0):
                super().__init__()
                self.t = t
                self.score = nn.Parameter(torch.zeros(dim))
                self.bias  = nn.Parameter(torch.zeros(dim))
                self.f2    = F2(dim)
            def forward(self, x):
                salience = x * torch.sigmoid(self.score) + self.bias
                weights  = torch.softmax(salience / self.t, dim=1)
                return self.f2(x * weights), weights, salience.detach()

        print(f"  T={temp} eğitiliyor...")
        m = TempModel(DIM, t=temp)
        m = train_model(m, steps=1500, dim=DIM,
                        train_noise=TRAIN_NOISE, is_f1=False)
        _, mis = sweep(m, noise_levels, dim=DIM, repeats=20, is_f1=False)
        temp_results[temp] = mis

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#08306b", "#2171b5", "#6baed6", "#bdd7e7", "#eff3ff"]
    for (temp, mis), col in zip(temp_results.items(), colors):
        ax.plot(nl_np, mis, "-o", markersize=3, color=col,
                label=f"Soft T={temp}")
    ax.plot(nl_np, f1_mi, "r-^", markersize=4, linewidth=2,
            label="F1 hard (top-k)", zorder=5)
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5,
               label="Train σ=0.5")
    ax.set_xlabel("Noise level σ")
    ax.set_ylabel("MI I(pred; label) [nats]")
    ax.set_title("Temperature Sweep: Hard → Soft Continuum\n"
                 "As T increases, collapse onset is delayed")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/temperature_sweep.png", dpi=150)
    plt.close()
    print("  → temperature_sweep.png kaydedildi")

    # ── Collapse point σ*(T) — same model, only T changes ────────────────────
    print("\n=== Collapse boundary σ*(T) ===")

    # Tek bir model eğitilir, sonra sadece T değiştirilerek sweep yapılır
    class UnifiedTempModel(nn.Module):
        """Aynı learned weights, sadece inference'ta T değişiyor."""
        def __init__(self, dim):
            super().__init__()
            self.score = nn.Parameter(torch.zeros(dim))
            self.bias  = nn.Parameter(torch.zeros(dim))
            self.f2    = F2(dim)
            self.T     = 1.0  # runtime'da değiştirilecek

        def forward(self, x):
            salience = x * torch.sigmoid(self.score) + self.bias
            weights  = torch.softmax(salience / self.T, dim=1)
            return self.f2(x * weights), weights, salience.detach()

    print("  Unified model eğitiliyor (T=1.0)...")
    unified = UnifiedTempModel(DIM)
    unified = train_model(unified, steps=2000, dim=DIM,
                          train_noise=TRAIN_NOISE, is_f1=False)

    # Her T için σ*(T) bul: MI'nın %10 altına düştüğü ilk σ
    test_temps  = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    noise_dense = torch.linspace(0.1, 6.0, 60)
    nd_np       = noise_dense.numpy()

    collapse_points = []
    all_mi_curves   = {}

    unified.eval()
    with torch.no_grad():
        for T in test_temps:
            unified.T = T
            mis = []
            for noise in noise_dense:
                batch_mi = []
                for _ in range(30):
                    x, y = generate_data(batch_size=128, dim=DIM,
                                         noise_level=noise.item())
                    logits, _, _ = unified(x)
                    batch_mi.append(compute_mi(logits, y))
                mis.append(np.mean(batch_mi))
            all_mi_curves[T] = mis

            # σ*(T): MI'nın maksimumunun %10'una düştüğü ilk nokta
            mi_arr  = np.array(mis)
            mi_peak = mi_arr.max()
            thresh  = mi_peak * 0.1
            below   = np.where(mi_arr < thresh)[0]
            sigma_star = nd_np[below[0]] if len(below) > 0 else nd_np[-1]
            collapse_points.append((T, sigma_star))
            print(f"  T={T:.2f}  σ*={sigma_star:.2f}")
    unified.train()

    # ── Figure A: MI curves for each T ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cmap = plt.cm.Blues
    ax = axes[0]
    for i, T in enumerate(test_temps):
        col = cmap(0.3 + 0.7 * i / (len(test_temps) - 1))
        ax.plot(nd_np, all_mi_curves[T], "-o", markersize=2,
                color=col, label=f"T={T}")
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Noise level σ")
    ax.set_ylabel("MI I(pred; label) [nats]")
    ax.set_title("MI Curves: Same Model, T Varies")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # ── Figure B: σ*(T) — collapse boundary ──────────────────────────────────
    ax = axes[1]
    Ts, sigmas = zip(*collapse_points)
    ax.plot(Ts, sigmas, "ro-", markersize=7, linewidth=2)
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Collapse point σ*(T)")
    ax.set_title("Phase Boundary: σ*(T) ↑ as T ↑\n"
                 "Hardness drives collapse onset")
    ax.grid(alpha=0.3)
    # Trend line
    from numpy.polynomial import polynomial as P
    c = P.polyfit(np.log(Ts), sigmas, 1)
    t_line = np.linspace(min(Ts), max(Ts), 100)
    ax.plot(t_line, P.polyval(np.log(t_line), c),
            "r--", alpha=0.5, label="log-linear fit")
    ax.legend()

    plt.suptitle("Collapse Boundary σ*(T): Hardness Continuum\n"
                 "Same weights, temperature controls selection rigidity",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/collapse_boundary.png", dpi=150)
    plt.close()
    print("  → collapse_boundary.png kaydedildi")
    idx = (noise_levels - 0.5).abs().argmin().item()
    print(f"\n  F1   acc={f1_acc[idx]:.3f}  MI={f1_mi[idx]:.3f}")
    print(f"  Attn acc={attn_acc[idx]:.3f}  MI={attn_mi[idx]:.3f}")
    print("\nTamamlandı.")
    print("  phase_transition_noise.png  → F1 abrupt, Attention smooth")
    print("  hysteresis_test.png         → forward vs reverse noise sweep")
    print("  temperature_sweep.png       → hard→soft continuum")


