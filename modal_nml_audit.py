"""Modal experiment: MDL audit of spurious dependencies in deep representations.

Preliminary result for the Matsushima (広域システム) research proposal.

Extends Matsushima's NML latent-common-cause detection from raw variables to LEARNED
deep-representation directions. Given a frozen encoder's features Z, a label Y, and an
environment indicator E, do a hyperparameter-free two-part-MDL model selection among:
    M_indep : z ~ N(mu, s^2)                (z carries no Y info)
    M_Y     : z | Y ~ N(mu_y, s^2)          (z-Y relation SHARED across environments -> robust)
    M_E     : z | E ~ N(mu_e, s^2)          (z driven by environment only)
    M_YE    : z | (Y,E) ~ N(mu_{y,e}, s^2)  (z-Y relation MODULATED by environment -> spurious)
Two-part MDL codelength = -loglik(MLE) + (k/2) log n.  Shortest code wins. No tuned threshold.

We audit interpretable CONCEPT directions (linear probes on the frozen representation):
color (spurious cue), shape (invariant content), PCA-1 (dominant), random (nuisance).
On ColoredMNIST the audit should select M_YE for the color concept (spurious) and M_Y for the
shape concept (robust). A per-coordinate scan gives a continuous E-help-vs-color-alignment trend.

Run smoke:  .venv/bin/modal run 松島/code/modal_nml_audit.py
Run full:   .venv/bin/modal run 松島/code/modal_nml_audit.py --full
"""
import modal

app = modal.App("koiki-nml-audit")

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "torch", "torchvision", "numpy", "scikit-learn", "matplotlib"
)


@app.function(image=image, timeout=2400, cpu=4.0)
def run_audit(smoke: bool = True):
    import io, base64, json
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import datasets, transforms
    from sklearn.linear_model import LogisticRegression
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.manual_seed(0)
    np.random.seed(0)
    dev = "cpu"

    # ---------- config ----------
    n_per_env = 800 if smoke else 6000
    epochs = 4 if smoke else 6
    feat_dim = 64
    # Color strength VARIES across train envs (0.2 vs 0.5) and is pooled-weaker (~0.65) than
    # shape (~0.88), so the net encodes BOTH a robust (shape) and a spurious (color) direction.
    env_pe = [0.2, 0.5]
    test_pe = 0.9            # held-out color-flipped split -> confirms shortcut reliance
    label_noise = 0.12
    n_env = len(env_pe)

    # ---------- ColoredMNIST ----------
    tf = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root="/tmp/mnist", train=True, download=True, transform=tf)
    imgs = mnist.data.float() / 255.0
    digits = mnist.targets.numpy()

    def make_env(idx, pe):
        x = imgs[idx]
        d = digits[idx]
        y = (d >= 5).astype(np.int64)
        flip_y = np.random.rand(len(y)) < label_noise
        y = np.where(flip_y, 1 - y, y)
        color = np.where(np.random.rand(len(y)) < pe, 1 - y, y)   # 1=red,0=green (spurious)
        img2 = torch.zeros(len(y), 2, 28, 28)
        for c in (0, 1):
            m = color == c
            img2[m, c] = x[m]
        return img2, torch.tensor(y), color.astype(np.int64), d

    perm = np.random.permutation(len(imgs))
    ptr = 0
    parts = []
    for e, pe in enumerate(env_pe):
        idx = perm[ptr: ptr + n_per_env]; ptr += n_per_env
        xi, yi, ci, di = make_env(idx, pe)
        parts.append((xi, yi, ci, di, np.full(len(yi), e)))
    idx_t = perm[ptr: ptr + n_per_env]; ptr += n_per_env
    xt, yt, ct, dt = make_env(idx_t, test_pe)

    X = torch.cat([p[0] for p in parts])
    Y = torch.cat([p[1] for p in parts])
    Cc = np.concatenate([p[2] for p in parts])
    Dd = np.concatenate([p[3] for p in parts])
    Ee = np.concatenate([p[4] for p in parts])
    Ysh = (Dd >= 5).astype(np.int64)          # noiseless shape signal (invariant content)

    # ---------- small CNN encoder ----------
    class Net(nn.Module):
        def __init__(self, fd):
            super().__init__()
            self.c1 = nn.Conv2d(2, 16, 3, padding=1)
            self.c2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc = nn.Linear(32 * 7 * 7, fd)
            self.head = nn.Linear(fd, 2)

        def feat(self, x):
            h = F.max_pool2d(F.relu(self.c1(x)), 2)
            h = F.max_pool2d(F.relu(self.c2(h)), 2)
            return F.relu(self.fc(h.flatten(1)))

        def forward(self, x):
            return self.head(self.feat(x))

    net = Net(feat_dim).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    bs = 128
    net.train()
    for ep in range(epochs):
        order = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            b = order[i: i + bs]
            opt.zero_grad()
            F.cross_entropy(net(X[b]), Y[b]).backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        acc_id = (net(X).argmax(1) == Y).float().mean().item()
        acc_flip = (net(xt).argmax(1) == yt).float().mean().item()
        Z = net.feat(X).cpu().numpy()         # FROZEN representation (n, feat_dim)

    # ---------- two-part MDL machinery ----------
    def codelen(z, groups):
        n = len(z)
        if groups is None:
            var = z.var() + 1e-6
            ll = -0.5 * n * (np.log(2 * np.pi * var) + 1.0); k = 2
        else:
            resid = np.empty_like(z); uq = np.unique(groups)
            for g in uq:
                m = groups == g; resid[m] = z[m] - z[m].mean()
            var = resid.var() + 1e-6
            ll = -0.5 * n * (np.log(2 * np.pi * var) + 1.0); k = len(uq) + 1
        return -ll + 0.5 * k * np.log(n)

    Yn = Y.numpy()
    YE = Yn * n_env + Ee

    def audit(z):
        z = (z - z.mean()) / (z.std() + 1e-8)
        cl = {"indep": codelen(z, None), "Y": codelen(z, Yn),
              "E": codelen(z, Ee), "YE": codelen(z, YE)}
        win = min(cl, key=cl.get)
        return cl, win, float((cl["Y"] - cl["YE"]) / len(z))   # E-help RATE (nats/sample); >0 => env-dependent

    # ---------- (A) interpretable concept probes ----------
    def probe(target):
        w = LogisticRegression(max_iter=400, C=1.0).fit(Z, target).coef_[0]
        return Z @ w
    Zc = Z - Z.mean(0)
    pc1 = np.linalg.svd(Zc, full_matrices=False)[2][0]
    rng = np.random.default_rng(0)
    z_color = probe(Cc)
    z_shape_raw = probe(Ysh)
    # shape concept CONTROLLED FOR the known shortcut: remove the linear color component, so the
    # audit asks "is there invariant content BEYOND the spurious color cue?" (a partial concept).
    b = float(np.dot(z_shape_raw, z_color) / (np.dot(z_color, z_color) + 1e-9))
    z_shape = z_shape_raw - b * z_color
    concepts = {
        "color (spurious cue)": z_color,
        "shape ⊥ color (invariant)": z_shape,
        "PCA-1 (dominant dir.)": Z @ pc1,
        "random (nuisance)": rng.standard_normal(len(Yn)),   # external noise (indep control)
    }
    concept_audit = {}
    for name, zc in concepts.items():
        cl, win, eh = audit(zc)
        concept_audit[name] = dict(cl=cl, winner=win, e_help=eh)

    # ---------- (B) per-coordinate continuous trend ----------
    rows = []
    n_dead = 0
    for j in range(Z.shape[1]):
        s = Z[:, j].std()
        if s < 1e-6:
            n_dead += 1; continue
        zz = (Z[:, j] - Z[:, j].mean()) / (s + 1e-8)
        _, win, eh = audit(Z[:, j])
        rows.append(dict(winner=win, e_help=eh,
                         col_align=float(abs(np.corrcoef(zz, Cc)[0, 1])),
                         shp_align=float(abs(np.corrcoef(zz, Ysh)[0, 1]))))
    align = np.array([r["col_align"] - r["shp_align"] for r in rows])
    ehv = np.array([r["e_help"] for r in rows])
    trend = float(np.corrcoef(align, ehv)[0, 1]) if len(rows) > 2 else float("nan")
    pen = 0.5 * (2 * n_env - 2) * np.log(len(Yn)) / len(Yn)   # per-sample MDL consistency floor

    summary = dict(
        acc_in_distribution=round(acc_id, 4),
        acc_color_flipped=round(acc_flip, 4),
        n_live_features=len(rows), n_dead_features=n_dead,
        mdl_floor_logn_over_n=round(pen, 5),
        concept_e_help_rate={k: round(v["e_help"], 4) for k, v in concept_audit.items()},
        concept_verdict={k: v["winner"] for k, v in concept_audit.items()},
        trend_corr_align_vs_ehelp=round(trend, 3),
        smoke=smoke,
    )

    # ---------- Fig.1 (two panels) ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    names = list(concept_audit.keys())
    rates = [concept_audit[n]["e_help"] for n in names]
    x = np.arange(len(names))
    ax1.bar(x, rates, color="#4c72b0")
    ax1.axhline(pen, color="#b03b3b", ls="--", lw=1.2,
                label=f"MDL consistency floor (dk/2)*ln(n)/n = {pen:.4f}")
    ax1.axhline(0, color="k", lw=0.6)
    for xi, r in zip(x, rates):
        ax1.text(xi, r, f"{r:.3f}", ha="center", va="bottom" if r >= 0 else "top", fontsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels([n.split(" (")[0] for n in names], rotation=12, ha="right", fontsize=8)
    ax1.set_ylabel("environment-help rate  [CL(z|Y)-CL(z|Y,E)]/n   (nats/sample)")
    ax1.set_title("(a) per-concept audit:  color >> shape / PCA > nuisance", fontsize=9)
    ax1.legend(fontsize=6.5, loc="upper right")
    sc = ax2.scatter(align, ehv, c=align, cmap="coolwarm", s=24, alpha=0.9)
    ax2.axhline(pen, color="#b03b3b", ls="--", lw=1.0)
    ax2.axhline(0, color="gray", lw=0.6, ls=":")
    ax2.set_xlabel("per-feature  |corr COLOR| - |corr SHAPE|")
    ax2.set_ylabel("environment-help rate  (nats/sample)")
    ax2.set_title(f"(b) per-coordinate audit  (trend r={trend:.2f}; dashed = MDL floor)", fontsize=9)
    fig.colorbar(sc, ax=ax2, label="color - shape alignment")
    fig.suptitle("Hyperparameter-light MDL audit of environment-dependent concepts in deep representations "
                 f"(ColoredMNIST; in-dist acc={acc_id:.2f}, color-flip acc={acc_flip:.2f})", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150)
    png_b64 = base64.b64encode(buf.getvalue()).decode()

    print("=== KOIKI NML AUDIT ===")
    print(json.dumps(summary, indent=2))
    return {"summary": summary, "fig_png_b64": png_b64}


@app.local_entrypoint()
def main(full: bool = False):
    import base64, os, json
    res = run_audit.remote(smoke=not full)
    print("SUMMARY:", json.dumps(res["summary"], indent=2))
    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    tag = "full" if full else "smoke"
    p = os.path.join(out_dir, f"fig1_mdl_audit_{tag}.png")
    with open(p, "wb") as f:
        f.write(base64.b64decode(res["fig_png_b64"]))
    print("WROTE", p)
