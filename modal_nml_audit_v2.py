"""松島 v2 — NML representation-audit, hardened to kill 3 weaknesses (codex-designed, detector fixed):
(1) anti-circular: audit raw coords WITHOUT colour/patch labels; with 2 shortcuts (colour + corner patch)
    the audit flags whichever the model actually relied on.
(2) exact NML: stochastic complexity via exact multinomial NML (Kontkanen-Myllymaki recurrence), not BIC.
    Detector = INVARIANT Z->Y vs ENVIRONMENT-MODULATED Z->Y codelength (the spuriousness in ColoredMNIST is
    an environment-modulated predictiveness, not a pure common cause).
(3) utility: env-help score PREDICTS OOD failure across N CNNs (Spearman); projecting out flagged directions
    improves OOD vs random/PCA/shape controls.

Run smoke: .venv/bin/modal run 松島/code/modal_nml_audit_v2.py
Run full:  .venv/bin/modal run 松島/code/modal_nml_audit_v2.py --full
"""
import modal

app = modal.App("matsushima-nml-audit-v2")
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch", "torchvision", "numpy", "scipy", "scikit-learn", "matplotlib"))
vol = modal.Volume.from_name("gtsinger-audio", create_if_missing=True)


def make_var(X, yflip, idxs, rho_col, rho_pat, rng):
    """ColoredMNIST + corner patch. rho_col / rho_pat may be scalar or per-sample array (for 2-env audit)."""
    import numpy as np, torch
    xb = X[idxs]; y = yflip[idxs]; n = len(xb)
    rc = np.broadcast_to(rho_col, (n,)); rp = np.broadcast_to(rho_pat, (n,))
    col = (y ^ (rng.random(n) >= rc)).astype(int)
    pat = (y ^ (rng.random(n) >= rp)).astype(int)
    img = np.zeros((n, 2, 28, 28), np.float32)
    img[:, 0] = xb * col[:, None, None]; img[:, 1] = xb * (1 - col[:, None, None])
    img[:, :, 0:3, 0:3] += pat[:, None, None, None].astype(np.float32) * 0.9
    return torch.tensor(img), torch.tensor(y.astype(int)), col, pat


@app.function(image=image, gpu="t4", timeout=7200, memory=16384, volumes={"/cache": vol})
def run(full: bool = False):
    import json, base64
    import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    from functools import lru_cache
    from scipy.special import logsumexp, gammaln
    from scipy.stats import spearmanr, pearsonr
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    import torchvision, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    # ---------- exact multinomial NML stochastic complexity (Kontkanen-Myllymaki) ----------
    @lru_cache(maxsize=400000)
    def log_C(n, K):
        if n == 0 or K == 1: return 0.0
        r = np.arange(0, n + 1)
        lr = np.where((r > 0) & (r < n),
                      r * (np.log(np.maximum(r, 1)) - np.log(n)) + (n - r) * (np.log(np.maximum(n - r, 1)) - np.log(n)), 0.0)
        lbin = gammaln(n + 1) - gammaln(r + 1) - gammaln(n - r + 1)
        logc2 = float(logsumexp(lbin + lr))
        if K == 2: return logc2
        logCj, logCj1 = 0.0, logc2
        for j in range(1, K - 1):
            logCj, logCj1 = logCj1, float(np.logaddexp(logCj1, np.log(n) - np.log(j) + logCj))
        return logCj1

    def SC(counts, K):
        counts = np.asarray(counts, float); n = int(counts.sum())
        if n == 0: return 0.0
        nz = counts[counts > 0]
        return (n * np.log(n) - float((nz * np.log(nz)).sum())) + log_C(n, K)

    def env_help(z, y, e, B=8, Ky=2):
        """invariant Z->Y vs environment-modulated Z->Y (exact-NML codelength gain, nats/sample). >0 => spurious."""
        n = len(z)
        ranks = np.argsort(np.argsort(z)); zb = np.minimum((ranks * B // n), B - 1)
        def cnt(a, K): return np.bincount(a, minlength=K)[:K]
        L_inv = sum(SC(cnt(y[zb == b], Ky), Ky) for b in range(B))
        L_mod = sum(SC(cnt(y[(zb == b) & (e == k)], Ky), Ky) for b in range(B) for k in range(2))
        return (L_inv - L_mod) / n

    def bic_env_help(z, y, e, B=8, Ky=2):                       # BIC counterpart (for the NML-vs-BIC inset)
        n = len(z); ranks = np.argsort(np.argsort(z)); zb = np.minimum((ranks * B // n), B - 1)
        def nll(mask):
            yy = y[mask];
            if len(yy) == 0: return 0.0
            c = np.bincount(yy, minlength=Ky); p = c / max(len(yy), 1)
            return -float((c * np.log(np.clip(p, 1e-9, 1))).sum())
        L_inv = sum(nll(zb == b) for b in range(B)) + (B * (Ky - 1) / 2) * np.log(n)
        L_mod = sum(nll((zb == b) & (e == k)) for b in range(B) for k in range(2)) + (2 * B * (Ky - 1) / 2) * np.log(n)
        return (L_inv - L_mod) / n

    # ---------- data ----------
    mnist = torchvision.datasets.MNIST("/cache/mnist", train=True, download=True)
    X = (mnist.data.float() / 255.0).numpy(); Yd = mnist.targets.numpy()
    idx = rng.permutation(len(X)); X, Yd = X[idx], Yd[idx]
    ybin = (Yd < 5).astype(int); yflip = ybin ^ (rng.random(len(ybin)) < 0.25)
    ntr = 30000 if full else 8000
    tr_i = np.arange(0, ntr); va_i = np.arange(ntr, ntr + 5000); te_i = np.arange(ntr + 5000, ntr + 10000)
    e_va = (rng.random(len(va_i)) < 0.5).astype(int)           # audit environments
    GAP = (0.9, 0.5)                                            # detectable env gap on the held-out audit set

    class CNN(nn.Module):
        def __init__(s, d=64):
            super().__init__(); s.c = nn.Sequential(nn.Conv2d(2, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten())
            s.f = nn.Linear(32 * 16, d); s.head = nn.Linear(d, 2)
        def feat(s, x): return torch.relu(s.f(s.c(x)))
        def forward(s, x): return s.head(s.feat(x))

    def train_cnn(Xtr, Ytr, wd, epochs):
        net = CNN().to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=wd)
        Xtr, Ytr = Xtr.to(dev), Ytr.to(dev); bs = 256
        for _ in range(epochs):
            for i in range(0, len(Xtr), bs):
                loss = F.cross_entropy(net(Xtr[i:i + bs]), Ytr[i:i + bs]); opt.zero_grad(); loss.backward(); opt.step()
        return net.eval()

    @torch.no_grad()
    def acc(net, Xt, Yt): Xt = Xt.to(dev); return float((net(Xt).argmax(1).cpu() == Yt).float().mean())
    @torch.no_grad()
    def feats(net, Xt): Xt = Xt.to(dev); return net.feat(Xt).cpu().numpy()

    def audit(net, Xva, Yva, eva, col=None, pat=None, pca=None):
        Z = feats(net, Xva); Zc = Z - Z.mean(0); y = Yva.numpy().astype(int)
        Vt = pca if pca is not None else np.linalg.svd(Zc, full_matrices=False)[2]   # PC directions (label-free)
        P = Zc @ Vt.T; P = (P - P.mean(0)) / (P.std(0) + 1e-6)        # audit in PCA basis (colour concentrates in few PCs)
        H = np.array([env_help(P[:, j], y, eva) for j in range(P.shape[1])])
        o = {"H": H, "P": P, "pca": Vt}
        if col is not None: o["col_auc"] = np.array([max(roc_auc_score(col, P[:, j]), 1 - roc_auc_score(col, P[:, j])) for j in range(P.shape[1])])
        if pat is not None: o["pat_auc"] = np.array([max(roc_auc_score(pat, P[:, j]), 1 - roc_auc_score(pat, P[:, j])) for j in range(P.shape[1])])
        return o
    def Hmodel(H): top = np.sort(H)[::-1][:5]; return float(top[top > 0].mean()) if (top > 0).any() else 0.0

    EP = 4 if full else 2
    rhos = [0.55, 0.65, 0.75, 0.85, 0.95]; wds = [0.0, 1e-3]; seeds = [1, 2, 3] if full else [1]

    # ============ (B) OOD PREDICTION across N CNNs ============
    Hs, drops, rho_tag = [], [], []
    for rho in rhos:
        for wd in wds:
            for sd in seeds:
                torch.manual_seed(sd)
                Xtr, Ytr, _, _ = make_var(X, yflip, tr_i, rho, 0.5, rng)              # train: colour rho, patch off
                Xva, Yva, colva, _ = make_var(X, yflip, va_i, np.where(e_va == 0, *GAP), 0.5, rng)   # audit: 2-env gap
                Xid, Yid, _, _ = make_var(X, yflip, te_i, 0.9, 0.5, rng)
                Xood, Yood, _, _ = make_var(X, yflip, te_i, 0.1, 0.5, rng)
                net = train_cnn(Xtr, Ytr, wd, EP)
                Hs.append(Hmodel(audit(net, Xva, Yva, e_va)["H"]))
                drops.append(acc(net, Xid, Yid) - acc(net, Xood, Yood)); rho_tag.append(rho)
    Hs, drops = np.array(Hs), np.array(drops)
    rs = spearmanr(Hs, drops)[0]; pr = pearsonr(Hs, drops)[0]
    boot = [spearmanr(Hs[i], drops[i])[0] for i in (rng.integers(0, len(Hs), len(Hs)) for _ in range(1000))]
    ci = (float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5)))

    # ============ (C) INTERVENTION: project out flagged dirs, refit linear head ============
    torch.manual_seed(1)
    Xtr, Ytr, _, _ = make_var(X, yflip, tr_i, 0.9, 0.5, rng)
    Xva, Yva, colva, _ = make_var(X, yflip, va_i, np.where(e_va == 0, *GAP), 0.5, rng)
    Xid, Yid, _, _ = make_var(X, yflip, te_i, 0.9, 0.5, rng); Xood, Yood, _, _ = make_var(X, yflip, te_i, 0.1, 0.5, rng)
    net = train_cnn(Xtr, Ytr, 0.0, EP + 1); au = audit(net, Xva, Yva, e_va, colva)
    Vt = au["pca"]; Ztr, Zid, Zood = feats(net, Xtr), feats(net, Xid), feats(net, Xood); mu = Ztr.mean(0); d = Ztr.shape[1]
    def lin_ood(P):
        clf = LogisticRegression(max_iter=300).fit((Ztr - mu) @ P + mu, Ytr.numpy())
        return float((clf.predict((Zid - mu) @ P + mu) == Yid.numpy()).mean()), float((clf.predict((Zood - mu) @ P + mu) == Yood.numpy()).mean())
    def proj_out(D): V = np.linalg.qr(D.T)[0] if D.shape[0] else np.zeros((d, 0)); return np.eye(d) - V @ V.T
    order = np.argsort(au["H"])[::-1]; k = 8
    Ptr = (Ztr - mu) @ Vt.T; wsh = LogisticRegression(max_iter=300).fit(Ptr, Ytr.numpy()).coef_[0]
    sh = np.argsort(np.abs(wsh) * (au["H"] < np.median(au["H"])))[::-1]
    res = {"none": lin_ood(np.eye(d)), "flagged": lin_ood(proj_out(Vt[order[:k]])),
           "pca": lin_ood(proj_out(Vt[:k])), "shape": lin_ood(proj_out(Vt[sh[:k]]))}
    rnd = [lin_ood(proj_out(Vt[rng.permutation(d)[:k]])) for _ in range(10)]
    res["random"] = (float(np.mean([a for a, _ in rnd])), float(np.mean([b for _, b in rnd])))

    # ============ (D) ANTI-CIRCULAR: two shortcuts, audit blind to col/patch labels ============
    def regime(rc, rp):
        torch.manual_seed(7)
        Xtr2, Ytr2, _, _ = make_var(X, yflip, tr_i, rc, rp, rng)
        Xva2, Yva2, col2, pat2 = make_var(X, yflip, va_i, np.where(e_va == 0, *GAP), np.where(e_va == 0, *GAP), rng)
        a2 = audit(train_cnn(Xtr2, Ytr2, 0.0, EP), Xva2, Yva2, e_va, col2, pat2)
        top = np.argsort(a2["H"])[::-1][:5]
        return float(np.mean(a2["col_auc"][top])), float(np.mean(a2["pat_auc"][top]))
    regA = regime(0.95, 0.55); regB = regime(0.55, 0.95)

    # ============ FIGURE ============
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.3))
    Pp = au["P"]; jcol = int(np.argmax(au["col_auc"])); y = Yva.numpy().astype(int)
    def codelen_pair(zj, scorer):                              # (invariant, modulated) per-sample codelengths
        B, Ky, n = 8, 2, len(zj); ranks = np.argsort(np.argsort(zj)); zb = np.minimum(ranks * B // n, B - 1)
        if scorer == "nml":
            def cnt(a, K): return np.bincount(a, minlength=K)[:K]
            Li = sum(SC(cnt(y[zb == b], Ky), Ky) for b in range(B)); Lm = sum(SC(cnt(y[(zb == b) & (e_va == k)], Ky), Ky) for b in range(B) for k in range(2))
        return Li / n, Lm / n
    li, lm = codelen_pair(Pp[:, jcol], "nml")
    ax[0].bar(["invariant\nZ→Y", "env-modulated\nZ→Y"], [li, lm], color=["#1f77b4", "#d62728"])
    ax[0].set_title(f"(a) exact-NML codelength, colour dir\nmodulated shorter ⇒ spurious (H={li-lm:.3f})"); ax[0].set_ylabel("nats / sample")
    sc = ax[1].scatter(Hs, drops, c=rho_tag, cmap="viridis", s=45)
    ax[1].set_xlabel("env-help H (audit, no colour labels)"); ax[1].set_ylabel("OOD drop (ID − colour-flip)")
    ax[1].set_title(f"(b) audit predicts OOD failure\nSpearman={rs:.2f} [{ci[0]:.2f},{ci[1]:.2f}], N={len(Hs)}"); plt.colorbar(sc, ax=ax[1], label="train ρ")
    names = ["none", "random", "pca", "shape", "flagged"]; ood = [res[k_][1] for k_ in names]; idd = [res[k_][0] for k_ in names]
    xb = np.arange(len(names)); ax[2].bar(xb - 0.2, ood, 0.4, label="OOD acc", color="#d62728"); ax[2].bar(xb + 0.2, idd, 0.4, label="ID acc", color="#1f77b4")
    ax[2].set_xticks(xb); ax[2].set_xticklabels(names, rotation=20); ax[2].set_title("(c) project out flagged dirs → OOD ↑"); ax[2].legend(fontsize=8)
    ax[3].bar([0, 1], regA, 0.6, color="#1f77b4"); ax[3].bar([2.5, 3.5], regB, 0.6, color="#ff7f0e")
    ax[3].set_xticks([0, 1, 2.5, 3.5]); ax[3].set_xticklabels(["col", "patch", "col", "patch"]); ax[3].set_ylim(0.45, 1.0); ax[3].axhline(0.5, ls=":", c="k")
    ax[3].set_title("(d) flags whichever is spurious\nA:colour-strong   B:patch-strong (top-5 post-hoc AUC)")
    plt.tight_layout(); png = "/cache/matsushima_fig_v2.png"; plt.savefig(png, dpi=130, bbox_inches="tight")
    with open(png, "rb") as f: b64 = base64.b64encode(f.read()).decode()

    summary = dict(full=full, N_models=len(Hs), ood_spearman=round(float(rs), 3), ood_pearson=round(float(pr), 3),
                   spearman_CI=[round(c, 3) for c in ci],
                   intervention={k_: {"ID": round(res[k_][0], 3), "OOD": round(res[k_][1], 3)} for k_ in names},
                   regimeA_colAUC_patAUC=[round(regA[0], 3), round(regA[1], 3)],
                   regimeB_colAUC_patAUC=[round(regB[0], 3), round(regB[1], 3)],
                   nml_colour_codelen={"invariant": round(li, 4), "modulated": round(lm, 4), "H": round(li - lm, 4)})
    with open("/cache/matsushima_v2_result.json", "w") as f: json.dump(summary, f, indent=2)
    vol.commit(); print("=== MATSUSHIMA v2 ==="); print(json.dumps(summary, indent=2))
    return {"summary": summary, "png_b64": b64}


@app.local_entrypoint()
def main(full: bool = False):
    import json, base64
    r = run.remote(full=full)
    open("松島/figures/fig_v2.png", "wb").write(base64.b64decode(r["png_b64"]))
    print("SUMMARY:", json.dumps(r["summary"], indent=2)); print("saved 松島/figures/fig_v2.png")
