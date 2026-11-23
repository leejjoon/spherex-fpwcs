import json, glob, sys, collections, numpy as np, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, "spherex-fpwcs/src")
import spherex_fpwcs as fp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
plt.rcParams.update({"font.size": 9, "axes.grid": False, "figure.dpi": 110})
OUT = "spherex-fpwcs/docs/figures/"; T = fp.load_template(); PR = T.pixel_rad
A1 = lambda P: np.c_[P, np.ones(len(P))]
COL = {1: "#1f77b4", 2: "#2a9d8f", 3: "#6a4c93", 4: "#e76f51", 5: "#d4a017", 6: "#c1121f"}

# 1. focal-plane layout
fig, ax = plt.subplots(figsize=(9, 3.6))
e = np.array([[0, 0], [2039, 0], [2039, 2039], [0, 2039], [0, 0]], float)
for d in range(1, 7):
    t = np.linspace(0, 1, 40)[:, None]; edge = np.concatenate([e[i] + t*(e[i+1]-e[i]) for i in range(4)])
    p = T.plane(d, edge[:, 0], edge[:, 1]) / PR
    ax.plot(p[:, 0], p[:, 1], color=COL[d], lw=1.6 if d < 4 else 1.0, ls="-" if d < 4 else "--", label=f"D{d}")
    c = T.plane(d, [1019.5], [1019.5])[0] / PR; ax.text(c[0], c[1] + (350 if d < 4 else -350), f"D{d}", color=COL[d], ha="center", va="center")
ax.set_aspect("equal"); ax.set_xlabel("focal-plane $\\xi$  [pixels of 6.15\"]"); ax.set_ylabel("$\\eta$  [pixels]")
ax.set_title("Detector footprints in the common focal-plane frame (solid: bands 1-3, dashed: bands 4-6)")
fig.tight_layout(); fig.savefig(OUT + "focal_plane_layout.png"); plt.close(fig)

# 2. non-linear distortion per detector
g = np.linspace(0, 2039, 15); X, Y = (a.ravel() for a in np.meshgrid(g, g))
fig, axs = plt.subplots(2, 3, figsize=(10, 7), sharex=True, sharey=True)
for d in range(1, 7):
    ax = axs[(d-1)//3, (d-1) % 3]; p = T.plane(d, X, Y) / PR; pix = np.c_[X, Y]
    r = p - A1(pix) @ np.linalg.lstsq(A1(pix), p, rcond=None)[0]
    Minv = np.linalg.inv(np.linalg.lstsq(A1(pix), p, rcond=None)[0][:2].T); r = r @ Minv.T     # into pixel axes
    m = np.hypot(*r.T); ax.quiver(X, Y, r[:, 0], r[:, 1], angles="xy", scale_units="xy", scale=1/60., width=0.004, color=COL[d])
    ax.quiver([150], [2190], [5], [0], angles="xy", scale_units="xy", scale=1/60., width=0.004, color="k"); ax.text(520, 2160, "5 px", fontsize=8)
    ax.set_title(f"D{d}: rms {np.sqrt((m**2).mean()):.2f} px, max {m.max():.2f} px"); print("nonlinear distortion", d, round(float(np.sqrt((m**2).mean())),2), round(float(m.max()),2)); ax.set_xlim(-80, 2120); ax.set_ylim(-80, 2330); ax.set_aspect("equal")
fig.supxlabel("x [pixel]"); fig.supylabel("y [pixel]"); fig.suptitle("Non-linear part of the pixel-to-sky mapping (best affine removed), arrows x60"); fig.tight_layout(); fig.savefig(OUT + "distortion_per_detector.png"); plt.close(fig)

# 3. before / after star residuals
V = sum((json.load(open(f)) for f in sorted(glob.glob("out/pkg_validate_*.json"))), [])
by = {}
for r in V: by[r["file"]] = r          # later groups override (epochs2 re-measurement)
V = list(by.values())
e8 = np.linspace(0, 2040, 9); cc = (e8[:-1]+e8[1:])/2; CX, CY = (a.ravel() for a in np.meshgrid(cc, cc))
show = [r for r in sorted(V, key=lambda r: -r["header"]["max"]) if np.isfinite(r["header"]["max"])][:6]
fig, axs = plt.subplots(2, 6, figsize=(16, 5.8), sharex=True, sharey=True)
for j, r in enumerate(show):
    for i, k in enumerate(("header", "all6")):
        ax = axs[i, j]; c = np.array(r[k]["cells"])
        ax.quiver(CX, CY, c[:, 0], c[:, 1], angles="xy", scale_units="xy", scale=1/40., width=0.006, color="#c1121f" if i == 0 else "#1f77b4")
        ax.quiver([150], [2160], [5], [0], angles="xy", scale_units="xy", scale=1/40., width=0.006, color="k"); ax.text(420, 2130, "5 px", fontsize=7)
        ax.set_xlim(-60, 2100); ax.set_ylim(-60, 2300); ax.set_aspect("equal")
        ax.set_title(f"{r['file'][7:26]}\n{'DR1 header' if i == 0 else 'spherex-fpwcs'}: rms {r[k]['rms']:.2f}, max {r[k]['max']:.2f} px", fontsize=8)
fig.suptitle("Measured 2MASS star offsets per 255-px cell (arrows x40): DR1 header WCS (top) vs regenerated WCS (bottom)")
fig.tight_layout(); fig.savefig(OUT + "before_after.png"); plt.close(fig)

# 4. failure statistics (week 2026W13_2A, package scan)
W = json.load(open("out/pkg_week.json")); mx = np.array([r["max"] for r in W]); det = np.array([r["det"] for r in W])
b = np.abs(SkyCoord([r["crval"][0] for r in W]*u.deg, [r["crval"][1] for r in W]*u.deg).galactic.b.deg)
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
bins = [0, 2, 5, 10, 20, 40, 90]; fr = [100*(mx[(b >= lo) & (b < hi)] > 1).mean() for lo, hi in zip(bins[:-1], bins[1:])]
n = [int(((b >= lo) & (b < hi)).sum()) for lo, hi in zip(bins[:-1], bins[1:])]
axs[0].bar(range(len(fr)), fr, color="#2a9d8f"); axs[0].set_xticks(range(len(fr)), [f"{lo}-{hi}" for lo, hi in zip(bins[:-1], bins[1:])])
for i, (f_, n_) in enumerate(zip(fr, n)): axs[0].text(i, f_ + 0.08, f"{f_:.1f}%\nn={n_}", ha="center", fontsize=7)
axs[0].set_xlabel("|b| [deg]"); axs[0].set_ylabel("files with WCS error > 1 px [%]"); axs[0].set_ylim(0, max(fr)*1.3); axs[0].set_title("Failure rate vs Galactic latitude")
fd = [100*(mx[det == d] > 1).mean() for d in range(1, 7)]
axs[1].bar(range(1, 7), fd, color=[COL[d] for d in range(1, 7)]); axs[1].set_xlabel("detector"); axs[1].set_ylabel("files > 1 px [%]"); axs[1].set_title("Failure rate by detector")
for d, f_ in zip(range(1, 7), fd): axs[1].text(d, f_ + 0.02, f"{f_:.2f}%", ha="center", fontsize=7)
axs[2].hist(np.clip(mx, 1e-2, 20), bins=np.logspace(-2, np.log10(20), 70), color="#555"); axs[2].set_xscale("log"); axs[2].set_yscale("log")
axs[2].axvline(0.3, color="#d4a017", ls="--"); axs[2].axvline(1, color="#c1121f", ls="--")
axs[2].set_xlabel("max |header WCS - focal-plane solution| [px]"); axs[2].set_ylabel("files"); axs[2].set_title(f"All {len(mx)} files of week 2026W13_2A")
fig.tight_layout(); fig.savefig(OUT + "failure_statistics.png"); plt.close(fig)

# 5. aberration
S = np.load("out/long_scale_aberr.npy")
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
axs[0].plot(1e6*S[:, 1], 1e6*(S[:, 0]-1), ".", ms=2, color="#1f77b4"); lim = [-115, 115]; axs[0].plot(lim, lim, "k-", lw=0.7)
axs[0].set_xlabel("predicted  $\\beta\\cos\\theta$  (Earth + spacecraft velocity)  [ppm]"); axs[0].set_ylabel("fitted plate scale - 1  [ppm]")
axs[0].set_title(f"Per-exposure scale is velocity aberration ({len(S)} exposures)")
res = 1e6*(S[:, 0]-1-S[:, 1]); res -= np.median(res)
axs[1].plot(S[:, 2]-S[:, 2].min(), res, ".", ms=2, color="#555"); axs[1].set_ylim(-8, 8)
axs[1].set_xlabel("days since 2025-04-24"); axs[1].set_ylabel("scale residual after removing aberration [ppm]"); axs[1].set_title(f"Residual rms {np.std(res):.1f} ppm  (1 ppm = 0.001 px per 1000 px)")
fig.tight_layout(); fig.savefig(OUT + "aberration_scale.png"); plt.close(fig)

# 6. long-term stability
L = json.load(open("out/pkg_long.json")); D = np.load("out/long_template_drift.npy")
wk = sorted(set(r["ex"].split(":")[0] for r in L)); date = {}
for r in L: date.setdefault(r["ex"].split(":")[0], []).append(r["date"][:10])
from datetime import date as dt
days = np.array([(dt.fromisoformat(sorted(date[w])[len(date[w])//2]) - dt(2025, 4, 24)).days for w in wk])
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
for d in range(6):
    axs[0].plot(days, [np.hypot(*D[i, d*64:(d+1)*64].T).max() for i in range(len(wk))], "o-", ms=3, color=COL[d+1], label=f"D{d+1}")
axs[0].set_ylim(0, 0.05); axs[0].legend(ncol=6, fontsize=7, loc="upper center"); axs[0].set_xlabel("days since 2025-04-24")
axs[0].set_ylabel("max template change vs March 2026 [px]"); axs[0].set_title("Focal-plane geometry: change of the epoch median")
med, p90, p99 = [], [], []
for w in wk:
    v = np.array([r["loo"] for r in L if r["ex"].startswith(w) and r["max"] < 0.3 and r["loo"] is not None]); med.append(np.median(v)); p90.append(np.percentile(v, 90)); p99.append(np.percentile(v, 99))
axs[1].plot(days, med, "o-", ms=3, label="median"); axs[1].plot(days, p90, "s-", ms=3, label="90th pct"); axs[1].plot(days, p99, "^-", ms=3, label="99th pct")
axs[1].set_ylim(0, 0.3); axs[1].legend(fontsize=8); axs[1].set_xlabel("days since 2025-04-24"); axs[1].set_ylabel("max |prediction - header| per file [px]")
axs[1].set_title("Leave-one-out prediction with the single shipped template")
fig.tight_layout(); fig.savefig(OUT + "long_term_stability.png"); plt.close(fig)
print("loo per epoch median range", min(med), max(med), "p99 range", min(p99), max(p99))
# numbers for the report
print("validated files:", len(V))
for r in sorted(V, key=lambda r: (r["group"], -r["header"]["max"])):
    print(f"| `{r['file'][7:26]}` | {r['date']} | {r['b']:+.1f} | {r['header']['rms']:.2f} / {r['header']['max']:.2f} | {r['all6']['rms']:.3f} / {r['all6']['max']:.3f} | {r['noself']['rms']:.3f} / {r['noself']['max']:.3f} | {r['group']}")
a = np.array([r["all6"]["rms"] for r in V]); print("fpwcs rms: median", np.median(a), "max", a.max(), " max of max:", max(r["all6"]["max"] for r in V))
