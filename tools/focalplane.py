"""Focal-plane model: is the relative geometry of the 6 detectors rigid from exposure to exposure?
If so, one telescope attitude per exposure (fit to the 6 header WCSs, no stars) predicts every detector's WCS.
Also: decompose the per-detector distortion into a distortion common to a focal plane + per-band residual."""
import json, re, os, glob, warnings, collections, numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import least_squares
warnings.simplefilter("ignore")
RAD2PX = 206264.806 / 6.15
H = [json.loads(l) for l in open("out/week_headers.jsonl")]
SC = {o["key"]: o for o in json.load(open("out/week_scan.json"))}
e = np.linspace(0, 2040, 9); cc = (e[:-1]+e[1:])/2; CX, CY = np.meshgrid(cc, cc); CX, CY = CX.ravel(), CY.ravel()
NP = len(CX)
WK = re.compile(r"(CRPIX|CRVAL|CTYPE|CUNIT|CDELT)[12]$|PC\d_\d$|CD\d_\d$|[AB]P?_\d_\d$|[AB]P?_ORDER$|LONPOLE$|LATPOLE$|RADESYS$")
def mkwcs(h):
    hd = fits.Header()
    for k, v in h.items():
        if WK.match(k): hd[k] = v
    return WCS(hd)
def vec(w, x, y):
    ra, dec = np.radians(w.all_pix2world(x, y, 0))
    return np.c_[np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)]

if os.path.exists("out/week_vec.npy"): V = np.load("out/week_vec.npy")
else:
    V = np.array([vec(mkwcs(h), CX, CY) for h in H]); np.save("out/week_vec.npy", V)

# index by exposure
exp = collections.defaultdict(dict)
for i, h in enumerate(H):
    n = h["_key"].split("/")[-1].split("_"); ex, d = n[3] + "_" + n[4][0], int(n[4][2])
    if d not in exp[ex] or "retry" not in h["_key"]: exp[ex][d] = i
print("exposures:", len(exp), " detectors per exposure:", dict(sorted(collections.Counter(len(v) for v in exp.values()).items())))
dev = np.array([SC[h["_key"]]["devmax"] for h in H])

def kabsch(G, S, w=None):
    w = np.ones(len(G)) if w is None else w
    U, _, Vt = np.linalg.svd((G * w[:, None]).T @ S); D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T                      # S ~ R @ G
def plane(v, fr):                              # gnomonic about frame z axis, in pixels
    p = v @ fr.T; return p[:, :2] / p[:, 2:3] * RAD2PX
def unplane(p, fr):
    v = np.c_[p / RAD2PX, np.ones(len(p))]; v /= np.linalg.norm(v, axis=1)[:, None]; return v @ fr
def fit2d(pg, ps, kind, w):
    """ps ~ T(pg);  kind: rigid (rot+shift), sim (+scale), affine"""
    sw = w / w.sum(); mg, ms = (pg*sw[:, None]).sum(0), (ps*sw[:, None]).sum(0); a, b = pg-mg, ps-ms
    if kind == "affine":
        M = np.linalg.lstsq(a*np.sqrt(w)[:, None], b*np.sqrt(w)[:, None], rcond=None)[0].T
    else:
        c = (w*(a[:, 0]*b[:, 0]+a[:, 1]*b[:, 1])).sum(); s_ = (w*(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])).sum()
        th = np.arctan2(s_, c); sc = np.hypot(c, s_)/(w*(a**2).sum(1)).sum() if kind == "sim" else 1.0
        M = sc*np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return M, ms - M @ mg
def solve(G2, fr, Gv, S, mask, kind, robust=True):
    """S: (ndet*NP,3) sky vectors; returns function mapping sky vectors -> aligned template-plane coords, and prediction."""
    R = kabsch(Gv[mask], S[mask]); ps = plane(S @ R, fr)        # rotate sky into template frame
    w = mask.astype(float)
    for it in range(15 if robust else 1):
        M, t = fit2d(G2, ps, kind, np.maximum(w, 1e-12) * mask)
        res = ps - (G2 @ M.T + t)
        if robust: w = 1/(1 + ((res**2).sum(1))/0.15**2)
    return R, M, t, res

# reference exposure: all six detectors clean
ex0 = next(k for k, v in exp.items() if len(v) == 6 and max(dev[i] for i in v.values()) < 0.04)
S0 = np.concatenate([V[exp[ex0][d]] for d in range(1, 7)])
z = S0.mean(0); z /= np.linalg.norm(z); x = np.cross([0, 0, 1.], z); x /= np.linalg.norm(x); fr = np.array([x, np.cross(z, x), z])
results = {}
for kind in ("rigid", "sim", "affine"):
    G2 = plane(S0, fr)
    for it in range(4):
        Gv = unplane(G2, fr); acc = np.full((len(exp), 6*NP, 2), np.nan)
        for j, (ex, dd) in enumerate(exp.items()):
            if len(dd) < 3: continue
            S = np.zeros((6*NP, 3)); mask = np.zeros(6*NP, bool)
            for d, i in dd.items(): S[(d-1)*NP:d*NP] = V[i]; mask[(d-1)*NP:d*NP] = True
            S[~mask] = Gv[~mask]
            R, M, t, res = solve(G2, fr, Gv, S, mask, kind)
            back = (plane(S @ R, fr) - t) @ np.linalg.inv(M).T
            acc[j][mask] = back[mask]
        G2 = np.nanmedian(acc, axis=0)
    Gv = unplane(G2, fr)
    # evaluate: prediction of each detector from the robust all-detector attitude, and leave-one-out
    err_all = np.full(len(H), np.nan); err_loo = np.full(len(H), np.nan); sols = {}
    for ex, dd in exp.items():
        if len(dd) < 3: continue
        S = np.zeros((6*NP, 3)); mask = np.zeros(6*NP, bool)
        for d, i in dd.items(): S[(d-1)*NP:d*NP] = V[i]; mask[(d-1)*NP:d*NP] = True
        S[~mask] = Gv[~mask]
        R, M, t, res = solve(G2, fr, Gv, S, mask, kind); sols[ex] = (R, M, t)
        for d, i in dd.items():
            err_all[i] = np.hypot(*res[(d-1)*NP:d*NP].T).max()
            m2 = mask.copy(); m2[(d-1)*NP:d*NP] = False
            R2, M2, t2, _ = solve(G2, fr, Gv, S, m2, kind)
            r2 = plane(S[(d-1)*NP:d*NP] @ R2, fr) - (G2[(d-1)*NP:d*NP] @ M2.T + t2)
            err_loo[i] = np.hypot(*r2.T).max()
    good = (dev < 0.3) & np.isfinite(err_all)
    print(f"\n[{kind}] per-exposure attitude model; max |header WCS - focal-plane prediction| per file, clean files (n={good.sum()}):")
    for nm, er in (("all-6 robust", err_all), ("leave-one-out", err_loo)):
        print(f"   {nm:14s} median={np.nanmedian(er[good]):.3f}  p90={np.nanpercentile(er[good],90):.3f}  p99={np.nanpercentile(er[good],99):.3f} px")
    results[kind] = dict(G2=G2, Gv=Gv, sols=sols, err_all=err_all, err_loo=err_loo)
np.save("out/focalplane_template_sim.npy", results["sim"]["G2"])

# ---- truth test on the image-checked files
meas = {}
for f in glob.glob("out/resid_*.json"):
    j = json.load(open(f)); meas["0568_" + j["tag"]] = np.array([[c["dx"], c["dy"]] for c in j["cells"]])
for f in glob.glob("out/spot_*.json"):
    j = json.load(open(f)); meas[j["tag"]] = np.array(j["cells"])[:, :2]
names = [h["_key"].split("/")[-1] for h in H]
print("\nstar-truth test (64 cell means): error of header WCS vs error of star-free focal-plane prediction [rms / max px]")
print("file        header WCS     | sim all-6 robust | sim leave-one-out | affine leave-one-out")
agg = collections.defaultdict(list); PRED = {}
for tag in sorted(meas, key=lambda t: -np.hypot(*meas[t].T).max()):
    i = next(k for k, n in enumerate(names) if f"_{tag}_" in n and "retry" not in n)
    ex, d = tag[:6], int(tag[7]); r = meas[tag]; dd = exp[ex]
    T = vec(mkwcs(H[i]), CX - r[:, 0], CY - r[:, 1])     # true sky direction of the cell centres
    out = [np.hypot(*r.T)]
    for kind, loo in (("sim", False), ("sim", True), ("affine", True)):
        G2, Gv = results[kind]["G2"], results[kind]["Gv"]
        S = Gv.copy(); mask = np.zeros(6*NP, bool)
        for d2, i2 in dd.items(): S[(d2-1)*NP:d2*NP] = V[i2]; mask[(d2-1)*NP:d2*NP] = True
        if loo: mask[(d-1)*NP:d*NP] = False
        R, M, t, _ = solve(G2, fr, Gv, S, mask, kind)
        out.append(np.hypot(*(plane(T @ R, fr) - (G2[(d-1)*NP:d*NP] @ M.T + t)).T))
        if kind == "sim" and loo:
            pv = unplane(G2[(d-1)*NP:d*NP] @ M.T + t, fr) @ R.T
            PRED[tag] = np.degrees(np.c_[np.arctan2(pv[:, 1], pv[:, 0]) % (2*np.pi), np.arcsin(pv[:, 2])]).tolist()
    st = [(np.sqrt((o**2).mean()), o.max()) for o in out]
    for k, s_ in enumerate(st): agg[k].append(s_)
    if st[0][1] > 0.3 or tag in ("0568_4D5", "0018_1D1", "0670_3D2"):
        print(f"{tag} n={len(dd)} {st[0][0]:5.2f} / {st[0][1]:5.2f}  |  {st[1][0]:5.3f} / {st[1][1]:5.3f}  |  {st[2][0]:5.3f} / {st[2][1]:5.3f}   |  {st[3][0]:5.3f} / {st[3][1]:5.3f}")
for k, lab in enumerate(["header", "sim all-6", "sim LOO", "affine LOO"]):
    a = np.array(agg[k]); print(f"   {lab:11s} over 36 files: median rms={np.median(a[:,0]):.3f}  worst rms={a[:,0].max():.3f}  worst max={a[:,1].max():.3f} px")

json.dump(PRED, open("out/focalplane_pred_loo.json", "w"))
# ---- decomposition: common focal-plane distortion + per-band residual
G2 = results["sim"]["G2"]
def terms(u, v, order):
    return np.stack([u**p * v**q for p in range(order+1) for q in range(order+1-p)], 1)
def decomp(dets, order, mirror=()):
    pix = np.c_[CX - 1019.5, CY - 1019.5]; tgt = np.concatenate([G2[(d-1)*NP:d*NP] for d in dets])
    tgt = tgt - tgt.mean(0)
    def place(p):
        out = []
        for k, d in enumerate(dets):
            x0, y0, th = p[3*k:3*k+3]; c, s = np.cos(th), np.sin(th); q = pix * ([-1, 1] if d in mirror else [1, 1])
            out.append(np.c_[c*q[:, 0]-s*q[:, 1]+x0, s*q[:, 0]+c*q[:, 1]+y0])
        return np.concatenate(out)
    def resid(p):
        A = terms(*(place(p).T/1000.), order); coef = np.linalg.lstsq(A, tgt, rcond=None)[0]
        return (tgt - A @ coef).ravel()
    # initial placement from the template centroids, in pixel units
    cen = np.array([G2[(d-1)*NP:d*NP].mean(0) for d in dets]); cen -= cen.mean(0)
    best = None
    for th0 in (0, np.pi/2, np.pi, -np.pi/2):
        p0 = np.concatenate([[c_[0], c_[1], th0] for c_ in cen])
        s = least_squares(resid, p0, x_scale=np.tile([10, 10, 1e-3], len(dets)))
        if best is None or s.cost < best.cost: best = s
    r = np.hypot(*best.fun.reshape(-1, 2).T).reshape(len(dets), NP)
    return {d: (np.sqrt((r[k]**2).mean()), r[k].max()) for k, d in enumerate(dets)}, best.x
print("\ndecomposition of the template: rigid detector placement + ONE polynomial distortion per focal plane")
print("residual per band [rms / max px] (this residual is the band-specific part)")
for dets, lab in (((1, 2, 3), "SWIR 1-3"), ((4, 5, 6), "MWIR 4-6"), ((1, 2, 3, 4, 5, 6), "all six")):
    for order in (3, 4, 5):
        r, p = decomp(dets, order)
        print(f"  {lab:9s} order {order}: " + "  ".join(f"D{d} {v[0]:.3f}/{v[1]:.3f}" for d, v in r.items()))
