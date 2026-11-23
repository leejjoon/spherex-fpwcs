"""Long-term stability of (1) per-detector distortion shape and (2) six-detector focal-plane geometry."""
import json, re, os, warnings, collections, numpy as np
from astropy.io import fits
from astropy.wcs import WCS
warnings.simplefilter("ignore")
RAD2PX = 206264.806 / 6.15
H = [json.loads(l) for l in open("out/long_headers.jsonl")]
e = np.linspace(0, 2040, 9); cc = (e[:-1]+e[1:])/2; CX, CY = np.meshgrid(cc, cc); CX, CY = CX.ravel(), CY.ravel(); NP = 64
WK = re.compile(r"(CRPIX|CRVAL|CTYPE|CUNIT|CDELT)[12]$|PC\d_\d$|CD\d_\d$|[AB]P?_\d_\d$|[AB]P?_ORDER$|LONPOLE$|LATPOLE$|RADESYS$")
def vec(h):
    hd = fits.Header()
    for k, v in h.items():
        if WK.match(k): hd[k] = v
    ra, dec = np.radians(WCS(hd).all_pix2world(CX, CY, 0))
    return np.c_[np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)]
if os.path.exists("out/long_vec.npy"): V = np.load("out/long_vec.npy")
else: V = np.array([vec(h) for h in H]); np.save("out/long_vec.npy", V)
wk = np.array([h["_key"].split("/")[2] for h in H]); det = np.array([h["DETECTOR"] for h in H])
exid = np.array([h["_key"].split("/")[2] + ":" + h["_key"].split("/")[-1].split("_")[3] + "_" + h["_key"].split("/")[-1].split("_")[4][0] for h in H])
weeks = sorted(set(wk)); REF = "2026W13_2A"
date = {w: sorted(h["DATE-OBS"] for h, ww in zip(H, wk) if ww == w)[60*6][:10] for w in weeks}
A1 = lambda P: np.c_[P, np.ones(len(P))]
def frame(z):
    z = z/np.linalg.norm(z); x = np.cross([0, 0, 1.], z); x /= np.linalg.norm(x); return np.array([x, np.cross(z, x), z])
def plane(v, fr):
    p = v @ fr.T; return p[:, :2]/p[:, 2:3]*RAD2PX
def unplane(p, fr):
    v = np.c_[p/RAD2PX, np.ones(len(p))]; v /= np.linalg.norm(v, axis=1)[:, None]; return v @ fr

# ---------- (1) per-detector shape, affine removed
P = np.array([plane(v, frame(v.mean(0))) for v in V])
def align(p, R): return A1(p) @ np.linalg.lstsq(A1(p), R, rcond=None)[0]
shape = {}; devfile = np.zeros(len(H))
for d in range(1, 7):
    idx = np.nonzero((det == d) & (wk == REF))[0]; R0 = P[idx[0]]
    for it in range(3): R0 = np.median([align(P[i], R0) for i in idx], axis=0)
    for w in weeks:
        idx = np.nonzero((det == d) & (wk == w))[0]; al = np.array([align(P[i], R0) for i in idx])
        E = np.median(al, axis=0); shape[w, d] = E - R0
        devfile[idx] = np.hypot(*(al - E).transpose(2, 0, 1)).max(1)      # deviation from own epoch median
print("(1) per-detector distortion shape: epoch median minus 2026W13 median, affine removed  [max px over detector]")
print("epoch       date        " + "     ".join(f"D{d}" for d in range(1, 7)) + "    | files >1px from own epoch median")
for w in weeks:
    m = [np.hypot(*shape[w, d].T).max() for d in range(1, 7)]
    print(f"{w}  {date[w]}  " + "  ".join(f"{x:.3f}" for x in m) + f"   | {int((devfile[wk == w] > 1).sum())}/720")
clean = devfile < 0.3

# ---------- (2) six-detector template, similarity per exposure
def kabsch(G, S):
    U, _, Vt = np.linalg.svd(G.T @ S); return Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
def sim(pg, ps, w):
    sw = w/w.sum(); mg, ms = (pg*sw[:, None]).sum(0), (ps*sw[:, None]).sum(0); a, b = pg-mg, ps-ms
    c = (w*(a*b).sum(1)).sum(); s_ = (w*(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])).sum(); th = np.arctan2(s_, c)
    sc = np.hypot(c, s_)/(w*(a**2).sum(1)).sum(); M = sc*np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return M, ms - M @ mg, sc
def solve(G2, Gv, fr, S, mask):
    R = kabsch(Gv[mask], S[mask]); ps = plane(S @ R, fr); w = mask.astype(float)
    for it in range(15):
        M, t, sc = sim(G2, ps, np.maximum(w, 1e-12)*mask); res = ps - (G2 @ M.T + t); w = 1/(1 + (res**2).sum(1)/0.15**2)
    return R, M, t, res, sc
exps = collections.defaultdict(dict)
for i in range(len(H)): exps[exid[i]][det[i]] = i
stack = lambda dd: np.concatenate([V[dd[d]] for d in range(1, 7)])
ref_ex = [k for k in exps if k.startswith(REF)]
S0 = stack(exps[ref_ex[0]]); fr = frame(S0.mean(0)); G2 = plane(S0, fr); allm = np.ones(6*NP, bool)
def template(keys, G2):
    Gv = unplane(G2, fr); acc = []
    for k in keys:
        S = stack(exps[k]); R, M, t, res, sc = solve(G2, Gv, fr, S, allm); acc.append((plane(S @ R, fr) - t) @ np.linalg.inv(M).T)
    return np.median(acc, axis=0)
for it in range(4): G2 = template(ref_ex, G2)
Gv = unplane(G2, fr)
print("\n(2) six-detector focal-plane template: epoch median minus 2026W13 template (one similarity per exposure removed)")
print("    'shift' = mean displacement of the detector as a whole, 'max' = largest displacement of any point  [px]")
print("epoch       date        " + "   ".join(f"D{d} shift/max" for d in range(1, 7)))
drift = {}
for w in weeks:
    T = template([k for k in exps if k.startswith(w)], G2); D = T - G2; drift[w] = D; row = []
    for d in range(6):
        dd = D[d*NP:(d+1)*NP]; row.append(f"{np.hypot(*dd.mean(0)):.2f}/{np.hypot(*dd.T).max():.2f}")
    print(f"{w}  {date[w]}  " + "    ".join(row))
np.save("out/long_template_drift.npy", np.array([drift[w] for w in weeks]))

# ---------- (3) practical: leave-one-out prediction of each clean file using (a) the fixed 2026W13 template, (b) own-epoch template
print("\n(3) star-free leave-one-out prediction vs header WCS, clean files only: max deviation per file [median / p90 px]")
print("epoch       date        fixed 2026W13 template | own-epoch template | scale-1 (ppm, median)")
for w in weeks:
    keys = [k for k in exps if k.startswith(w)]; Tw = G2 + drift[w]; out = {"f": [], "o": []}; scs = []
    for lab, T in (("f", G2), ("o", Tw)):
        Tv = unplane(T, fr)
        for k in keys:
            S = stack(exps[k])
            for d in range(6):
                i = exps[k][d+1]
                if not clean[i]: continue
                m = allm.copy(); m[d*NP:(d+1)*NP] = False
                R, M, t, _, sc = solve(T, Tv, fr, S, m); scs.append(sc)
                out[lab].append(np.hypot(*(plane(S[d*NP:(d+1)*NP] @ R, fr) - (T[d*NP:(d+1)*NP] @ M.T + t)).T).max())
    f, o = np.array(out["f"]), np.array(out["o"])
    print(f"{w}  {date[w]}   {np.median(f):.3f} / {np.percentile(f,90):.3f}         |  {np.median(o):.3f} / {np.percentile(o,90):.3f}     | {1e6*(np.median(scs)-1):+.0f}")
