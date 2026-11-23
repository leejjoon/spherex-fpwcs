"""Build the six-detector focal-plane template from the year-long header sample (out/long_headers.jsonl).
Output: template JSON with a per-detector polynomial  pixel -> focal-plane tangent coords (radians)."""
import json, re, os, warnings, collections, numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
import astropy.units as u
warnings.simplefilter("ignore")
RAD2PX = 206264.806 / 6.15
H = [json.loads(l) for l in open("out/long_headers.jsonl")]
g = np.linspace(0, 2039, 13); GX, GY = np.meshgrid(g, g); GX, GY = GX.ravel(), GY.ravel(); NP = len(GX)
WK = re.compile(r"(CRPIX|CRVAL|CTYPE|CUNIT|CDELT)[12]$|PC\d_\d$|CD\d_\d$|[AB]P?_\d_\d$|[AB]P?_ORDER$|LONPOLE$|LATPOLE$|RADESYS$")
def vec(h):
    hd = fits.Header()
    for k, v in h.items():
        if WK.match(k): hd[k] = v
    ra, dec = np.radians(WCS(hd).all_pix2world(GX, GY, 0))
    return np.c_[np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)]
if os.path.exists("out/long_vec13.npy"): V = np.load("out/long_vec13.npy")
else: V = np.array([vec(h) for h in H]); np.save("out/long_vec13.npy", V)
det = np.array([h["DETECTOR"] for h in H])
exps = collections.defaultdict(dict)
for i, h in enumerate(H):
    n = h["_key"].split("/")[-1].split("_"); exps[h["_key"].split("/")[2] + ":" + n[3] + "_" + n[4][0]][det[i]] = i
keys = list(exps); stack = lambda k: np.concatenate([V[exps[k][d]] for d in range(1, 7)])
def frame(z):
    z = z/np.linalg.norm(z); x = np.cross([0, 0, 1.], z); x /= np.linalg.norm(x); return np.array([x, np.cross(z, x), z])
def plane(v, fr):
    p = v @ fr.T; return p[:, :2]/p[:, 2:3]*RAD2PX
def unplane(p, fr):
    v = np.c_[p/RAD2PX, np.ones(len(p))]; v /= np.linalg.norm(v, axis=1)[:, None]; return v @ fr
def kabsch(G, S):
    U, _, Vt = np.linalg.svd(G.T @ S); return Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
def sim(pg, ps, w):
    sw = w/w.sum(); mg, ms = (pg*sw[:, None]).sum(0), (ps*sw[:, None]).sum(0); a, b = pg-mg, ps-ms
    c = (w*(a*b).sum(1)).sum(); s_ = (w*(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])).sum(); th = np.arctan2(s_, c)
    sc = np.hypot(c, s_)/(w*(a**2).sum(1)).sum(); M = sc*np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return M, ms - M @ mg, sc
def solve(G2, Gv, fr, S):
    R = kabsch(Gv, S); ps = plane(S @ R, fr); w = np.ones(len(S))
    for it in range(15):
        M, t, sc = sim(G2, ps, w); res = ps - (G2 @ M.T + t); w = 1/(1 + (res**2).sum(1)/0.15**2)
    return R, M, t, res, sc
k0 = next(k for k in keys if k.startswith("2026W13_2A"))
S0 = stack(k0); fr = frame(S0.mean(0)); G2 = plane(S0, fr)
for it in range(5):
    Gv = unplane(G2, fr); acc = []; scs = []; bores = []
    for k in keys:
        S = stack(k); R, M, t, res, sc = solve(G2, Gv, fr, S)
        acc.append((plane(S @ R, fr) - t) @ np.linalg.inv(M).T); scs.append(sc); bores.append(R @ fr[2])
    new = np.median(acc, axis=0); print("iter", it, "template change max px:", np.abs(new - G2).max()); G2 = new
scs = np.array(scs); G2 = G2 / 1.0
# normalise the template scale so that the MEDIAN exposure has scale exactly 1
G2 = G2 * np.median(scs); scs = scs / np.median(scs)
# rotate plane so detector-2 pixel +x runs along +xi
d2 = G2[NP:2*NP].reshape(13, 13, 2); dx = (d2[:, -1] - d2[:, 0]).mean(0); th = -np.arctan2(dx[1], dx[0])
Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); G2 = G2 @ Rz.T
# ---- aberration check: scale vs beta*cos(angle to observer velocity)
t = Time([H[exps[k][1]]["MJD-OBS"] for k in keys], format="mjd")
ve = get_body_barycentric_posvel("earth", t)[1].xyz.to(u.km/u.s).value.T
vs = np.array([[H[exps[k][1]].get(f"V{a}_SC", 0.0) for a in "XYZ"] for k in keys])
for lab, v in (("Earth orbital velocity only", ve), ("Earth + spacecraft velocity", ve + vs)):
    pred = (np.array(bores) * v).sum(1) / 299792.458
    A = np.c_[pred, np.ones(len(pred))]; sl, ic = np.linalg.lstsq(A, scs - 1, rcond=None)[0]
    print(f"scale vs aberration ({lab}): slope={sl:.3f} corr={np.corrcoef(pred, scs-1)[0,1]:.4f} "
          f"scale rms before={1e6*np.std(scs):.1f} ppm, after removing prediction={1e6*np.std(scs-1-sl*pred-ic):.1f} ppm; with slope fixed to 1: {1e6*np.std(scs-1-pred):.1f} ppm")
np.save("out/long_scale_aberr.npy", np.c_[scs, (np.array(bores)*(ve+vs)).sum(1)/299792.458, t.mjd])
# ---- per-detector polynomial  (u,v in [-1,1]) -> plane radians
def terms(uu, vv, order): return np.stack([uu**p * vv**q for p in range(order+1) for q in range(order+1-p)], 1)
uu, vv = (GX - 1019.5)/1020.0, (GY - 1019.5)/1020.0
out = {"description": "SPHEREx six-detector focal-plane template (DR1, 2025-04 .. 2026-05)", "pixel_scale_arcsec": 6.15,
       "order": 5, "norm": {"x0": 1019.5, "y0": 1019.5, "half": 1020.0}, "n_exposures": len(keys), "units": "radians (gnomonic plane about template boresight)",
       "term_order": [[p, q] for p in range(6) for q in range(6-p)], "detectors": {}}
for order in (3, 4, 5, 6):
    r = []
    for d in range(6):
        A = terms(uu, vv, order); tg = G2[d*NP:(d+1)*NP]; c = np.linalg.lstsq(A, tg, rcond=None)[0]; r.append(np.hypot(*(tg - A @ c).T).max())
        if order == 5: out["detectors"][str(d+1)] = {"xi": (c[:, 0]/RAD2PX).tolist(), "eta": (c[:, 1]/RAD2PX).tolist()}
    print(f"polynomial order {order}: max fit residual per detector (px):", np.round(r, 4))
json.dump(out, open("out/fp_template_v1.json", "w"), indent=1)
np.save("out/fp_template_grid13.npy", G2)
print("detector centres in template plane (px):", np.round([G2[d*NP:(d+1)*NP].mean(0) for d in range(6)], 1).tolist())
