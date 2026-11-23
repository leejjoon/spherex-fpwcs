"""Data for the GitHub Pages site (docs/index.html): sky images of the reported files with 2MASS
positions under the DR1 header WCS and under the spherex-fpwcs WCS, per-exposure consistency maps,
failure statistics and the star-validation table. Run from the analysis directory."""
import json, sys, collections, warnings, numpy as np
warnings.simplefilter("ignore")
sys.path.insert(0, "spherex-fpwcs/src")
import spherex_fpwcs as fp
from spherex_fpwcs.solve import celestial_wcs
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import zoom, median_filter
from PIL import Image

OUT = "spherex-fpwcs/docs/site"
T = fp.load_template()
N = 2040


def stretch(img):
    """Subtract a smooth background (the spectral image has strong gradients), asinh stretch."""
    bad = ~np.isfinite(img); a = np.where(bad, np.nan, img)
    B = 60; blocks = a[:N // B * B, :N // B * B].reshape(N // B, B, N // B, B)
    bg = np.nanmedian(blocks, axis=(1, 3)); bg = np.where(np.isfinite(bg), bg, np.nanmedian(bg))
    bg = zoom(median_filter(bg, 3), B, order=1, mode="nearest", grid_mode=True)
    d = a - bg; sig = 1.4826 * np.nanmedian(np.abs(d - np.nanmedian(d)))
    s = np.arcsinh(np.clip(d, -2 * sig, None) / (3 * sig)); s = (s - np.arcsinh(-2 / 3)) / (np.arcsinh(400.) - np.arcsinh(-2 / 3))
    s = np.where(bad, 0, np.clip(s, 0, 1))
    return (255 * s).astype(np.uint8)[::-1]          # row 0 = top = largest y


sky = []
for tag, cat, note in (("2D5", "D25", "reported"), ("1D4", "D14", "reported"), ("3D5", "D25", "reported"), ("2D3", "D36", "control")):
    f = f"data/dr1/{tag[2]}/level2_2026W13_2A_0568_{tag}_spx_l2b-v27-2026-237.fits"
    with fits.open(f) as h: hdr = h[1].header; img = h[1].data.astype(float)
    Image.fromarray(stretch(img), "L").save(f"{OUT}/sky_{tag}.png", optimize=True)
    res = fp.solve(f); w0, w1 = celestial_wcs(hdr), celestial_wcs(res.header)
    t = Table.read(f"cat/2mass_{cat}.fits")
    x0, y0 = w0.all_world2pix(t["ra"], t["dec"], 0, quiet=True); x1, y1 = w1.all_world2pix(t["ra"], t["dec"], 0, quiet=True)
    m = ((x0 > -15) & (x0 < N + 15) & (y0 > -15) & (y0 < N + 15)) | ((x1 > -15) & (x1 < N + 15) & (y1 > -15) & (y1 < N + 15))
    np.round(np.c_[x0, y0, x1, y1, np.asarray(t["k_m"], float)][m] * 10).astype("<i2").tofile(f"{OUT}/stars_{tag}.bin")   # x, y (DR1), x, y (new) in 0.1 px; K in 0.1 mag
    g = np.linspace(60, N - 60, 16); gx, gy = np.meshgrid(g, g)
    ra, dec = w1.all_pix2world(gx, gy, 0); hx, hy = w0.all_world2pix(ra, dec, 0, quiet=True)
    dg = res.diagnostics["target"]
    sky.append(dict(tag=tag, file=f.split("/")[-1], note=note, nstars=int(m.sum()), check_max=dg["max"], check_rms=dg["rms"],
                    grid=np.round(np.c_[gx.ravel(), gy.ravel(), (hx - gx).ravel(), (hy - gy).ravel()], 3).tolist()))
    print(tag, int(m.sum()), "stars; check max", round(dg["max"], 2))

# per-exposure consistency maps: header WCS minus focal-plane solution, in pixels
H = collections.defaultdict(dict)
for l in open("out/week_headers.jsonl"):
    h = json.loads(l); k = h["_key"]; n = k.split("/")[-1].split("_"); ex = n[3] + "_" + n[4][0]
    if ex in ("0568_1", "0568_2", "0568_3", "0568_4", "0491_2", "0140_3"):
        H[ex][int(n[4][2])] = h
labels = {"0568_2": "reported file 2D5", "0568_1": "reported file 1D4", "0568_3": "reported file 3D5", "0568_4": "same pointing, all consistent",
          "0491_2": "two inconsistent detectors", "0140_3": "largest error validated on stars"}
exps = []
for ex in labels:
    hd = {d: fits.Header({k: v for k, v in h.items() if not k.startswith("_") and isinstance(v, (int, float, str, bool))}) for d, h in H[ex].items()}
    att = fp.fit_attitude(hd, T); g = np.linspace(0, N - 1, 12); gx, gy = np.meshgrid(g, g); dets = {}
    for d, h in sorted(hd.items()):
        ra, dec = att.predict(d, gx.ravel(), gy.ravel()); hx, hy = celestial_wcs(h).all_world2pix(ra, dec, 0, quiet=True)
        dets[d] = dict(v=np.round(np.c_[hx - gx.ravel(), hy - gy.ravel()], 3).tolist(), max=att.residuals[d]["max"], rms=att.residuals[d]["rms"])
    exps.append(dict(id="2026W13_2A_" + ex, label=labels[ex], scale_ppm=(att.scale - 1) * 1e6, dets=dets))
    print(ex, {d: round(v["max"], 2) for d, v in dets.items()})

# failure statistics
def stats(path):
    rows = json.load(open(path)); c = SkyCoord([r["crval"][0] for r in rows] * u.deg, [r["crval"][1] for r in rows] * u.deg).galactic
    b = np.abs(c.b.deg); mx = np.array([r["max"] for r in rows]); det = np.array([r["det"] for r in rows])
    edges = [0, 2, 5, 10, 20, 90]
    return dict(n=len(rows), bad=int((mx > 1).sum()), marginal=int(((mx > 0.3) & (mx <= 1)).sum()),
                lat=[dict(lo=lo, hi=hi, n=int(((b >= lo) & (b < hi)).sum()), bad=int(((b >= lo) & (b < hi) & (mx > 1)).sum())) for lo, hi in zip(edges[:-1], edges[1:])],
                det=[dict(det=d, n=int((det == d).sum()), bad=int(((det == d) & (mx > 1)).sum())) for d in range(1, 7)])

# star validation
val = {}
for g in ("w13", "epochs", "epochs2", "multi"):
    for r in json.load(open(f"out/pkg_validate_{g}.json")):
        if not np.isfinite(r["header"]["max"]) and r["file"] in val: continue
        hm = r["header"]["max"]; grp = "control" if hm < 0.3 else ("two bad detectors" if g == "multi" else ("March 2026" if g == "w13" else "other months"))
        val[r["file"]] = dict(file=r["file"][7:26], date=r["date"], b=round(r["b"], 1), group=grp,
                              hdr_max=hm, hdr_rms=r["header"]["rms"], new_max=r["all6"]["max"], new_rms=r["all6"]["rms"])
val = [v for v in val.values() if np.isfinite(v["hdr_max"])]
print(len(val), "validated files")
json.dump(dict(sky=sky, exposures=exps, week=stats("out/pkg_week.json"), year=stats("out/pkg_long.json"), validation=val),
          open(f"{OUT}/data.json", "w"), separators=(",", ":"))
