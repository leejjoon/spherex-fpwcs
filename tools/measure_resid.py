"""Measure WCS residuals (detected - catalogue-predicted pixel position) over a grid of cells.

For each cell, the offset is the peak of the 2D histogram of all pairwise (detected - predicted)
separations, which is robust to crowding and to offsets of many pixels.
"""
import sys, glob, json, warnings
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy import ndimage as ndi
warnings.simplefilter("ignore")

NCELL, R, KMAX = 8, 25, 9.5

def detect(img):
    bad = ~np.isfinite(img)
    im = np.where(bad, 0, img)
    bg = ndi.median_filter(im[::4, ::4], size=15)
    bg = ndi.zoom(bg, 4, order=1)[:im.shape[0], :im.shape[1]]
    d = im - bg
    sm = ndi.gaussian_filter(d, 0.8)
    mx = (sm == ndi.maximum_filter(sm, size=5)) & (sm > 0)
    y, x = np.nonzero(mx)
    ok = (x > 3) & (x < im.shape[1]-4) & (y > 3) & (y < im.shape[0]-4)
    y, x = y[ok], x[ok]
    pk = sm[y, x]
    o = np.argsort(pk)[::-1][:30000]
    y, x, pk = y[o], x[o], pk[o]
    # 5x5 centre of mass
    xs, ys = np.empty(len(x)), np.empty(len(x))
    gy, gx = np.mgrid[-2:3, -2:3]
    for i, (yy, xx) in enumerate(zip(y, x)):
        s = np.clip(d[yy-2:yy+3, xx-2:xx+3], 0, None); t = s.sum()
        xs[i] = xx + (s*gx).sum()/t if t > 0 else xx
        ys[i] = yy + (s*gy).sum()/t if t > 0 else yy
    return xs, ys, pk

def cell_offset(dx, dy):
    """Peak of pairwise-offset histogram, refined by median within 1.5 px."""
    if len(dx) < 30: return np.nan, np.nan, 0, 0.0
    H, _, _ = np.histogram2d(dx, dy, bins=2*R, range=[[-R, R], [-R, R]])
    Hs = ndi.uniform_filter(H, 3)*9
    i, j = np.unravel_index(np.argmax(Hs), Hs.shape)
    px, py = -R + i + 0.5, -R + j + 0.5
    for _ in range(3):
        m = np.hypot(dx-px, dy-py) < 1.5
        if m.sum() < 5: break
        px, py = np.median(dx[m]), np.median(dy[m])
    m = np.hypot(dx-px, dy-py) < 1.5
    bgexp = len(dx) * np.pi*1.5**2 / (2*R)**2
    return px, py, int(m.sum()), float(m.sum()/max(bgexp, 1e-9))

def run(f):
    tag = f.split("_0568_")[1][:3]
    det = int(tag[2])
    with fits.open(f) as h:
        hdr = h[1].header; img = h[1].data.astype(float)
    w = WCS(hdr)
    cat = Table.read(f"cat/2mass_D{'14' if det in (1,4) else '25' if det in (2,5) else '36'}.fits")
    cat = cat[cat["k_m"] < KMAX]
    cx, cy = w.all_world2pix(cat["ra"], cat["dec"], 0, quiet=True)
    ok = (cx > -R) & (cx < 2040+R) & (cy > -R) & (cy < 2040+R)
    cx, cy = cx[ok], cy[ok]
    xs, ys, pk = detect(img)
    n = min(len(xs), int(1.5*len(cx)))
    xs, ys = xs[:n], ys[:n]
    edges = np.linspace(0, 2040, NCELL+1)
    res = []
    for iy in range(NCELL):
        for ix in range(NCELL):
            x0, x1, y0, y1 = edges[ix], edges[ix+1], edges[iy], edges[iy+1]
            md = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
            mc = (cx >= x0-R) & (cx < x1+R) & (cy >= y0-R) & (cy < y1+R)
            ddx = (xs[md][:, None] - cx[mc][None, :]).ravel()
            ddy = (ys[md][:, None] - cy[mc][None, :]).ravel()
            k = (np.abs(ddx) < R) & (np.abs(ddy) < R)
            px, py, nm, sig = cell_offset(ddx[k], ddy[k])
            res.append(dict(ix=ix, iy=iy, xc=(x0+x1)/2, yc=(y0+y1)/2, dx=px, dy=py,
                            nmatch=nm, sig=sig, ndet=int(md.sum()), ncat=int(mc.sum())))
    keys = ["FINAST", "DAXIS", "CONTRAST", "MATCH_NSTARS", "QS_NMATCH_VAR", "ASTRONETSOLVED", "PSF_FWHM"]
    out = dict(tag=tag, file=f, hdr={k: (hdr[k] if not isinstance(hdr[k], (bool, np.bool_)) else bool(hdr[k])) for k in keys},
               ncat=int(len(cx)), ndet=int(n), cells=res)
    json.dump(out, open(f"out/resid_{tag}.json", "w"))
    r = np.array([np.hypot(c["dx"], c["dy"]) for c in res]); s = np.array([c["sig"] for c in res])
    print(f"{tag} DAXIS={hdr['DAXIS']:.3f} CONTRAST={hdr['CONTRAST']:5.1f} ncat={len(cx):5d} "
          f"median|d|={np.nanmedian(r):.2f} p90={np.nanpercentile(r,90):.2f} max={np.nanmax(r):.2f} px  minsig={s.min():.1f}", flush=True)

for f in sorted(glob.glob("data/dr1/*/*.fits")) if len(sys.argv) < 2 else sys.argv[1:]:
    run(f)
