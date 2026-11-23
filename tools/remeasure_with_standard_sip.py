"""Re-measure star residuals on the image using a WCS with a band-typical distortion
(SIP from a file that is within 0.02 px of the week median) and refit pointing + linear matrix."""
import sys, glob, json, warnings, numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.optimize import least_squares
warnings.simplefilter("ignore")
src = open("scripts/measure_resid.py").read().split("\nfor f in sorted(")[0]
ns = {}; exec(compile(src, "m", "exec"), ns)
detect, cell_offset, NCELL, R = ns["detect"], ns["cell_offset"], ns["NCELL"], ns["R"]
DONOR = {2: "data/dr1/2/*_3D2_*", 4: "data/dr1/4/*_4D4_*", 5: "data/dr1/5/*_4D5_*", 6: "data/dr1/6/*_3D6_*"}
e = np.linspace(0, 2040, NCELL+1); cc = (e[:-1]+e[1:])/2; CX, CY = np.meshgrid(cc, cc); C = np.c_[CX.ravel(), CY.ravel()]

def measure(w, cat, xs, ys):
    cx, cy = w.all_world2pix(cat["ra"], cat["dec"], 0, quiet=True)
    ok = (cx > -R) & (cx < 2040+R) & (cy > -R) & (cy < 2040+R); cx, cy = np.asarray(cx[ok])[:25000], np.asarray(cy[ok])[:25000]
    n = min(len(xs), int(1.5*len(cx))); x, y = xs[:n], ys[:n]; res = []
    for iy in range(NCELL):
        for ix in range(NCELL):
            md = (x >= e[ix]) & (x < e[ix+1]) & (y >= e[iy]) & (y < e[iy+1])
            mc = (cx >= e[ix]-R) & (cx < e[ix+1]+R) & (cy >= e[iy]-R) & (cy < e[iy+1]+R)
            ddx = (x[md][:, None]-cx[mc][None, :]).ravel(); ddy = (y[md][:, None]-cy[mc][None, :]).ravel()
            k = (np.abs(ddx) < R) & (np.abs(ddy) < R)
            res.append(cell_offset(ddx[k], ddy[k])[:2])
    return np.array(res)

def refit(w_cur, r, w_donor):
    sky = w_cur.all_pix2world(C - r, 0)
    w = w_donor.deepcopy(); lin = w_cur.wcs.cd if w_cur.wcs.has_cd() else w_cur.wcs.get_pc()
    def model(p):
        w.wcs.crval = p[:2]
        if w.wcs.has_cd(): w.wcs.cd = p[2:].reshape(2, 2)
        else: w.wcs.pc = p[2:].reshape(2, 2)
        w.wcs.set(); return w
    f = lambda p: (model(p).all_world2pix(sky, 0, quiet=True) - C).ravel()
    # start from donor linear terms: donor CRPIX is the detector centre, so use sky of centre under current WCS
    p0 = np.r_[w_cur.all_pix2world([[1019.5, 1019.5]], 0)[0], (w_donor.wcs.cd if w_donor.wcs.has_cd() else w_donor.wcs.get_pc()).ravel()]
    s = least_squares(f, p0, x_scale=np.r_[1e-3, 1e-3, [1e-6]*4])
    return model(s.x).deepcopy()

for tag in sys.argv[1:]:
    f = (glob.glob(f"data/spot/*_{tag}_*") + glob.glob(f"data/dr1/*/*_{tag}_*"))[0]
    with fits.open(f) as h: hdr = h[1].header; img = h[1].data.astype(float)
    cf = f"cat/spot_{tag}.fits"
    if tag.startswith("0568"):
        d = hdr["DETECTOR"]; cf = f"cat/2mass_D{'14' if d in (1,4) else '25' if d in (2,5) else '36'}.fits"
    cat = Table.read(cf); cat.sort("k_m")
    xs, ys, _ = detect(img)
    w = WCS(hdr); wd = WCS(fits.getheader(glob.glob(DONOR[hdr["DETECTOR"]])[0], 1))
    r = measure(w, cat, xs, ys); m = np.hypot(*r.T)
    line = f"{tag}: header rms={np.sqrt((m**2).mean()):.2f} max={m.max():.2f}"
    for it in range(3):
        w = refit(w, r, wd); r = measure(w, cat, xs, ys); m = np.hypot(*r.T)
        line += f" | pass{it+1} rms={np.sqrt((m**2).mean()):.3f} max={m.max():.3f}"
    print(line + " px", flush=True)
