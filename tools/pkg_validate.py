"""End-to-end star validation of the package: residuals of 2MASS stars under the DR1 header WCS
and under the WCS produced by spherex-fpwcs (siblings fetched from S3 unless present locally)."""
import sys, os, glob, json, subprocess, warnings, numpy as np, pyvo
warnings.simplefilter("ignore")
sys.path.insert(0, "spherex-fpwcs/src")
import spherex_fpwcs as fp
from spherex_fpwcs.solve import celestial_wcs
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
src = open("scripts/remeasure_with_standard_sip.py").read().split("\nfor tag in sys.argv")[0]
ns = {}; exec(compile(src, "r", "exec"), ns); detect, measure = ns["detect"], ns["measure"]
tap = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
def catalog(name, w):
    cf = f"cat/pk_{name[:-5]}.fits"
    if not os.path.exists(cf):
        ra, dec = w.all_pix2world(1019.5, 1019.5, 0); b = SkyCoord(ra*u.deg, dec*u.deg).galactic.b.deg
        kmax = 10.5 if abs(b) < 5 else 13.0
        job = tap.submit_job(f"SELECT ra, dec, k_m FROM fp_psc WHERE CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},2.6))=1 AND k_m<{kmax}", maxrec=3000000)
        job.run(); job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=900); job.raise_if_error()
        job.fetch_result().to_table().write(cf, overwrite=True)
    t = Table.read(cf); t.sort("k_m"); return t
group, items = sys.argv[1], sys.argv[2:]
out = []
for it in items:
    if os.path.exists(it): f = it
    else:
        f = "data/spot/" + it.split("/")[-1]
        if not os.path.exists(f):
            subprocess.run(["aws", "s3", "cp", "--no-sign-request", "--only-show-errors", "s3://nasa-irsa-spherex/" + it, f], check=True)
    name = os.path.basename(f)
    with fits.open(f) as h: hdr = h[1].header; img = h[1].data.astype(float)
    w0 = celestial_wcs(hdr); cat = catalog(name, w0); xs, ys, _ = detect(img)
    c = SkyCoord(*w0.all_pix2world(1019.5, 1019.5, 0), unit="deg").galactic
    row = dict(group=group, file=name, date=hdr["DATE-OBS"][:10], l=float(c.l.deg), b=float(c.b.deg))
    for lab, kw in (("all6", {}), ("noself", {"exclude_self": True})):
        res = fp.solve(f, **kw); r = measure(celestial_wcs(res.header), cat, xs, ys); m = np.hypot(*r.T)
        row[lab] = dict(rms=float(np.sqrt((m**2).mean())), max=float(m.max()), cells=r.tolist())
        row["ndet"] = len(res.attitude.used) + (1 if kw else 0); row["diag"] = {str(k): v["max"] for k, v in res.diagnostics["detectors"].items()}
    r = measure(w0, cat, xs, ys); m = np.hypot(*r.T); row["header"] = dict(rms=float(np.sqrt((m**2).mean())), max=float(m.max()), cells=r.tolist())
    out.append(row)
    print(f"{name[7:26]} {row['date']} b={row['b']:6.1f} ndet={row['ndet']} | header {row['header']['rms']:.3f}/{row['header']['max']:.2f} | "
          f"fpwcs {row['all6']['rms']:.3f}/{row['all6']['max']:.3f} | fpwcs excl. self {row['noself']['rms']:.3f}/{row['noself']['max']:.3f}   [rms/max px]", flush=True)
json.dump(out, open(f"out/pkg_validate_{group}.json", "w"))
