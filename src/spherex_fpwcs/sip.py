"""Represent a predicted pixel -> sky mapping as a FITS TAN-SIP header."""
from __future__ import annotations

import numpy as np
from astropy.io import fits

from .template import NPIX
from .solve import celestial_wcs

_S = 1000.0   # internal scaling of pixel offsets, for conditioning


def _gnomonic_deg(ra, dec, ra0, dec0):
    a, d, a0, d0 = np.radians(ra), np.radians(dec), np.radians(ra0), np.radians(dec0)
    den = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    xi = np.cos(d) * np.sin(a - a0) / den
    eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / den
    return np.degrees(xi), np.degrees(eta)


def fit_tan_sip(predict, *, order: int = 3, inv_order: int | None = None,
                crpix=(1020.5, 1020.5), grid: int = 32, linear: str = "pc"):
    """Fit a TAN-SIP WCS to ``predict(x, y) -> (ra_deg, dec_deg)`` (0-based pixels).

    Returns ``(header, info)``; ``info`` holds the representation errors in pixels:
    ``fwd_max`` (header vs. prediction, pixel -> sky) and ``inv_max`` (AP/BP polynomial
    vs. exact inverse).
    """
    if not 2 <= order <= 9:
        raise ValueError("SIP order must be between 2 and 9")
    inv_order = order + 1 if inv_order is None else inv_order
    x0, y0 = crpix[0] - 1.0, crpix[1] - 1.0
    ra0, dec0 = (float(v[0]) for v in predict(np.array([x0]), np.array([y0])))
    g = np.linspace(0, NPIX - 1, grid)
    gx, gy = (a.ravel() for a in np.meshgrid(g, g))
    ra, dec = predict(gx, gy)
    xi, eta = _gnomonic_deg(ra, dec, ra0, dec0)
    u, v = gx - x0, gy - y0

    terms = [(p, q) for p in range(order + 1) for q in range(order + 1 - p) if p + q >= 1]
    basis = np.stack([(u / _S) ** p * (v / _S) ** q for p, q in terms], axis=1)
    c = np.linalg.lstsq(basis, np.column_stack([xi, eta]), rcond=None)[0]
    c = {pq: c[i] / _S ** sum(pq) for i, pq in enumerate(terms)}
    cd = np.column_stack([c[(1, 0)], c[(0, 1)]])
    cdinv = np.linalg.inv(cd)
    ab = {pq: cdinv @ c[pq] for pq in terms if sum(pq) >= 2}

    # inverse polynomials: (u, v) = (U, V) + AP/BP(U, V) with (U, V) the distortion-corrected offsets
    fu = u + sum(k[0] * u ** p * v ** q for (p, q), k in ab.items())
    fv = v + sum(k[1] * u ** p * v ** q for (p, q), k in ab.items())
    iterms = [(p, q) for p in range(inv_order + 1) for q in range(inv_order + 1 - p)]
    ib = np.stack([(fu / _S) ** p * (fv / _S) ** q for p, q in iterms], axis=1)
    ic = np.linalg.lstsq(ib, np.column_stack([u - fu, v - fv]), rcond=None)[0]
    inv_max = float(np.abs(ib @ ic - np.column_stack([u - fu, v - fv])).max())
    apbp = {pq: ic[i] / _S ** sum(pq) for i, pq in enumerate(iterms)}

    h = fits.Header()
    h["WCSAXES"] = (2, "Number of coordinate axes")
    h["CRPIX1"] = (crpix[0], "Pixel coordinate of reference point")
    h["CRPIX2"] = (crpix[1], "Pixel coordinate of reference point")
    if linear == "pc":
        for i in (0, 1):
            for j in (0, 1):
                h[f"PC{i+1}_{j+1}"] = (float(cd[i, j]), "Coordinate transformation matrix element")
        h["CDELT1"] = (1.0, "[deg] Coordinate increment at reference point")
        h["CDELT2"] = (1.0, "[deg] Coordinate increment at reference point")
    elif linear == "cd":
        for i in (0, 1):
            for j in (0, 1):
                h[f"CD{i+1}_{j+1}"] = (float(cd[i, j]), "[deg/pixel] Linear transformation matrix")
    else:
        raise ValueError("linear must be 'pc' or 'cd'")
    h["CUNIT1"] = ("deg", "Units of coordinate increment and value")
    h["CUNIT2"] = ("deg", "Units of coordinate increment and value")
    h["CTYPE1"] = ("RA---TAN-SIP", "TAN (gnomonic) projection + SIP distortions")
    h["CTYPE2"] = ("DEC--TAN-SIP", "TAN (gnomonic) projection + SIP distortions")
    h["CRVAL1"] = (ra0, "[deg] Coordinate value at reference point")
    h["CRVAL2"] = (dec0, "[deg] Coordinate value at reference point")
    h["LONPOLE"] = (180.0, "[deg] Native longitude of celestial pole")
    h["LATPOLE"] = (dec0, "[deg] Native latitude of celestial pole")
    h["RADESYS"] = ("ICRS", "Equatorial coordinate system")
    for name, idx, table, n in (("A", 0, ab, order), ("B", 1, ab, order),
                                ("AP", 0, apbp, inv_order), ("BP", 1, apbp, inv_order)):
        h[f"{name}_ORDER"] = (n, f"SIP polynomial order, axis {idx}, "
                                 + ("detector to sky" if len(name) == 1 else "sky to detector"))
        for (p, q) in sorted(table):
            h[f"{name}_{p}_{q}"] = (float(table[(p, q)][idx]), "SIP distortion coefficient")

    # forward representation error on an offset grid
    w = celestial_wcs(h)
    t = np.linspace(7.3, NPIX - 8.7, grid + 5)
    tx, ty = (a.ravel() for a in np.meshgrid(t, t))
    pra, pdec = predict(tx, ty)
    bx, by = w.all_world2pix(pra, pdec, 0, tolerance=1e-8, quiet=True)
    fwd_max = float(np.hypot(bx - tx, by - ty).max())
    return h, {"fwd_max": fwd_max, "inv_max": inv_max, "order": order, "inv_order": inv_order}
