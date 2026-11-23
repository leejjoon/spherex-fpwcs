"""Fit one telescope attitude (rotation + aberration scale) to the header WCS of several detectors."""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

from .template import Template, NPIX, plane_to_vec, vec_to_plane

_WCS_KEY = re.compile(r"(CRPIX|CRVAL|CTYPE|CUNIT|CDELT)[12]$|PC[12]_[12]$|CD[12]_[12]$|"
                      r"(A|B|AP|BP)_\d+_\d+$|(A|B|AP|BP)_ORDER$|LONPOLE$|LATPOLE$|RADESYS$|EQUINOX$")


def is_wcs_key(key: str) -> bool:
    """True for primary celestial WCS keywords (not the alternate 'A'/'W' systems)."""
    return bool(_WCS_KEY.match(key))


def celestial_wcs(header) -> WCS:
    """Build the primary celestial TAN-SIP WCS from a header (or mapping), ignoring everything else."""
    hd = fits.Header()
    for k in header.keys():
        if is_wcs_key(k):
            hd[k] = header[k]
    if "CRVAL1" not in hd or "CTYPE1" not in hd:
        raise ValueError("header has no celestial WCS (CRVAL1/CTYPE1 missing)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        return WCS(hd)


def radec_to_vec(ra_deg, dec_deg) -> np.ndarray:
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def vec_to_radec(v: np.ndarray):
    return np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0, np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))


def _kabsch(g: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Rotation R minimising |s - g R^T| (i.e. s_i ~ R g_i)."""
    u, _, vt = np.linalg.svd(g.T @ s)
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    return vt.T @ d @ u.T


def _similarity(pg, ps, w, fit_scale=True):
    """Weighted 2-D similarity ps ~ M pg + t."""
    sw = w / w.sum()
    mg, ms = (pg * sw[:, None]).sum(0), (ps * sw[:, None]).sum(0)
    a, b = pg - mg, ps - ms
    c = (w * (a * b).sum(1)).sum()
    s = (w * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])).sum()
    th = np.arctan2(s, c)
    scale = np.hypot(c, s) / (w * (a ** 2).sum(1)).sum() if fit_scale else 1.0
    m = scale * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return m, ms - m @ mg, scale


@dataclass
class Attitude:
    """Solved telescope attitude for one exposure."""

    template: Template
    rot: np.ndarray                      # 3x3, sky = rot @ template-frame vector
    m: np.ndarray                        # 2x2 in-plane similarity
    t: np.ndarray                        # 2, in-plane shift (radians)
    scale: float
    used: tuple
    residuals: dict = field(default_factory=dict)   # det -> dict(rms, max, frac_gt1, weight)

    def predict(self, det: int, x, y):
        """Predicted (ra, dec) in degrees for 0-based pixel coordinates of detector ``det``."""
        p = self.template.plane(det, x, y) @ self.m.T + self.t
        return vec_to_radec(plane_to_vec(p) @ self.rot.T)

    @property
    def boresight(self):
        """(ra, dec) in degrees of the template boresight for this exposure."""
        ra, dec = vec_to_radec(plane_to_vec(self.t[None, :]) @ self.rot.T)
        return float(ra[0]), float(dec[0])


def fit_attitude(headers: dict, template: Template, *, exclude=(), grid: int = 12,
                 robust_px: float = 0.15, fit_scale: bool = True, min_detectors: int = 3,
                 n_iter: int = 20) -> Attitude:
    """Fit the attitude to the header WCS of the given detectors.

    Parameters
    ----------
    headers : dict
        ``{detector: header}``; any mapping with the celestial WCS keywords works.
    exclude : iterable of int
        Detectors whose header is evaluated (diagnostics) but not used in the fit.
    robust_px : float
        Cauchy scale of the robust weighting, in pixels.  Points where a header disagrees
        with the common solution by much more than this are down-weighted.
    """
    exclude = {int(d) for d in exclude}
    g = np.linspace(0, NPIX - 1, grid)
    gx, gy = (a.ravel() for a in np.meshgrid(g, g))
    sky, tpl, dets = {}, {}, []
    for det, hdr in sorted(headers.items()):
        det = int(det)
        w = hdr if isinstance(hdr, WCS) else celestial_wcs(hdr)
        ra, dec = w.all_pix2world(gx, gy, 0)
        sky[det] = radec_to_vec(ra, dec)
        tpl[det] = template.plane(det, gx, gy)
        dets.append(det)
    used = tuple(d for d in dets if d not in exclude)
    if len(used) < min_detectors:
        raise ValueError(f"only {len(used)} detector(s) available for the fit; need {min_detectors}")

    pg = np.concatenate([tpl[d] for d in used])
    s = np.concatenate([sky[d] for d in used])
    rot = _kabsch(plane_to_vec(pg), s)
    ps = vec_to_plane(s @ rot)
    w = np.ones(len(pg))
    r0 = robust_px * template.pixel_rad
    for _ in range(n_iter):
        m, t, scale = _similarity(pg, ps, w, fit_scale)
        res = ps - (pg @ m.T + t)
        w = 1.0 / (1.0 + (res ** 2).sum(1) / r0 ** 2)

    att = Attitude(template=template, rot=rot, m=m, t=t, scale=float(scale), used=used)
    for det in dets:
        r = (vec_to_plane(sky[det] @ rot) - (tpl[det] @ m.T + t)) / template.pixel_rad
        d = np.hypot(r[:, 0], r[:, 1])
        att.residuals[det] = {"rms": float(np.sqrt((d ** 2).mean())), "max": float(d.max()),
                              "frac_gt1": float((d > 1.0).mean()), "used": det in used}
    return att
