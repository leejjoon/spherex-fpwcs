"""Focal-plane template: detector pixel -> gnomonic plane coordinates about the template boresight."""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources

import numpy as np

DETECTORS = (1, 2, 3, 4, 5, 6)
NPIX = 2040


@dataclass(frozen=True)
class Template:
    """Per-detector polynomial mapping 0-based pixel (x, y) to plane coordinates in radians."""

    name: str
    terms: np.ndarray          # (nterm, 2) integer exponents (p, q)
    coef: dict                 # det -> (nterm, 2) coefficients for (xi, eta)
    x0: float
    y0: float
    half: float
    pixel_scale_arcsec: float

    def plane(self, det: int, x, y) -> np.ndarray:
        """Plane coordinates (N, 2) in radians for 0-based pixel coordinates of detector ``det``."""
        u = (np.asarray(x, float).ravel() - self.x0) / self.half
        v = (np.asarray(y, float).ravel() - self.y0) / self.half
        basis = np.stack([u ** p * v ** q for p, q in self.terms], axis=1)
        return basis @ self.coef[int(det)]

    def vectors(self, det: int, x, y) -> np.ndarray:
        """Unit vectors (N, 3) in the template frame (boresight along +z)."""
        return plane_to_vec(self.plane(det, x, y))

    @property
    def pixel_rad(self) -> float:
        return np.radians(self.pixel_scale_arcsec / 3600.0)


def plane_to_vec(p: np.ndarray) -> np.ndarray:
    v = np.column_stack([p[:, 0], p[:, 1], np.ones(len(p))])
    return v / np.linalg.norm(v, axis=1)[:, None]


def vec_to_plane(v: np.ndarray) -> np.ndarray:
    return v[:, :2] / v[:, 2:3]


def load_template(path: str | None = None) -> Template:
    """Load a template JSON; with no argument, the template shipped with the package."""
    if path is None:
        text = resources.files("spherex_fpwcs").joinpath("data/fp_template_v1.json").read_text()
        name = "fp_template_v1"
    else:
        with open(path) as f:
            text = f.read()
        name = str(path)
    d = json.loads(text)
    coef = {int(k): np.column_stack([v["xi"], v["eta"]]) for k, v in d["detectors"].items()}
    return Template(name=name, terms=np.array(d["term_order"], int), coef=coef,
                    x0=d["norm"]["x0"], y0=d["norm"]["y0"], half=d["norm"]["half"],
                    pixel_scale_arcsec=d["pixel_scale_arcsec"])
