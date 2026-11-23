"""High-level interface: gather the six headers of an exposure, solve, and write the new WCS."""
from __future__ import annotations

import glob
import os
import shutil
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits

from . import remote
from .sip import fit_tan_sip
from .solve import Attitude, fit_attitude, is_wcs_key
from .template import DETECTORS, Template, load_template

OK_PX, BAD_PX = 0.3, 1.0


@dataclass
class Result:
    """Outcome of :func:`solve`."""

    detector: int
    exposure: str
    header: fits.Header            # new celestial WCS cards for the target detector
    attitude: Attitude
    diagnostics: dict
    sources: dict = field(default_factory=dict)   # det -> where its header came from

    def summary(self) -> str:
        d = self.diagnostics
        lines = [f"exposure {self.exposure}  target detector D{self.detector}",
                 f"detectors used in fit: {', '.join('D%d' % k for k in self.attitude.used)}"
                 f"   scale-1 = {d['scale_ppm']:+.1f} ppm",
                 "header WCS vs focal-plane solution [pixels]:",
                 "  det    rms     max   area>1px  verdict"]
        for det, r in sorted(d["detectors"].items()):
            mark = " <- target" if det == self.detector else ""
            used = "" if r["used"] else " (not used in fit)"
            lines.append(f"  D{det}  {r['rms']:6.3f} {r['max']:7.3f}   {100*r['frac_gt1']:5.1f}%   {r['verdict']}{used}{mark}")
        lines.append(f"new WCS: TAN-SIP order {d['sip']['order']}, representation error "
                     f"{d['sip']['fwd_max']:.4f} px (forward), {d['sip']['inv_max']:.4f} px (inverse)")
        return "\n".join(lines)


def _verdict(mx: float) -> str:
    return "consistent" if mx < OK_PX else ("marginal" if mx < BAD_PX else "INCONSISTENT")


def _image_header(path: str) -> fits.Header:
    with fits.open(path) as hdul:
        for hdu in hdul:
            if "CRVAL1" in hdu.header and str(hdu.header.get("CTYPE1", "")).startswith("RA"):
                return hdu.header.copy()
    raise ValueError(f"no celestial WCS found in {path}")


def _local_siblings(fid: remote.FileId, dirs) -> dict:
    found = {}
    for det in DETECTORS:
        if det == fid.det:
            continue
        pat = f"level2_{fid.week}_{fid.seq}_{fid.slew}D{det}_spx_*.fits"
        for d in dirs:
            for cand in (os.path.join(d, pat), os.path.join(d, str(det), pat),
                         os.path.join(d, os.pardir, str(det), pat)):
                hits = sorted(glob.glob(cand), key=lambda p: (fid.proc not in p, "retry" in p, p))
                if hits:
                    found[det] = hits[0]
                    break
            if det in found:
                break
    return found


def gather_headers(target, *, siblings=None, sibling_dir=None, offline=False,
                   releases=("dr1",), base_url=remote.DEFAULT_BASE_URL):
    """Collect ``{det: header}`` for the exposure of ``target``.

    ``target`` is a local path, an ``s3://`` / ``https://`` URL, or a bucket key.  Sibling headers
    are taken from ``siblings`` (explicit paths), then from local files next to the target (or in
    ``sibling_dir``), then from the public bucket unless ``offline``.
    """
    target = str(target)
    fid = remote.parse_name(target)
    headers, sources = {}, {}
    is_local = os.path.exists(target)
    if is_local:
        headers[fid.det] = _image_header(target)
        sources[fid.det] = target
    else:
        if offline:
            raise FileNotFoundError(f"{target} is not a local file and offline=True")
        url = remote.to_url(target, base_url)
        hdr = remote.read_remote_image_header(url)
        if hdr is None:
            raise FileNotFoundError(f"{url} not found")
        headers[fid.det], sources[fid.det] = hdr, url
        parts = urllib_key_parts(url)
        if parts and parts not in releases:
            releases = (parts,) + tuple(releases)

    for p in siblings or ():
        f2 = remote.parse_name(p)
        if f2.exposure != fid.exposure:
            raise ValueError(f"{p} belongs to exposure {f2.exposure}, not {fid.exposure}")
        if f2.det not in headers:
            headers[f2.det], sources[f2.det] = _image_header(str(p)), str(p)

    dirs = [d for d in ([sibling_dir] if sibling_dir else []) + ([os.path.dirname(os.path.abspath(target))] if is_local else []) if d]
    for det, path in _local_siblings(fid, dirs).items():
        if det not in headers:
            headers[det], sources[det] = _image_header(path), path

    missing = [d for d in DETECTORS if d not in headers]
    if missing and not offline:
        for det, (hdr, url) in remote.find_remote_headers(fid, missing, releases=releases, base_url=base_url).items():
            headers[det], sources[det] = hdr, url
    return fid, headers, sources


def urllib_key_parts(url: str):
    """Release name (first path component) of a bucket URL, e.g. 'dr1'."""
    from urllib.parse import urlparse
    path = urlparse(url).path.lstrip("/")
    first = path.split("/", 1)[0]
    return first if "/level2/" in "/" + path and first else None


def solve(target, *, template: Template | None = None, exclude_self: bool = False,
          sip_order: int = 3, linear: str | None = None, min_detectors: int = 3,
          headers: dict | None = None, **gather_kw) -> Result:
    """Solve the exposure attitude from the available header WCS and build a new WCS for ``target``.

    Pass ``headers={det: header}`` to skip file discovery entirely (``target`` is then only used
    for its name, or may be the detector number).
    """
    template = template or load_template()
    if headers is not None:
        if isinstance(target, int):
            det, exposure = target, "unknown"
        else:
            fid = remote.parse_name(str(target)); det, exposure = fid.det, fid.exposure
        sources = {d: "caller" for d in headers}
    else:
        fid, headers, sources = gather_headers(target, **gather_kw)
        det, exposure = fid.det, fid.exposure
    exclude = (det,) if exclude_self else ()
    att = fit_attitude(headers, template, exclude=exclude, min_detectors=min_detectors)
    if linear is None:
        linear = "cd" if (det in headers and "CD1_1" in headers[det]) else "pc"
    new, info = fit_tan_sip(lambda x, y: att.predict(det, x, y), order=sip_order, linear=linear)
    dets = {d: dict(r, verdict=_verdict(r["max"])) for d, r in att.residuals.items()}
    diag = {"detectors": dets, "scale_ppm": 1e6 * (att.scale - 1.0), "n_used": len(att.used),
            "boresight_deg": att.boresight, "sip": info,
            "target": dets.get(det), "template": template.name}
    return Result(detector=det, exposure=exposure, header=new, attitude=att, diagnostics=diag, sources=sources)


def check(target, **kw) -> dict:
    """Diagnostics only: how far each detector's header WCS is from the common focal-plane solution."""
    return solve(target, **kw).diagnostics


def replace_wcs(header: fits.Header, new: fits.Header) -> None:
    """Replace the primary celestial WCS cards of ``header`` in place.

    Existing keywords keep their position, new ones are inserted next to them, and primary WCS
    keywords absent from ``new`` (e.g. stale SIP terms) are removed.  Alternate WCS systems
    (``...A``, ``...W``) are not touched.
    """
    existing = [k for k in header.keys() if is_wcs_key(k)]
    stale = [k for k in existing if k not in new]
    prev = existing[-1] if existing else None
    for k in new:
        card = (k, new[k], new.comments[k])
        if k in header:
            header[k] = card[1:]
        elif prev is None:
            header.append(card, end=True)
        else:
            header.insert(prev, card, after=True)
        prev = k
    for k in stale:
        del header[k]


def apply(path, output=None, *, inplace: bool = False, overwrite: bool = False, **kw) -> Result:
    """Write a copy of ``path`` with the WCS of every celestial image HDU replaced.

    Returns the :class:`Result`.  Nothing is written unless ``output`` is given or ``inplace=True``.
    """
    from . import __version__
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if (output is None) == (not inplace):
        raise ValueError("give either output=... or inplace=True")
    res = solve(path, **kw)
    if not inplace:
        if os.path.exists(output) and not overwrite:
            raise FileExistsError(output)
        shutil.copyfile(path, output)
    dst = path if inplace else str(output)
    t = res.diagnostics["target"]
    with fits.open(dst, mode="update") as hdul:
        for hdu in hdul:
            h = hdu.header
            if "CRVAL1" not in h or not str(h.get("CTYPE1", "")).startswith("RA"):
                continue
            replace_wcs(h, res.header)
            h["FPWCS"] = (True, "WCS regenerated from focal-plane geometry")
            h["FPWVER"] = (__version__, "spherex-fpwcs version")
            h["FPWTMPL"] = (res.attitude.template.name, "focal-plane template")
            h["FPWNDET"] = (len(res.attitude.used), "detectors used in attitude fit")
            h["FPWSCALE"] = (round(res.diagnostics["scale_ppm"], 2), "[ppm] fitted plate-scale offset")
            if t is not None:
                h["FPWDMAX"] = (round(t["max"], 4), "[pix] max shift of new WCS vs original")
                h["FPWDRMS"] = (round(t["rms"], 4), "[pix] rms shift of new WCS vs original")
            h.add_history(f"spherex-fpwcs {__version__}: WCS replaced using detectors "
                          + ",".join(str(d) for d in res.attitude.used))
        hdul.flush()
    return res
