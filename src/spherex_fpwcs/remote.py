"""Locate the other detectors of an exposure and read only their image headers.

Uses plain HTTPS against the public SPHEREx bucket (no credentials, standard library only).
"""
from __future__ import annotations

import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from astropy.io import fits

DEFAULT_BASE_URL = os.environ.get("SPHEREX_FPWCS_BASE_URL", "https://nasa-irsa-spherex.s3.amazonaws.com/")
_NAME = re.compile(r"level2_(?P<week>\d{4}W\d{2}_\d[A-Z])_(?P<seq>\d{4})_(?P<slew>\d)D(?P<det>\d)_spx_(?P<proc>.+)\.fits$")
BLOCK = 2880


@dataclass(frozen=True)
class FileId:
    week: str
    seq: str
    slew: str
    det: int
    proc: str

    @property
    def exposure(self) -> str:
        return f"{self.week}_{self.seq}_{self.slew}"

    def name(self, det: int | None = None, proc: str | None = None) -> str:
        return f"level2_{self.week}_{self.seq}_{self.slew}D{det or self.det}_spx_{proc or self.proc}.fits"


def parse_name(name: str) -> FileId:
    m = _NAME.search(os.path.basename(str(name)))
    if not m:
        raise ValueError(f"not a SPHEREx Level-2 file name: {name!r}")
    return FileId(m["week"], m["seq"], m["slew"], int(m["det"]), m["proc"])


def to_url(target: str, base_url: str = DEFAULT_BASE_URL) -> str:
    """Turn ``s3://bucket/key`` or a bare key into an HTTPS URL."""
    if target.startswith(("http://", "https://")):
        return target
    if target.startswith("s3://"):
        bucket, _, key = target[5:].partition("/")
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return base_url.rstrip("/") + "/" + target.lstrip("/")


def _get(url: str, headers=None, timeout=60, retries=4) -> bytes | None:
    """GET with retries; returns None on 404."""
    err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            err = e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            err = e
        time.sleep(1 + attempt)
    raise ConnectionError(f"failed to fetch {url}: {err!r}")


def _header_end(buf: bytes, start: int):
    """Offset just past the header starting at ``start`` (block aligned), or None if incomplete."""
    pos = start
    while pos + 80 <= len(buf):
        if buf[pos:pos + 8] == b"END     ":
            return ((pos + 80 - 1) // BLOCK + 1) * BLOCK
        pos += 80
    return None


def read_remote_image_header(url: str, size: int = 14 * BLOCK) -> fits.Header | None:
    """Header of the first image extension, fetched with a byte-range request. None if absent."""
    while True:
        buf = _get(url, headers={"Range": f"bytes=0-{size - 1}"})
        if buf is None:
            return None
        e0 = _header_end(buf, 0)
        if e0 is not None:
            primary = fits.Header.fromstring(buf[:e0].decode("ascii", "replace"))
            if primary.get("NAXIS", 0) > 0 and "CRVAL1" in primary:
                return primary
            e1 = _header_end(buf, e0)   # primary of SPHEREx L2 files carries no data
            if e1 is not None:
                return fits.Header.fromstring(buf[e0:e1].decode("ascii", "replace"))
        if len(buf) < size or size > 400 * BLOCK:
            raise ValueError(f"could not find an image header in {url}")
        size *= 2


def list_prefixes(prefix: str, base_url: str = DEFAULT_BASE_URL) -> list[str]:
    """Immediate sub-'directories' of a key prefix (S3 ListObjectsV2)."""
    q = urllib.parse.urlencode({"list-type": "2", "prefix": prefix, "delimiter": "/"})
    buf = _get(base_url.rstrip("/") + "/?" + q)
    if buf is None:
        return []
    root = ET.fromstring(buf)
    ns = root.tag[:root.tag.index("}") + 1] if root.tag.startswith("{") else ""
    return [e.findtext(f"{ns}Prefix") for e in root.findall(f"{ns}CommonPrefixes")]


def find_remote_headers(fid: FileId, dets, *, releases=("dr1",), base_url: str = DEFAULT_BASE_URL):
    """Fetch image headers of the given detectors of exposure ``fid``.

    Returns ``{det: (header, url)}`` for the detectors that exist.  The processing directory of
    the target is preferred; other directories of the same week (e.g. ``l2b_retry-*``) are tried next.
    """
    out = {}
    for release in releases:
        dirs = list_prefixes(f"{release}/level2/{fid.week}/", base_url)
        procs = [d.rstrip("/").rsplit("/", 1)[-1] for d in dirs]
        procs.sort(key=lambda p: (p != fid.proc, "retry" in p, p))
        for det in dets:
            if det in out:
                continue
            for proc in procs:
                url = to_url(f"{release}/level2/{fid.week}/{proc}/{det}/{fid.name(det, proc)}", base_url)
                hdr = read_remote_image_header(url)
                if hdr is not None:
                    out[det] = (hdr, url)
                    break
        if len(out) == len(list(dets)):
            break
    return out
