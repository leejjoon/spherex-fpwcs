"""Command line interface: ``spherex-fpwcs {check,header,apply}``."""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__, api


def _common(p):
    p.add_argument("file", help="Level-2 FITS file (local path, s3:// or https:// URL, or bucket key)")
    p.add_argument("--sibling", action="append", default=[], metavar="FITS",
                   help="local file of another detector of the same exposure (repeatable)")
    p.add_argument("--sibling-dir", help="directory to search for the other detectors")
    p.add_argument("--offline", action="store_true", help="never contact the bucket")
    p.add_argument("--release", action="append", metavar="NAME",
                   help="bucket release(s) to search for siblings (default: dr1)")
    p.add_argument("--exclude-self", action="store_true",
                   help="do not use the target's own WCS in the attitude fit")
    p.add_argument("--sip-order", type=int, default=3, help="order of the output SIP polynomial (default 3)")
    p.add_argument("--min-detectors", type=int, default=3, help="minimum detectors in the fit (default 3)")
    p.add_argument("--template", help="alternative template JSON")
    p.add_argument("--json", action="store_true", help="print diagnostics as JSON")


def _kwargs(a):
    from .template import load_template
    kw = dict(siblings=a.sibling, sibling_dir=a.sibling_dir, offline=a.offline,
              exclude_self=a.exclude_self, sip_order=a.sip_order, min_detectors=a.min_detectors)
    if a.release:
        kw["releases"] = tuple(a.release)
    if a.template:
        kw["template"] = load_template(a.template)
    return kw


def _report(res, a):
    if a.json:
        d = dict(res.diagnostics, exposure=res.exposure, detector=res.detector,
                 sources={str(k): v for k, v in res.sources.items()})
        d["detectors"] = {str(k): v for k, v in d["detectors"].items()}
        print(json.dumps(d, indent=1))
    else:
        print(res.summary(), file=sys.stderr if a.cmd == "header" and not a.output else sys.stdout)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="spherex-fpwcs", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check", help="compare each detector's header WCS with the common solution")
    _common(p)
    p = sub.add_parser("header", help="print the regenerated WCS header cards")
    _common(p)
    p.add_argument("-o", "--output", help="write the cards to this text file instead of stdout")
    p = sub.add_parser("apply", help="write a FITS file with the regenerated WCS")
    _common(p)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("-o", "--output", help="output FITS file")
    g.add_argument("--inplace", action="store_true", help="modify the input file")
    p.add_argument("--overwrite", action="store_true", help="overwrite an existing output file")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "apply":
            res = api.apply(a.file, a.output, inplace=a.inplace, overwrite=a.overwrite, **_kwargs(a))
            _report(res, a)
            print(f"wrote {a.file if a.inplace else a.output}")
        else:
            res = api.solve(a.file, **_kwargs(a))
            if a.cmd == "header":
                text = res.header.tostring(sep="\n", endcard=False, padding=False) + "\n"
                if a.output:
                    with open(a.output, "w") as f:
                        f.write(text)
                    _report(res, a)
                else:
                    _report(res, a)
                    sys.stdout.write(text)
            else:
                _report(res, a)
                t = res.diagnostics["target"]
                return 0 if t is None or t["max"] < api.BAD_PX else 2
    except (ValueError, FileNotFoundError, FileExistsError, ConnectionError) as e:
        print(f"spherex-fpwcs: error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
