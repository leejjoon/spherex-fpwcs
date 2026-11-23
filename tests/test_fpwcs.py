import json
import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import spherex_fpwcs as fp
from spherex_fpwcs import remote
from spherex_fpwcs.api import replace_wcs
from spherex_fpwcs.solve import celestial_wcs

DATA = json.loads((Path(__file__).parent / "data" / "exposure_2026W13_2A_0568_2.json").read_text())
HEADERS = {remote.parse_name(n).det: h for n, h in DATA["headers"].items()}
NAMES = {remote.parse_name(n).det: n for n in DATA["headers"]}


def test_template_layout():
    t = fp.load_template()
    assert sorted(t.coef) == [1, 2, 3, 4, 5, 6]
    c = {d: t.plane(d, [1019.5], [1019.5])[0] / t.pixel_rad for d in t.coef}
    # three detectors side by side, ~2267 px apart, the two focal planes overlapping on the sky
    assert abs(c[2][0]) < 5 and abs(c[5][0]) < 5
    assert 2200 < c[3][0] - c[2][0] < 2330 and 2200 < c[2][0] - c[1][0] < 2330
    for a, b in ((1, 4), (2, 5), (3, 6)):
        assert np.hypot(*(c[a] - c[b])) < 40
    # local pixel scale close to 6.15 arcsec
    p = t.plane(2, [1019.5, 1020.5], [1019.5, 1019.5]) / t.pixel_rad
    assert abs(np.hypot(*(p[1] - p[0])) - 1) < 0.01


def test_parse_name():
    f = remote.parse_name("some/dir/level2_2026W13_2A_0568_2D5_spx_l2b-v27-2026-237.fits")
    assert (f.week, f.seq, f.slew, f.det, f.proc) == ("2026W13_2A", "0568", "2", 5, "l2b-v27-2026-237")
    assert f.exposure == "2026W13_2A_0568_2"
    assert f.name(3) == "level2_2026W13_2A_0568_2D3_spx_l2b-v27-2026-237.fits"
    assert remote.to_url("s3://nasa-irsa-spherex/dr1/x.fits") == "https://nasa-irsa-spherex.s3.amazonaws.com/dr1/x.fits"
    with pytest.raises(ValueError):
        remote.parse_name("image.fits")


def test_bad_detector_is_flagged():
    att = fp.fit_attitude(HEADERS, fp.load_template())
    r = att.residuals
    assert r[5]["max"] > 5                       # the known-bad 2D5 solution
    assert all(r[d]["max"] < 0.35 for d in (1, 2, 3, 4, 6))
    assert abs(att.scale - 1) < 3e-4             # velocity aberration is at most ~1e-4


def test_self_exclusion_changes_nothing():
    t = fp.load_template()
    a = fp.fit_attitude(HEADERS, t)
    b = fp.fit_attitude(HEADERS, t, exclude=(5,))
    g = np.linspace(0, 2039, 9)
    x, y = (v.ravel() for v in np.meshgrid(g, g))
    ra1, de1 = a.predict(5, x, y)
    ra2, de2 = b.predict(5, x, y)
    sep = np.hypot((ra1 - ra2) * np.cos(np.radians(de1)), de1 - de2) * 3600 / 6.15
    assert sep.max() < 0.02


def test_new_wcs_matches_stars():
    """The regenerated WCS must agree with star positions measured on the image."""
    res = fp.solve(NAMES[5], headers=HEADERS)
    cells = np.array(DATA["star_truth_2D5"]["cells"])
    xc, yc, dx, dy = cells.T
    old = celestial_wcs(HEADERS[5])
    # a star truly at the cell centre has catalogue position old(centre - offset)
    ra, dec = old.all_pix2world(xc - dx, yc - dy, 0)
    nx, ny = celestial_wcs(res.header).all_world2pix(ra, dec, 0)
    err_new = np.hypot(nx - xc, ny - yc)
    err_old = np.hypot(dx, dy)
    assert err_old.max() > 7
    assert np.sqrt((err_new ** 2).mean()) < 0.06 and err_new.max() < 0.25
    assert res.diagnostics["target"]["verdict"] == "INCONSISTENT"
    assert res.diagnostics["sip"]["fwd_max"] < 0.005


@pytest.mark.parametrize("order", [3, 4, 5])
def test_sip_roundtrip(order):
    att = fp.fit_attitude(HEADERS, fp.load_template())
    hdr, info = fp.fit_tan_sip(lambda x, y: att.predict(2, x, y), order=order)
    assert info["fwd_max"] < 0.005 and info["inv_max"] < 0.005
    w = celestial_wcs(hdr)
    x = np.array([3.0, 1000.0, 2030.0]); y = np.array([2030.0, 500.0, 4.0])
    ra, dec = w.all_pix2world(x, y, 0)
    pra, pdec = att.predict(2, x, y)
    assert np.allclose(ra, pra, atol=2e-6) and np.allclose(dec, pdec, atol=2e-6)
    # the closed-form inverse (AP/BP polynomials, evaluated by hand) recovers the pixel
    bx, by = w.wcs_world2pix(ra, dec, 0)              # linear + projection only
    uu, vv = bx - 1019.5, by - 1019.5
    n = w.sip.ap_order
    fx = uu + sum(w.sip.ap[p, q] * uu ** p * vv ** q for p in range(n + 1) for q in range(n + 1 - p))
    fy = vv + sum(w.sip.bp[p, q] * uu ** p * vv ** q for p in range(n + 1) for q in range(n + 1 - p))
    assert np.allclose(fx + 1019.5, x, atol=0.01) and np.allclose(fy + 1019.5, y, atol=0.01)


def test_too_few_detectors():
    with pytest.raises(ValueError):
        fp.fit_attitude({5: HEADERS[5], 6: HEADERS[6]}, fp.load_template())


def _write(tmp, det):
    h = fits.Header()
    for k, v in HEADERS[det].items():
        h[k] = v
    h["CRPIX1A"] = 1.0
    h["CTYPE1A"] = "LINEAR"
    h["CTYPE1W"] = "WAVE-TAB"
    hdul = fits.HDUList([fits.PrimaryHDU(),
                         fits.ImageHDU(np.arange(16, dtype="f4").reshape(4, 4), header=h, name="IMAGE"),
                         fits.ImageHDU(np.ones((4, 4), "f4"), header=h, name="VARIANCE"),
                         fits.BinTableHDU.from_columns([fits.Column("a", "J", array=[1, 2])], name="EPSF")])
    path = os.path.join(tmp, NAMES[det])
    hdul.writeto(path)
    return path


def test_apply_offline(tmp_path):
    paths = {d: _write(tmp_path, d) for d in HEADERS}
    out = tmp_path / "fixed.fits"
    res = fp.apply(paths[5], out, offline=True)
    assert res.attitude.used == (1, 2, 3, 4, 5, 6)
    with fits.open(out) as new, fits.open(paths[5]) as old:
        for name in ("IMAGE", "VARIANCE"):
            h = new[name].header
            assert h["FPWCS"] is True and h["FPWNDET"] == 6 and h["FPWDMAX"] > 5
            assert "A_0_0" not in h and "A_1_0" not in h and "B_0_1" not in h   # stale non-standard terms removed
            assert h["CRPIX1A"] == 1.0 and h["CTYPE1W"] == "WAVE-TAB"            # alternate WCS untouched
            assert h["CTYPE1"] == "RA---TAN-SIP" and "PC1_1" in h and "CD1_1" not in h
            assert np.array_equal(new[name].data, old[name].data)
            assert abs(h["CRVAL1"] - old[name].header["CRVAL1"]) < 0.01
        assert "FPWCS" not in new["EPSF"].header
    with pytest.raises(FileExistsError):
        fp.apply(paths[5], out, offline=True)
    with pytest.raises(ValueError):
        fp.apply(paths[5], offline=True)


def test_replace_wcs_without_previous_wcs():
    h = fits.Header([("SIMPLE", True), ("FOO", 1)])
    new, _ = fp.fit_tan_sip(lambda x, y: fp.fit_attitude(HEADERS, fp.load_template()).predict(1, x, y))
    replace_wcs(h, new)
    assert h["FOO"] == 1 and h["CTYPE2"] == "DEC--TAN-SIP"


def test_cli(tmp_path, capsys):
    from spherex_fpwcs.cli import main
    paths = {d: _write(tmp_path, d) for d in HEADERS}
    assert main(["check", paths[5], "--offline"]) == 2          # target inconsistent
    assert main(["check", paths[2], "--offline"]) == 0
    capsys.readouterr()
    assert main(["header", paths[5], "--offline", "--exclude-self"]) == 0
    assert "CRVAL1" in capsys.readouterr().out
    assert main(["check", paths[5], "--offline", "--json"]) == 2
    d = json.loads(capsys.readouterr().out)
    assert d["detectors"]["5"]["verdict"] == "INCONSISTENT"


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("SPHEREX_FPWCS_NETWORK_TESTS"), reason="set SPHEREX_FPWCS_NETWORK_TESTS=1")
def test_remote_end_to_end():
    res = fp.solve("dr1/level2/2026W13_2A/l2b-v27-2026-237/5/" + NAMES[5])
    assert len(res.attitude.used) == 6 and res.diagnostics["target"]["max"] > 5
