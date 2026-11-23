"""Sample ~120 complete (6-detector) exposures from every 4th DR1 half-week; fetch IMAGE headers only."""
import json, subprocess, urllib.request, time, collections, re
from concurrent.futures import ThreadPoolExecutor
from astropy.io import fits
BASE = "https://nasa-irsa-spherex.s3.amazonaws.com/"
weeks = [w.strip("/\n") for w in open("out/dr1_weeks.txt")][::4]
if "2026W20_1A" not in weeks: weeks.append("2026W20_1A")
KEEP = re.compile(r"(CRPIX|CRVAL|CTYPE|CUNIT|CDELT)[12]$|PC\d_\d$|CD\d_\d$|[AB]P?_\d_\d$|[AB]P?_ORDER$|LONPOLE$|LATPOLE$|RADESYS$|"
                  r"FINAST$|DAXIS$|CONTRAST$|MATCH_NSTARS$|QS_NMATCH_VAR$|DETECTOR$|OBSID$|DATE-OBS$|MJD-OBS$|PSF_FWHM$|V[XYZ]_SC$")
def get(key, n=40320):
    err = ""
    for attempt in range(5):
        try:
            req = urllib.request.Request(BASE + key, headers={"Range": f"bytes=2880-{2880+n-1}"})
            h = fits.Header.fromstring(urllib.request.urlopen(req, timeout=60).read().decode("ascii", "replace"))
            if "CRVAL1" not in h and n < 200000: n *= 2; continue
            d = {k: (bool(v) if isinstance(v, bool) else v) for k, v in h.items() if KEEP.match(k)}
            d["_key"] = key; return d
        except Exception as e:
            err = repr(e); time.sleep(1 + attempt)
    return {"_key": key, "_error": err}
keys = []
for w in weeks:
    out = subprocess.run(["aws", "s3", "ls", "--no-sign-request", "--recursive", f"s3://nasa-irsa-spherex/dr1/level2/{w}/"],
                         capture_output=True, text=True).stdout
    ex = collections.defaultdict(dict)
    for l in out.splitlines():
        k = l.split()[3]
        if "retry" in k or not k.endswith(".fits"): continue
        n = k.split("/")[-1].split("_"); ex[n[3] + "_" + n[4][0]][int(n[4][2])] = k
    full = sorted(e for e, v in ex.items() if len(v) == 6)
    sel = full[::max(1, len(full)//120)][:120]
    print(w, "exposures:", len(ex), "complete:", len(full), "sampled:", len(sel), flush=True)
    for e in sel: keys += list(ex[e].values())
t0 = time.time()
with ThreadPoolExecutor(48) as p, open("out/long_headers.jsonl", "w") as f:
    for d in p.map(get, keys): f.write(json.dumps(d, default=str) + "\n")
print("fetched", len(keys), f"{time.time()-t0:.0f}s")
