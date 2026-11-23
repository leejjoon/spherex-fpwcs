"""Fetch only the IMAGE-extension header of every listed file via HTTP range requests."""
import sys, json, urllib.request, time
from concurrent.futures import ThreadPoolExecutor
from astropy.io import fits
BASE = "https://nasa-irsa-spherex.s3.amazonaws.com/"
keys = [l.split()[3] for l in open("out/week_listing.txt") if l.strip().endswith(".fits")]

def get(key, n=40320):
    for attempt in range(5):
        try:
            req = urllib.request.Request(BASE + key, headers={"Range": f"bytes=2880-{2880+n-1}"})
            raw = urllib.request.urlopen(req, timeout=60).read()
            h = fits.Header.fromstring(raw.decode("ascii", "replace"))
            if "CRVAL1" not in h and n < 200000:
                n *= 2; continue
            d = {k: (v if not isinstance(v, bool) else bool(v)) for k, v in h.items()
                 if k not in ("COMMENT", "HISTORY", "") and not k.startswith(("KOVR", "KEN_", "KSN_"))}
            d["_key"] = key
            return d
        except Exception as e:
            err = repr(e); time.sleep(1 + attempt)
    return {"_key": key, "_error": err}

t0 = time.time()
with ThreadPoolExecutor(48) as ex, open("out/week_headers.jsonl", "w") as out:
    for i, d in enumerate(ex.map(get, keys)):
        out.write(json.dumps(d, default=str) + "\n")
        if i % 2000 == 0: print(i, f"{time.time()-t0:.0f}s", flush=True)
print("done", len(keys), f"{time.time()-t0:.0f}s")
