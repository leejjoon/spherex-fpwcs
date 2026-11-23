"""Run the package's attitude fit over cached headers (week 2026W13_2A, and the year-long sample)."""
import json, sys, collections, numpy as np, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, "spherex-fpwcs/src")
import spherex_fpwcs as fp
T = fp.load_template()
def run(jsonl, out):
    H = [json.loads(l) for l in open(jsonl)]
    exps = collections.defaultdict(dict)
    for h in H:
        k = h["_key"]; n = k.split("/")[-1].split("_"); ex = k.split("/")[2] + ":" + n[3] + "_" + n[4][0]; d = int(n[4][2])
        if d not in exps[ex] or "retry" not in k: exps[ex][d] = h
    rows = []
    for ex, hd in exps.items():
        if len(hd) < 3: continue
        att = fp.fit_attitude(hd, T)
        for d, h in hd.items():
            r = att.residuals[d]; loo = None
            if len(hd) >= 4:
                loo = fp.fit_attitude(hd, T, exclude=(d,)).residuals[d]["max"]
            rows.append(dict(key=h["_key"], ex=ex, det=d, ndet=len(hd), rms=r["rms"], max=r["max"], frac=r["frac_gt1"], loo=loo,
                             scale=att.scale, date=h.get("DATE-OBS"), CONTRAST=h.get("CONTRAST"), DAXIS=h.get("DAXIS"),
                             crval=[h["CRVAL1"], h["CRVAL2"]]))
    json.dump(rows, open(out, "w")); return rows
for jsonl, out in (("out/week_headers.jsonl", "out/pkg_week.json"), ("out/long_headers.jsonl", "out/pkg_long.json")):
    rows = run(jsonl, out); mx = np.array([r["max"] for r in rows]); loo = np.array([r["loo"] if r["loo"] is not None else np.nan for r in rows])
    clean = mx < 0.3
    print(f"{out}: files={len(rows)}  >0.3px={int((mx>0.3).sum())} >0.5px={int((mx>.5).sum())} >1px={int((mx>1).sum())} >2px={int((mx>2).sum())} >5px={int((mx>5).sum())}")
    print(f"   clean files: all-six max  median={np.median(mx[clean]):.3f} p90={np.percentile(mx[clean],90):.3f} p99={np.percentile(mx[clean],99):.3f};"
          f"  leave-one-out max median={np.nanmedian(loo[clean]):.3f} p90={np.nanpercentile(loo[clean],90):.3f} p99={np.nanpercentile(loo[clean],99):.3f}")
    nb = collections.Counter(); 
    for r in rows:
        if r["max"] > 1: nb[r["ex"]] += 1
    print("   exposures by number of detectors >1px:", dict(sorted(collections.Counter(nb.values()).items())), " of", len(set(r["ex"] for r in rows)))
    print("   >1px by detector:", dict(sorted(collections.Counter(r["det"] for r in rows if r["max"] > 1).items())))
    multi = [e for e, c in nb.items() if c >= 2]
    for e in multi[:8]:
        print("     multi-bad", e, [(r["det"], round(r["max"], 2)) for r in rows if r["ex"] == e])
rows = json.load(open("out/pkg_long.json"))
bad = sorted([r for r in rows if r["max"] > 2.0], key=lambda r: -r["max"])
seen = set(); pick = []
for r in bad:
    wk = r["ex"].split(":")[0]
    if wk in seen or wk == "2026W13_2A": continue
    seen.add(wk); pick.append(r)
print("picks from other epochs:"); [print("  ", r["key"], round(r["max"], 2), r["date"][:10]) for r in pick[:8]]
ctrl = [next(r for r in rows if r["ex"].startswith(w) and r["max"] < 0.05 and r["det"] == d) for w, d in (("2025W17_4B", 3), ("2025W35_2A", 6), ("2025W51_2A", 1))]
json.dump([r["key"] for r in pick[:8]] + [r["key"] for r in ctrl], open("out/epoch_picks.json", "w"))
