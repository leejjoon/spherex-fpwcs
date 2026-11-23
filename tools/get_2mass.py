"""Fetch 2MASS PSC (K<10.5) around each of the three detector footprints."""
import pyvo, sys
tap = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
fields = {"D14": (282.064, -6.84), "D25": (278.18, -7.08), "D36": (274.28, -7.17)}
for name, (ra, dec) in fields.items():
    q = (f"SELECT ra, dec, j_m, h_m, k_m FROM fp_psc WHERE "
         f"CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},2.85))=1 AND k_m<10.5")
    job = tap.submit_job(q, maxrec=3000000); job.run(); job.wait(phases=["COMPLETED","ERROR","ABORTED"], timeout=900)
    job.raise_if_error()
    t = job.fetch_result().to_table()
    t.write(f"cat/2mass_{name}.fits", overwrite=True)
    print(name, len(t), flush=True)
