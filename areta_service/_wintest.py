import base64, gzip, hashlib, json, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import areta_client
here = os.path.dirname(os.path.abspath(__file__))
data = json.loads(gzip.decompress(base64.b64decode(open(os.path.join(here, "_wintest.b64")).read().strip())))
pairs = [(p["src"], p["tgt"]) for p in data["pairs"]]
t0 = time.time()
res = areta_client.annotate(pairs, timeout=600)
toks = [[x["src"], x["tgt"], x["tag"]] for s in res for x in s]
h = hashlib.sha256(json.dumps(toks, ensure_ascii=False).encode()).hexdigest()
print("python", sys.version.split()[0], "| pasangan", len(pairs), "| token", len(toks), "| detik", round(time.time() - t0, 1))
print("hash", h)
print("COCOK dengan rujukan Linux:", h == "bd560cd93ac76efe47ff15d7729d86bc0a4950910ffbcdc6433c96a466c42146")
for x in res[0]:
    print(x["src"], "->", x["tgt"], x["tag"])
