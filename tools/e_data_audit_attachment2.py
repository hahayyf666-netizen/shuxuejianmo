from __future__ import annotations
import gc, hashlib, json, pickle, re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")
MANIFEST = ROOT / "dataset_manifest.json"
OUT = ROOT / "_e_data_audit_out"
OUT.mkdir(exist_ok=True)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    return x


def sha256_array(arr, chunk_rows=64):
    h = hashlib.sha256()
    if arr.ndim == 0:
        h.update(np.ascontiguousarray(arr).tobytes())
    else:
        for i in range(0, arr.shape[0], chunk_rows):
            h.update(np.ascontiguousarray(arr[i:i+chunk_rows]).tobytes())
    return h.hexdigest()


def array_stats(arr, block_rows=32):
    if not np.issubdtype(arr.dtype, np.number):
        return {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    finite_count = nan_count = inf_count = 0
    mn = None; mx = None
    total = 0.0; total2 = 0.0
    if arr.ndim == 0:
        blocks = [arr.reshape(1)]
    else:
        blocks = (arr[i:i+block_rows] for i in range(0, arr.shape[0], block_rows))
    for b in blocks:
        bb = np.asarray(b)
        if np.issubdtype(bb.dtype, np.floating):
            nan_count += int(np.isnan(bb).sum())
            inf_count += int(np.isinf(bb).sum())
            mask = np.isfinite(bb)
            vals = bb[mask]
        else:
            vals = bb.reshape(-1)
        if vals.size:
            vmin = float(vals.min()); vmax = float(vals.max())
            mn = vmin if mn is None else min(mn, vmin)
            mx = vmax if mx is None else max(mx, vmax)
            vals64 = vals.astype(np.float64, copy=False)
            total += float(vals64.sum())
            total2 += float(np.square(vals64).sum())
            finite_count += int(vals64.size)
    mean = total / finite_count if finite_count else None
    var = total2 / finite_count - mean * mean if finite_count and mean is not None else None
    return {
        "shape": list(arr.shape), "dtype": str(arr.dtype),
        "nan": nan_count, "inf": inf_count,
        "finite_count": finite_count, "min": mn, "max": mx,
        "mean": mean, "std": float(max(var,0.0) ** 0.5) if var is not None else None,
    }


def longest_true_run(mask):
    best = cur = 0
    for x in np.asarray(mask, dtype=bool):
        if x:
            cur += 1; best = max(best, cur)
        else:
            cur = 0
    return int(best)


def merge_item(item, out_path):
    h = hashlib.sha256(); size = 0
    with out_path.open("wb") as w:
        for part in item["parts"]:
            p = ROOT / part["path"]
            ph = hashlib.sha256()
            ps = 0
            with p.open("rb") as r:
                while True:
                    b = r.read(16 * 1024 * 1024)
                    if not b: break
                    w.write(b); h.update(b); ph.update(b)
                    size += len(b); ps += len(b)
            assert ps == part["size"], (p, ps, part["size"])
            assert ph.hexdigest() == part["sha256"], (p, "part sha mismatch")
    assert size == item["original_size"]
    assert h.hexdigest() == item["original_sha256"]
    return {"size": size, "sha256": h.hexdigest()}


def first_dim_len(v):
    if isinstance(v, np.ndarray) and v.ndim: return int(v.shape[0])
    if isinstance(v, (list, tuple)): return len(v)
    return None


def str_list(v):
    if isinstance(v, np.ndarray): return [str(x) for x in v.reshape(-1)]
    if isinstance(v, (list, tuple)): return [str(x) for x in v]
    return None


def small_value_summary(v):
    if isinstance(v, np.ndarray):
        if v.size <= 100000:
            flat = v.reshape(-1)
            if np.issubdtype(v.dtype, np.number):
                vals, counts = np.unique(flat, return_counts=True)
                if len(vals) <= 50:
                    return {"values": {str(jsonable(a)): int(b) for a,b in zip(vals,counts)}}
            elif flat.dtype.kind in "OUS":
                return {"values": dict(Counter(map(str,flat)))}
    if isinstance(v, (list,tuple)) and len(v) <= 100000:
        try: return {"values": dict(Counter(map(str,v)))}
        except: pass
    return {}


def summarize_split(d, version):
    out = {"keys": list(d.keys())}
    # n and field consistency
    lens = {}
    for k,v in d.items():
        n = first_dim_len(v)
        if n is not None: lens[k] = n
    out["first_dim_lengths"] = lens
    positive_lens = [x for x in lens.values() if x is not None]
    n = Counter(positive_lens).most_common(1)[0][0] if positive_lens else None
    out["n"] = n
    out["inconsistent_first_dim_fields"] = {k:v for k,v in lens.items() if n is not None and v != n}

    # identity/text
    if "id" in d:
        ids = str_list(d["id"])
        out["id_unique"] = len(set(ids)) if ids else None
        out["id_duplicates"] = len(ids) - len(set(ids)) if ids else None
        out["_ids"] = ids
    if "raw_text" in d:
        tx = str_list(d["raw_text"])
        out["raw_text_missing"] = sum((x == "" or x.lower()=="nan") for x in tx)
        words = [len(x.split()) for x in tx]
        out["raw_text_words"] = {
            "min": min(words), "max": max(words), "mean": float(np.mean(words)),
            "median": float(np.median(words))
        }
        out["_raw_text"] = tx

    # feature arrays
    for k in ["text","text_bert","audio","vision"]:
        if k in d and isinstance(d[k], np.ndarray):
            out[k] = array_stats(d[k])
            out[k]["sha256"] = sha256_array(d[k])

    # BERT validity / aligned zero-pattern
    if "text_bert" in d and isinstance(d["text_bert"], np.ndarray) and d["text_bert"].ndim == 3 and d["text_bert"].shape[1] >= 2:
        tb = d["text_bert"]
        mask = tb[:,1,:]
        lens2 = np.rint(mask.sum(axis=1)).astype(int)
        out["bert_attention_length"] = {
            "min": int(lens2.min()), "max": int(lens2.max()),
            "mean": float(lens2.mean()), "median": float(np.median(lens2)),
            "mask_non_binary": int(np.sum(~np.isin(mask,[0,1]))),
        }
        token_ids = tb[:,0,:]
        internal_token_zero = 0
        for i,L in enumerate(lens2):
            if L >= 2:
                internal_token_zero += int(np.sum(token_ids[i,1:L-1] == 0))
        out["bert_internal_token_zero_count"] = internal_token_zero

        if version == "aligned":
            for mod in ["audio","vision"]:
                if mod not in d: continue
                x = d[mod]
                zero_rates=[]; longest=[]; padding_nonzero=0
                for i,L in enumerate(lens2):
                    if L >= 2:
                        core = x[i,1:L-1]
                        z = np.all(core == 0, axis=1)
                        zero_rates.append(float(z.mean()) if len(z) else 0.0)
                        longest.append(longest_true_run(z))
                    # positions after SEP should be padding; check any nonzero
                    if L < x.shape[1]:
                        padding_nonzero += int(np.sum(np.any(x[i,L:] != 0, axis=1)))
                out[f"{mod}_aligned_zero_core"] = {
                    "mean_rate": float(np.mean(zero_rates)), "median_rate": float(np.median(zero_rates)),
                    "max_rate": float(np.max(zero_rates)), "mean_longest_run": float(np.mean(longest)),
                    "max_longest_run": int(np.max(longest)), "nonzero_rows_after_bert_length": padding_nonzero,
                }

    # unaligned lengths / zero rows
    if version == "unaligned":
        for mod in ["audio","vision"]:
            lk = f"{mod}_lengths"
            if mod in d and lk in d:
                x = d[mod]
                Ls = np.asarray(d[lk]).reshape(-1).astype(int)
                out[lk] = {
                    "shape": list(np.asarray(d[lk]).shape), "dtype": str(np.asarray(d[lk]).dtype),
                    "min": int(Ls.min()), "max": int(Ls.max()), "mean": float(Ls.mean()), "median": float(np.median(Ls))
                }
                mismatch_span=[]; mismatch_count=[]; zero_rates=[]; longest=[]; out_of_range=0
                for i,L in enumerate(Ls):
                    if L < 0 or L > x.shape[1]:
                        out_of_range += 1; continue
                    zfull = np.all(x[i] == 0, axis=1)
                    nz = np.flatnonzero(~zfull)
                    span = int(nz[-1]+1) if len(nz) else 0
                    cnt = int(len(nz))
                    if L != span: mismatch_span.append((i,int(L),span))
                    if L != cnt: mismatch_count.append((i,int(L),cnt))
                    z = zfull[:L]
                    zero_rates.append(float(z.mean()) if L else 0.0)
                    longest.append(longest_true_run(z))
                out[f"{mod}_length_consistency"] = {
                    "out_of_range": out_of_range,
                    "length_vs_nonzero_span_mismatches": len(mismatch_span),
                    "length_vs_nonzero_count_mismatches": len(mismatch_count),
                    "first_span_mismatches": mismatch_span[:10],
                    "first_count_mismatches": mismatch_count[:10],
                    "mean_zero_rate_within_length": float(np.mean(zero_rates)),
                    "max_zero_rate_within_length": float(np.max(zero_rates)),
                    "mean_longest_zero_run": float(np.mean(longest)),
                    "max_longest_zero_run": int(np.max(longest)),
                }

    # labels/annotations
    for k in ["annotations","classification_labels","regression_labels"]:
        if k in d:
            v = d[k]
            if isinstance(v,np.ndarray):
                out[k] = array_stats(v)
            else:
                out[k] = {"type": type(v).__name__, "len": first_dim_len(v)}
            out[k].update(small_value_summary(v))
            try:
                out[f"_{k}"] = jsonable(np.asarray(v).tolist())
            except Exception:
                pass
    return out


def audit_pickle(path, version):
    with path.open("rb") as f:
        data = pickle.load(f)
    assert isinstance(data,dict)
    report = {"top_keys": list(data.keys()), "splits": {}}
    for split, d in data.items():
        if isinstance(d,dict):
            report["splits"][str(split)] = summarize_split(d, version)
        else:
            report["splits"][str(split)] = {"type": type(d).__name__}
    del data
    gc.collect()
    return report


def strip_private_fields(report):
    out=json.loads(json.dumps(jsonable(report)))
    for sp in out.get("splits",{}).values():
        for k in list(sp):
            if k.startswith("_"): sp.pop(k,None)
    return out


manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
items = {Path(x["original_path"]).name:x for x in manifest["items"]}
summary = {"manifest": {
    "format": manifest.get("format"), "chunk_size": manifest.get("chunk_size"),
    "items": {k: {"original_size":v["original_size"], "original_sha256":v["original_sha256"], "parts":len(v["parts"])} for k,v in items.items()}
}}

# label.xlsx
label_path = ROOT / "E题数据/附件2-数据集特征文件/label.xlsx"
ldf = pd.read_excel(label_path)
label_report = {
    "rows": int(len(ldf)), "columns": [str(x) for x in ldf.columns],
    "dtypes": {str(k):str(v) for k,v in ldf.dtypes.items()},
    "nulls": {str(k):int(v) for k,v in ldf.isna().sum().items()},
    "duplicates_all": int(ldf.duplicated().sum()),
}
for col in ldf.columns:
    s=ldf[col]
    if s.nunique(dropna=False) <= 30:
        label_report.setdefault("value_counts",{})[str(col)] = {str(k):int(v) for k,v in s.value_counts(dropna=False).items()}
for c in ["label","regression_labels","sentiment"]:
    if c in ldf.columns and pd.api.types.is_numeric_dtype(ldf[c]):
        label_report[c+"_stats"] = {k:float(v) for k,v in ldf[c].describe().items()}
summary["label_xlsx"] = label_report
ldf.to_csv(OUT/"attachment2_label.csv", index=False)

private_cross = {}
for fname, version in [("aligned_50.pkl","aligned"),("unaligned_50.pkl","unaligned")]:
    item=items[fname]
    tmp=Path("/tmp")/fname
    merge_meta=merge_item(item,tmp)
    rep=audit_pickle(tmp,version)
    summary[fname]={"merge":merge_meta, **strip_private_fields(rep)}
    # save private cross-version data
    private_cross[version]={}
    rawrep=audit_pickle(tmp,version)
    for spname,sp in rawrep["splits"].items():
        private_cross[version][spname] = {
            "ids": sp.get("_ids"), "raw_text": sp.get("_raw_text"),
            "annotations": sp.get("_annotations"),
            "classification_labels": sp.get("_classification_labels"),
            "regression_labels": sp.get("_regression_labels"),
            "text_sha": sp.get("text",{}).get("sha256"),
            "bert_sha": sp.get("text_bert",{}).get("sha256"),
        }
    tmp.unlink()
    gc.collect()

# Cross-version parity
cross={}
A=private_cross.get("aligned",{}); U=private_cross.get("unaligned",{})
for split in sorted(set(A)|set(U)):
    a=A.get(split,{}); u=U.get(split,{})
    rec={}
    for k in ["ids","raw_text","annotations","classification_labels","regression_labels","text_sha","bert_sha"]:
        if a.get(k) is not None and u.get(k) is not None:
            rec[k+"_equal"] = a[k] == u[k]
    cross[split]=rec
summary["aligned_vs_unaligned"] = cross

# cross-split id overlap
for fname in ["aligned_50.pkl","unaligned_50.pkl"]:
    splits = summary[fname].get("splits",{})
    # public report removed ids, so use private copy by version
version_map={"aligned_50.pkl":"aligned","unaligned_50.pkl":"unaligned"}
for fname,ver in version_map.items():
    ids={k:set(v.get("ids") or []) for k,v in private_cross[ver].items()}
    ov={}
    names=sorted(ids)
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            ov[names[i]+"__"+names[j]] = len(ids[names[i]] & ids[names[j]])
    summary[fname]["split_id_overlaps"]=ov

(OUT/"attachment2_audit.json").write_text(json.dumps(jsonable(summary),ensure_ascii=False,indent=2),encoding="utf-8")

# compact human-readable text
lines=[]
lines.append("E题附件2数据审计")
lines.append("="*60)
lines.append(f"label.xlsx rows={label_report['rows']} columns={label_report['columns']}")
for fname in ["aligned_50.pkl","unaligned_50.pkl"]:
    r=summary[fname]
    lines.append(f"\n{fname}: size={r['merge']['size']} sha256={r['merge']['sha256']}")
    lines.append(f"top_keys={r['top_keys']} split overlaps={r['split_id_overlaps']}")
    for sp,info in r["splits"].items():
        lines.append(f"  {sp}: n={info.get('n')} keys={info.get('keys')}")
        for k in ["text","text_bert","audio","vision"]:
            if k in info: lines.append(f"    {k}: shape={info[k]['shape']} dtype={info[k]['dtype']} nan={info[k].get('nan')} inf={info[k].get('inf')}")
        if "bert_attention_length" in info: lines.append(f"    bert_len={info['bert_attention_length']}")
        for mod in ["audio","vision"]:
            key=f"{mod}_aligned_zero_core"
            if key in info: lines.append(f"    {key}={info[key]}")
            key=f"{mod}_length_consistency"
            if key in info: lines.append(f"    {key}={info[key]}")
        for k in ["annotations","classification_labels","regression_labels"]:
            if k in info: lines.append(f"    {k}: {info[k]}")
lines.append("\nAligned vs unaligned parity:")
lines.append(json.dumps(summary["aligned_vs_unaligned"],ensure_ascii=False,indent=2))
(OUT/"attachment2_audit.txt").write_text("\n".join(lines),encoding="utf-8")
print("\n".join(lines[:80]))
