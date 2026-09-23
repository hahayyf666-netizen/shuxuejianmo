from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(".")
OUT = ROOT / "_e_data_audit_out"
OUT.mkdir(exist_ok=True)


def find_attachment1_label() -> Path:
    roots = sorted(p for p in (ROOT/"E题数据").glob("附件1*") if p.is_dir())
    if len(roots)!=1:
        raise RuntimeError(f"Expected one attachment1 directory, got {roots}")
    xs=sorted(roots[0].rglob("label-100.xlsx"))
    if len(xs)!=1:
        raise RuntimeError(f"Expected one label-100.xlsx, got {xs}")
    return xs[0]


def norm_clip(x):
    try:return str(int(float(x)))
    except Exception:return str(x).strip()


def key(vid,cid):
    return str(vid).strip()+"$_$"+norm_clip(cid)


def main():
    a1_path=find_attachment1_label()
    a2_path=OUT/"attachment2_label.csv"
    if not a2_path.exists():
        raise FileNotFoundError("Run attachment2 audit first: attachment2_label.csv missing")

    a1=pd.read_excel(a1_path)
    a2=pd.read_csv(a2_path)

    a1_rows={}
    for _,r in a1.iterrows():
        k=key(r["video_id"],r["clip_id"])
        a1_rows[k]={
            "video_id":str(r["video_id"]).strip(),
            "clip_id":norm_clip(r["clip_id"]),
            "label":float(r["label"]),
            "annotation":str(r["annotation"]),
            "text":str(r["text"]),
        }
    a2_rows={}
    for _,r in a2.iterrows():
        k=key(r["video_id"],r["clip_id"])
        a2_rows[k]={
            "video_id":str(r["video_id"]).strip(),
            "clip_id":norm_clip(r["clip_id"]),
            "label":float(r["label"]),
            "annotation":str(r["annotation"]),
            "mode":str(r["mode"]),
            "text":str(r["text"]),
        }

    overlap=sorted(set(a1_rows)&set(a2_rows))
    details=[]
    for k in overlap:
        x,y=a1_rows[k],a2_rows[k]
        details.append({
            "key":k,
            "mode":y["mode"],
            "label_equal":abs(x["label"]-y["label"])<1e-8,
            "annotation_equal":x["annotation"]==y["annotation"],
            "text_equal":x["text"]==y["text"],
            "attachment1_label":x["label"],
            "attachment2_label":y["label"],
        })

    mode_counts=dict(Counter(x["mode"] for x in details))
    report={
        "attachment1_rows":len(a1_rows),
        "attachment2_rows":len(a2_rows),
        "overlap_count":len(overlap),
        "overlap_by_split":mode_counts,
        "label_mismatch_count":sum(not x["label_equal"] for x in details),
        "annotation_mismatch_count":sum(not x["annotation_equal"] for x in details),
        "text_mismatch_count":sum(not x["text_equal"] for x in details),
        "overlap":details,
        "leakage_firewall":{
            "forbid_q1_to_q2q3":[
                "Q1 labels",
                "Q1 label-based feature/model/hyperparameter/threshold selection",
                "any parameter/statistic/model state fitted, estimated, calibrated, adapted, or selected using Q1 samples",
                "Q1-fitted normalization mean/std",
                "Q1-fitted PCA or dimensionality reduction",
                "Q1-fitted clustering/codebook",
                "self-supervised adaptation using Q1 videos",
                "feature extractor fine-tuning using Q1 samples",
                "any data-driven preprocessing parameter estimated from Q1 samples"
            ],
            "allow_reuse_only_if_q1_data_independent":[
                "stateless data readers",
                "stateless video decoding code",
                "fixed pretrained tools whose parameters are not adapted on Q1",
                "preprocessing functions with parameters fixed independently of Q1 data"
            ],
            "reason":"Attachment1 overlaps Attachment2 train/test; Q1-derived fitted state could leak Attachment2 test information even without using Q1 labels"
        }
    }
    expected={"overlap_count":18,"train":11,"valid":0,"test":7}
    checks={
        "overlap_count":report["overlap_count"]==expected["overlap_count"],
        "train":mode_counts.get("train",0)==expected["train"],
        "valid":mode_counts.get("valid",0)==expected["valid"],
        "test":mode_counts.get("test",0)==expected["test"],
        "labels_equal":report["label_mismatch_count"]==0,
        "annotations_equal":report["annotation_mismatch_count"]==0,
        "texts_equal":report["text_mismatch_count"]==0,
    }
    report["baseline_checks"]=checks
    report["status"]="PASS" if all(checks.values()) else "FAIL"

    (OUT/"cross_attachment_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=[
        "E题跨附件审计","="*60,
        f"status={report['status']}",
        f"attachment1_rows={len(a1_rows)} attachment2_rows={len(a2_rows)}",
        f"overlap={len(overlap)} split_counts={mode_counts}",
        f"label_mismatch={report['label_mismatch_count']} annotation_mismatch={report['annotation_mismatch_count']} text_mismatch={report['text_mismatch_count']}",
        f"baseline_checks={checks}",
    ]
    (OUT/"cross_attachment_audit.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
