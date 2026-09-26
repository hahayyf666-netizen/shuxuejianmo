"""Transcript-only candidate identity inventory for all 20 Attachment4 samples."""
import collections
import hashlib
import json
import pickle
import re
from pathlib import Path

import h5py

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
WORDS=HERE.parents[2]/'q3_native_t0_assets/CMU_MOSEI_TimestampedWords.csd'
BASE=ROOT/'E题数据/附件4-可解释专项视频样本与特征文件/未对齐版本'
def norm(text): return re.findall(r'[a-z0-9]+',str(text).casefold())
targets={}
for i in range(1,21):
    sid=f'{i:02d}'
    with (BASE/f'{sid}.pkl').open('rb') as h: row=pickle.load(h)
    targets[sid]={'raw_text':str(row['raw_text']),'tokens':norm(row['raw_text'])}
all_words={}
index=collections.defaultdict(list)
with h5py.File(WORDS,'r') as h:
    data=h['words/data']
    for video_id in data:
        raw=data[video_id]['features'][:].reshape(-1)
        tokens=[t for x in raw for t in norm(x.decode('utf-8',errors='replace')) if t!='sp']
        all_words[video_id]=tokens
        for j in range(len(tokens)-2):
            index[tuple(tokens[j:j+3])].append((video_id,j))
records=[]
for sid,target in targets.items():
    tokens=target['tokens']
    counter=collections.Counter()
    for j in range(len(tokens)-2):
        for vid,pos in index.get(tuple(tokens[j:j+3]),[]):
            counter[(vid,pos-j)]+=1
    top=[]
    for (vid,offset),hits in counter.most_common(3):
        source=all_words[vid]
        exact=sum(0<=offset+j<len(source) and tokens[j]==source[offset+j] for j in range(len(tokens)))
        top.append({'video_id':vid,'offset':offset,'trigram_hits':hits,
                    'ordered_exact_tokens':exact,'target_tokens':len(tokens)})
    records.append({'sample_id':sid,'raw_text':target['raw_text'],'top_candidates':top,
                    'identity_status':'transcript_candidate_only_not_feature_verified'})
out={'scope':'Attachment4 20 unaligned raw_text','timestamped_words_sha256':hashlib.sha256(WORDS.read_bytes()).hexdigest(),
     'samples':records}
(HERE/'results/source_candidates_20.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps([{'id':r['sample_id'],'top':r['top_candidates'][:1]} for r in records],ensure_ascii=False))
