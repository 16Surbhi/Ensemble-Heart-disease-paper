#!/usr/bin/env python3
"""Audit exact UCI input-file bytes (SHA-256) and Statlog/Cleveland row overlap.

Usage (Colab / local, Python 3):
    python verify_raw_inputs_and_overlap.py --download
or after placing the raw UCI files in ./data:
    python verify_raw_inputs_and_overlap.py --data-dir data

Creates raw_input_integrity_report.json and statlog_cleveland_overlap_report.csv.
A discrepancy in either hash is explicitly reported; no result is invented.
"""
import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import urllib.request
import csv
import sys

FILES = {
    'Cleveland': ('processed.cleveland.data', 'a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8', 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'),
    'Statlog': ('heart.dat', 'f5f3b4204c285bafadd85cb735f38b47689f2be7047feb172dcbeab648110bf9', 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat'),
    'Hungarian': ('processed.hungarian.data', 'd1ad108f785768cd3d7e82dc522e6f5a61eea93cccfb3a46ee8076f73fc3d796', 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'),
    'Swiss': ('processed.switzerland.data', '834a405ccf5b66ab4056bb77794adc8df0b7125186454c0a1d002d33c6c3b314', 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data'),
}

def read_rows(file_path, delim, label_style):
    rows=[]
    for line_no, line in enumerate(file_path.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip():continue
        toks=line.split(',') if delim == ',' else line.split()
        if len(toks)!=14:
            raise ValueError(f'{file_path}:{line_no} has {len(toks)} rather than 14 fields')
        feature = []
        for v in toks[:13]:
            if v.strip() in ['?','']:
                feature.append(None)
            else:
                feature.append(Decimal(v.strip()).normalize())
        target = int(Decimal(toks[13]))
        target = (1 if target > 0 else 0) if label_style == 'uci' else ({1:0,2:1}[target])
        rows.append((tuple(feature), target, line_no))
    return rows

def compare_rows(cleveland,statlog):
    # Exact numerical matches, with label checks and duplicate-aware one-to-one counts.
    c_features=Counter([r[0] for r in cleveland])
    c_with_target=Counter([(r[0],r[1]) for r in cleveland])
    match_features = sum((c_features & Counter(r[0] for r in statlog)).values())
    match_labels = sum((c_with_target & Counter((r[0],r[1]) for r in statlog)).values())
    detail=[]
    for f,y,n in statlog:
        same_feature=[(c_y,c_no) for c_f,c_y,c_no in cleveland if c_f==f]
        detail.append({'statlog_line':n,'matching_cleveland_lines':';'.join(str(j) for _,j in same_feature),
                       'feature_match':bool(same_feature),
                       'binary_label_agreement':(any(c_y == y for c_y,_ in same_feature) if same_feature else '')})
    with open('statlog_cleveland_overlap_report.csv','w',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=list(detail[0])); writer.writeheader();writer.writerows(detail)
    return {'cleveland_rows':len(cleveland),'statlog_rows':len(statlog),
            'statlog_rows_matching_cleveland_exact_13_features_one_to_one':match_features,
            'statlog_rows_matching_cleveland_features_and_binary_label_one_to_one':match_labels,
            'all_270_statlog_rows_match_with_labels':match_labels==len(statlog),
            'comparison':'13 numerical attributes, numeric normalization, label mapping Statlog 1->0 / 2->1 and Cleveland 0->0 / 1-4->1',
            'note':'This is exact feature matching; duplicate tuples are counted one-to-one. Near matches or partially missing values are not automatically treated as exact.'}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',type=Path,default=Path('data')); ap.add_argument('--download',action='store_true');args=ap.parse_args()
    args.data_dir.mkdir(parents=True,exist_ok=True)
    checks={}; absent=[]
    for ds,(filename,expected,url) in FILES.items():
        path=args.data_dir/filename
        if args.download and not path.exists():
            try:
                urllib.request.urlretrieve(url,path)
                print('Downloaded',url)
            except Exception as exc:
                print('Download failed:',ds,exc,file=sys.stderr)
        if not path.exists():
            absent.append(str(path));checks[ds]={'file':str(path),'found':False,'sha256_manifest':expected,'verification':'NOT CHECKED'}
            continue
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        checks[ds]={'file':str(path),'found':True,'bytes':path.stat().st_size,'sha256_manifest':expected,'sha256_recomputed':actual,'hash_matches_manifest':actual==expected}
    report={'input_checks':checks,'missing_files':absent}
    a=args.data_dir/'processed.cleveland.data';b=args.data_dir/'heart.dat'
    if a.exists() and b.exists():
        try: report['overlap']=compare_rows(read_rows(a,',','uci'),read_rows(b,'space','statlog'))
        except Exception as exc:report['overlap_error']=str(exc)
    else:report['overlap']='NOT CHECKED: both source files required'
    Path('raw_input_integrity_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if absent or any(not x.get('hash_matches_manifest',False) for x in checks.values()):
        sys.exit('ATTENTION: Missing source file(s) or unexpected hash(es). Do NOT claim hashes were reverified.')

if __name__ == "__main__":
    main()

