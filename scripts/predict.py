#!/usr/bin/env python3
"""Infer a closed-set label using a trusted locally trained artifact."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gpf_attribution.inference import predict_text
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--text');g.add_argument('--text-file',type=Path)
    p.add_argument('--top-k',type=int,default=3);a=p.parse_args()
    text=a.text if a.text is not None else a.text_file.read_text(encoding='utf-8')
    print(json.dumps(predict_text(a.artifact,text,a.top_k),indent=2))
