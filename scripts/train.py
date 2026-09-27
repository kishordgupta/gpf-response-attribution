#!/usr/bin/env python3
"""Run the predeclared finite response-only attribution experiment."""
from pathlib import Path
import argparse,sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gpf_attribution.training import train,export_demo_models,export_top_features
from gpf_attribution.data import load_dataset
from gpf_attribution.text import clean_response
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('data/responses.csv'))
    p.add_argument('--output',type=Path,default=Path('results'))
    p.add_argument('--fit-demo-only',action='store_true',help='Fit two full-data demonstration artifacts without evaluating the benchmark')
    a=p.parse_args()
    if a.fit_demo_only:
        df=load_dataset(a.data);df['clean_text']=df.response.map(clean_response)
        config=json.loads((Path(__file__).resolve().parents[1]/'config/experiment.json').read_text())
        artifacts=export_demo_models(df,a.output.parent/'artifacts',config)
        export_top_features(a.output.parent/'artifacts',a.output)
        print(json.dumps({'mode':'demonstration-only; no held-out evaluation performed','artifacts':artifacts},indent=2))
    else:
        train(a.data,a.output)
