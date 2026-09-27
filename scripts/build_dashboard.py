"""Build a self-contained, offline dashboard from saved attribution metrics.

No response text, model artifact, credentials, or network services are embedded.
"""
from pathlib import Path
import argparse,csv,json
ROOT=Path(__file__).resolve().parents[1]
def read_csv(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def build(results,output):
    names=['metrics.csv','per_class.csv','confusion_counts.csv']
    payload={n.removesuffix('.csv'):read_csv(results/n) for n in names}
    payload['summary']=json.loads((results/'summary.json').read_text())
    assert payload['metrics'] and any(x['fold']=='pooled' for x in payload['metrics'])
    template=(ROOT/'dashboard/template.html').read_text()
    data=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(template.replace('__RESULTS_JSON__',data),encoding='utf-8')
    print(json.dumps({'output':str(output),'metrics_rows':len(payload['metrics']),'per_class_rows':len(payload['per_class']),'confusion_rows':len(payload['confusion_counts']),'embedded_raw_response_text':False}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=ROOT/'results');p.add_argument('--output',type=Path,default=ROOT/'dashboard/index.html');a=p.parse_args();build(a.results,a.output)
