#!/usr/bin/env python3
"""Fetch a pinned prior-release gzip over HTTPS; verify decompressed canonical SHA."""
from pathlib import Path
import argparse,gzip,hashlib,sys,urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gpf_attribution.data import DATA_SHA256
URL='https://raw.githubusercontent.com/kishordgupta/gpf-semantic-response-audit/5a9d5ffa18419a637c9b1c9763f40d0e165a74d6/data/responses.csv.gz'

def download_data(output):
    output=Path(output)
    if output.exists():
        if hashlib.sha256(output.read_bytes()).hexdigest()==DATA_SHA256:
            print('Canonical dataset already present; checksum passed.');return output
        raise FileExistsError('Output exists with another checksum; refusing to replace it.')
    with urllib.request.urlopen(URL,timeout=120) as response:packed=response.read()
    data=gzip.decompress(packed)
    if hashlib.sha256(data).hexdigest()!=DATA_SHA256:raise ValueError('Downloaded data failed canonical SHA-256 check')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as handle:handle.write(data)
    print(f'Saved canonical CSV ({len(data):,} bytes); SHA-256 {DATA_SHA256}.')
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='data/responses.csv')
    download_data(parser.parse_args().output)
