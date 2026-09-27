"""Strict data contract and fixed developer mapping."""
from pathlib import Path
import hashlib
import pandas as pd

DATA_SHA256='c459043e9fb30eb653541d3a0ebaddb10735512025ed57190c0cb12966dd89da'
COLUMNS=['run_id','cell_id','experiment_stage','repeat','model_name','model_id','model_provider','prompt_family','prompt_question','full_prompt','x_group_id','x_group_name','news_source','news_source_link','news_author','news_published_utc','news_modified_utc','news_retrieved_utc','news_word_count','y_group_dimension','y_group_id','y_group_name','response','response_word_count','captured_utc','requested_word_range','length_status','thread_deleted']
MODEL_TO_COMPANY={'GPT 5.4':'OpenAI','GPT 5.4 Mini':'OpenAI','Claude Sonnet 4.5':'Anthropic','Claude Haiku 4.5':'Anthropic','Meta Llama 3.3 Turbo':'Meta','Meta Llama Maverick 4':'Meta','Gemini 2.5 Flash':'Google','Gemini 2.5 Pro':'Google','Mistral Large 3':'Mistral','Mistral Medium 3':'Mistral','Grok Fast':'xAI','3.5 Flash-Lite':'Google'}
EXCLUDED_CELLS=['bbc_brewdog__gender_sexuality__07__significance','bbc_brewdog__geography__05__community','bbc_brewdog__political__00__bias_check','reuters_denmark__gender_sexuality__00__emotion','reuters_denmark__religion__06__worldview']

def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def load_dataset(path):
    path=Path(path)
    if file_sha256(path)!=DATA_SHA256: raise ValueError('Canonical CSV SHA-256 mismatch')
    df=pd.read_csv(path,dtype=str,keep_default_na=False)
    if list(df.columns)!=COLUMNS or df.shape!=(7056,28):raise ValueError('Expected exact 7056 x 28 schema')
    if set(df.model_name)!=set(MODEL_TO_COMPANY):raise ValueError('Unexpected model labels')
    if df.duplicated(['model_name','cell_id']).any():raise ValueError('Duplicate model/cell keys')
    if not df.groupby('cell_id').model_name.nunique().eq(12).all():raise ValueError('Incomplete model coverage')
    if not df.groupby('cell_id').full_prompt.nunique().eq(1).all():raise ValueError('Cell prompt mismatch')
    if df.response.str.strip().eq('').any():raise ValueError('Empty response')
    df=df.copy();df['row_index']=range(len(df));df['company']=df.model_name.map(MODEL_TO_COMPANY)
    return df
