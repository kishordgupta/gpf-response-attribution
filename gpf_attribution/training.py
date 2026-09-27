"""Finite, fixed response-only attribution benchmark with grouped held-out tests."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import itertools
import json
import platform
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,precision_recall_fscore_support,confusion_matrix
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from .data import load_dataset,DATA_SHA256,MODEL_TO_COMPANY,EXCLUDED_CELLS
from .text import clean_response,normalized_response_hash,mask_model_names,CLEANING_VERSION,NAME_TERMS
from .estimators import REPRESENTATIONS,make_transformer,make_estimator

SEED=20260927
TARGET_COLUMNS={'model':'model_name','company':'company'}
PLANS=[('full','identity',REPRESENTATIONS),('full','article',('char_svm',)),
       ('full','family',('char_svm',)),('screened','identity',('char_svm',)),
       ('screened','article',('char_svm',))]
IDENTIFIERS=['experiment_id','population','representation','target','split_strategy']

def dump_json(obj,path):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def build_splits(df,population,strategy,identity_assignments):
    frame=df if population=='full' else df.loc[~df.cell_id.isin(EXCLUDED_CELLS)]
    if strategy=='identity':
        heldouts=[(f'fold_{i+1}',set(groups)) for i,groups in enumerate(identity_assignments)]
        group_column='y_group_id'
    else:
        group_column={'article':'x_group_id','family':'prompt_family'}[strategy]
        heldouts=[(group,{group}) for group in sorted(frame[group_column].unique())]
    splits=[]
    for fold,held in heldouts:
        test=frame.index[frame[group_column].isin(held)].to_numpy()
        before=frame.index[~frame[group_column].isin(held)].to_numpy()
        test_hashes=set(df.loc[test,'normalized_hash'])
        purge=np.array([i for i in before if df.at[i,'normalized_hash'] in test_hashes],dtype=int)
        train=np.array([i for i in before if df.at[i,'normalized_hash'] not in test_hashes],dtype=int)
        row_overlap=set(train)&set(test)
        cell_overlap=set(df.loc[train,'cell_id'])&set(df.loc[test,'cell_id'])
        group_overlap=set(df.loc[train,group_column])&set(df.loc[test,group_column])
        hash_overlap=set(df.loc[train,'normalized_hash'])&test_hashes
        if row_overlap or cell_overlap or group_overlap or hash_overlap:raise AssertionError('Train/test leakage in split')
        for target,col in TARGET_COLUMNS.items():
            if set(df.loc[train,col])!=set(df[col]):raise AssertionError('Training split missing a target class')
        audit={'population':population,'split_strategy':strategy,'fold':fold,'group_column':group_column,
            'held_out_groups':sorted(held),'n_train_before_purge':len(before),'n_train':len(train),'n_test':len(test),
            'purged_train_rows':len(purge),'purged_row_indices':[int(i) for i in purge],
            'train_test_row_overlap':len(row_overlap),'train_test_cell_overlap':len(cell_overlap),
            'train_test_group_overlap':len(group_overlap),'normalized_response_overlap_after_purge':len(hash_overlap),
            'feature_fit_policy':'transformer fitted on these training rows only; test is transform-only'}
        splits.append((fold,train,test,purge,audit))
    return splits

def score_rows(y,pred):
    return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred))}

def fit_target(name,target,train_features,test_features,y_train,seed):
    clf=make_estimator(name,seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');clf.fit(train_features,y_train)
    predicted=clf.predict(test_features)
    scores=clf.decision_function(test_features) if hasattr(clf,'decision_function') else None
    return target,clf,predicted,scores,[str(w.message) for w in caught]

def summarize_predictions(predictions,fold_metrics,out):
    if not predictions:return [],[],[]
    pred=pd.DataFrame(predictions)
    metrics=list(fold_metrics);perclass=[];confusions=[]
    for key,group in pred.groupby(IDENTIFIERS,sort=False):
        common=dict(zip(IDENTIFIERS,key));y=group.true_label.to_numpy();p=group.predicted_label.to_numpy()
        labels=sorted(set(y));metrics.append({**common,'fold':'pooled','n_train':None,'n_test':len(group),**score_rows(y,p)})
        pr,re,f1,support=precision_recall_fscore_support(y,p,labels=labels,zero_division=0)
        for i,label in enumerate(labels):perclass.append({**common,'label':label,'precision':float(pr[i]),'recall':float(re[i]),'f1':float(f1[i]),'support':int(support[i])})
        cm=confusion_matrix(y,p,labels=labels)
        for i,a in enumerate(labels):
            for j,b in enumerate(labels):confusions.append({**common,'true_label':a,'predicted_label':b,'count':int(cm[i,j])})
    pred.to_csv(out/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(perclass).to_csv(out/'per_class.csv',index=False)
    pd.DataFrame(confusions).to_csv(out/'confusion_counts.csv',index=False)
    return metrics,perclass,confusions

def train(data_path,output_path):
    start=time.monotonic();out=Path(output_path);out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'config/experiment.json').read_text())
    df=load_dataset(data_path)
    df['clean_text']=df.response.map(clean_response)
    if df.clean_text.eq('').any():raise ValueError('Response emptied by cleaning')
    df['normalized_hash']=df.clean_text.map(normalized_response_hash)
    assignments=np.array_split(np.random.default_rng(SEED).permutation(sorted(df.y_group_id.unique())),5)
    identity_assignments=[list(map(str,a)) for a in assignments]
    mask_changes=int(sum(mask_model_names(t)!=t for t in df.clean_text))
    prefix_changes=int(sum(clean_response(r)!=reflow(r) for r in df.response))
    cleaning={'version':CLEANING_VERSION,'leading_wrapper_changed_rows':prefix_changes,
        'gemini_said_leading_rows':int(df.response.str.match(r'^\s*Gemini said\s*\n',case=False).sum()),
        'remaining_leading_gemini_said':int(df.clean_text.str.match(r'^Gemini said',case=False).sum()),
        'literal_model_vendor_mask_changed_rows':mask_changes,'mask_terms':list(NAME_TERMS),
        'mask_sensitivity_action':'No additional fits: literal model/vendor masking is exactly identical to cleaned inputs.' if mask_changes==0 else 'Additional masking needed; see notes.',
        'normalized_hash_definition':'SHA256(NFKC(clean_response(text)).casefold()); cleaning collapses whitespace',
        'normalized_duplicate_rows':int(df.normalized_hash.duplicated(keep=False).sum())}
    if mask_changes:raise ValueError('Model-name mask changes were found; the fixed no-op sensitivity assumption must be revised before running.')
    dump_json(cleaning,out/'cleaning_audit.json')
    dump_json({'seed':SEED,'identity_folds':identity_assignments},out/'identity_fold_assignments.json')
    audits=[];manifest=[];all_splits={}
    for population,strategy,_ in PLANS:
        splits=build_splits(df,population,strategy,identity_assignments);all_splits[(population,strategy)]=splits
        for fold,train_rows,test_rows,purged,audit in splits:
            audits.append(audit)
            for role,indices in [('train',train_rows),('test',test_rows),('purged',purged)]:
                for i in indices:
                    row=df.loc[i];manifest.append({'population':population,'split_strategy':strategy,'fold':fold,'row_index':int(i),'role':role,
                        'cell_id':row.cell_id,'y_group_id':row.y_group_id,'article_id':row.x_group_id,'prompt_family':row.prompt_family})
    dump_json(audits,out/'split_audits.json')
    pd.DataFrame(manifest).to_csv(out/'split_manifest.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    total_jobs=sum(len(all_splits[(pop,split)])*len(reps) for pop,split,reps in PLANS)
    completed=0;predictions=[];fold_metrics=[];timings=[];warning_records=[]
    with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=2) as pool:
        for population,strategy,reps in PLANS:
            for fold,train_rows,test_rows,_,audit in all_splits[(population,strategy)]:
                texts_train=df.loc[train_rows,'clean_text'].tolist();texts_test=df.loc[test_rows,'clean_text'].tolist()
                for rep in reps:
                    tick=time.monotonic();features=make_transformer(rep)
                    train_features=features.fit_transform(texts_train)
                    test_features=features.transform(texts_test)
                    futures=[pool.submit(fit_target,rep,target,train_features,test_features,
                        df.loc[train_rows,col].to_numpy(),SEED) for target,col in TARGET_COLUMNS.items()]
                    for future in futures:
                        target,clf,pred,scores,warn=future.result();label_column=TARGET_COLUMNS[target]
                        exp=f'{population}__{strategy}__{rep}__{target}'
                        common={'experiment_id':exp,'population':population,'representation':rep,'target':target,'split_strategy':strategy}
                        truth=df.loc[test_rows,label_column].to_numpy()
                        fold_metrics.append({**common,'fold':fold,'n_train':len(train_rows),'n_test':len(test_rows),**score_rows(truth,pred)})
                        if warn:warning_records.append({**common,'fold':fold,'warnings':warn})
                        ranks=np.argsort(-scores,axis=1,kind='stable')[:,:3] if scores is not None else None
                        for k,i in enumerate(test_rows):
                            r=df.loc[i];record={**common,'fold':fold,'row_index':int(i),'cell_id':r.cell_id,'y_group_id':r.y_group_id,
                                'true_label':str(truth[k]),'predicted_label':str(pred[k]),'correct':bool(truth[k]==pred[k]),
                                'top1_margin':None,'top2_label':None,'top2_margin':None,'top3_label':None,'top3_margin':None}
                            if ranks is not None:
                                record['top1_margin']=float(scores[k,ranks[k,0]])
                                for rank in [1,2]:record[f'top{rank+1}_label']=str(clf.classes_[ranks[k,rank]]);record[f'top{rank+1}_margin']=float(scores[k,ranks[k,rank]])
                            predictions.append(record)
                    completed+=1
                    timings.append({'population':population,'split_strategy':strategy,'fold':fold,'representation':rep,'seconds':round(time.monotonic()-tick,3),'n_features':int(train_features.shape[1])})
                    status={'phase':'training','completed_feature_fits':completed,'planned_feature_fits':total_jobs,'current':{'population':population,'split_strategy':strategy,'fold':fold,'representation':rep},'elapsed_seconds':round(time.monotonic()-start,1)}
                    dump_json(status,out/'status.json');print(json.dumps(status),flush=True)
                    del train_features,test_features,features
            summarize_predictions(predictions,fold_metrics,out)
    metrics,_,_=summarize_predictions(predictions,fold_metrics,out)
    dump_json(timings,out/'timings.json');dump_json(warning_records,out/'fit_warnings.json')
    artifacts=export_demo_models(df,out.parent/'artifacts',config)
    export_top_features(out.parent/'artifacts',out)
    summary={'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'data_sha256':DATA_SHA256,'rows':len(df),'canonical_columns':28,
        'models':list(MODEL_TO_COMPANY),'companies':MODEL_TO_COMPANY,'n_models':12,'n_companies':6,'target_company_counts':df.company.value_counts().to_dict(),
        'config':config,'feature_source':'Only cleaned response text. Prompt, article text, metadata, identity labels, and timestamps are never input features.',
        'cleaning':cleaning,'identity_folds':identity_assignments,'screened_excluded_cells':EXCLUDED_CELLS,'screened_rows':int((~df.cell_id.isin(EXCLUDED_CELLS)).sum()),
        'prediction_rows':len(predictions),'split_count':len(audits),'feature_fit_jobs':completed,'target_fit_jobs':completed*2,
        'purged_training_rows_sum':int(sum(a['purged_train_rows'] for a in audits)),
        'pooled_metrics':[r for r in metrics if r['fold']=='pooled'],'warnings':warning_records,'artifacts':artifacts,
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,'joblib':joblib.__version__},
        'elapsed_seconds':round(time.monotonic()-start,2),
        'limitations':['Closed-set attribution of this captured corpus, not reliable provenance of arbitrary text.','Model, platform, capture process, system prompts, dates and length are confounded.','One response per model/prompt; no sampling-variance estimate.','Two article holdouts are specific transfer tests, not a population estimate of news generalization.','Identity and question-family holdouts share the two source articles.','Company classes are naturally unbalanced; macro-F1 and balanced accuracy supplement accuracy.','Scores are decision-function margins, not calibrated probabilities or out-of-distribution confidence.','Full-data demonstration artifacts have seen every row; their training predictions are not evaluation evidence.']}
    dump_json(summary,out/'summary.json');dump_json({'phase':'complete','completed_feature_fits':completed,'planned_feature_fits':total_jobs,'prediction_rows':len(predictions),'elapsed_seconds':summary['elapsed_seconds']},out/'status.json')
    print(json.dumps({'phase':'complete','prediction_rows':len(predictions),'elapsed_seconds':summary['elapsed_seconds'],'artifacts':artifacts}),flush=True)
    return summary

def reflow(text):
    import re
    return re.sub(r'\s+',' ',text).strip()

def export_demo_models(df,artifact_dir,config):
    artifact_dir.mkdir(parents=True,exist_ok=True);artifacts=[]
    with threadpool_limits(limits=1):
        features=make_transformer('char_svm');X=features.fit_transform(df.clean_text.tolist())
        for target,col in TARGET_COLUMNS.items():
            clf=make_estimator('char_svm',SEED);clf.fit(X,df[col].to_numpy())
            pipeline=Pipeline([('features',features),('classifier',clf)])
            obj={'estimator':pipeline,'target':target,'classes':list(map(str,clf.classes_)),'representation':'char_svm','cleaning_version':CLEANING_VERSION,
                 'data_sha256':DATA_SHA256,'masked_names':False,'training_population':'full','training_rows':len(df),'config':config,'score_type':'decision-function margins, not probabilities'}
            path=artifact_dir/f'{target}_char_full.joblib';joblib.dump(obj,path,compress=3)
            artifacts.append({'file':f'artifacts/{path.name}','target':target,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'training_rows':len(df),'evaluation_role':'Full-data demo only; evaluate using held-out result tables, not this fitted object.'})
    dump_json(artifacts,artifact_dir/'metadata.json');return artifacts


def export_top_features(artifact_dir,results_dir):
    """Descriptive full-data coefficient inspection, never a causal explanation."""
    rows=[]
    for path in sorted(Path(artifact_dir).glob('*_char_full.joblib')):
        artifact=joblib.load(path)
        pipeline=artifact['estimator']
        names=pipeline.named_steps['features'].get_feature_names_out()
        classifier=pipeline.named_steps['classifier']
        for class_index,label in enumerate(classifier.classes_):
            weights=classifier.coef_[class_index]
            for rank,index in enumerate(np.argsort(-weights,kind='stable')[:20],1):
                rows.append({'target':artifact['target'],'label':str(label),'rank':rank,
                    'feature':str(names[index]),'coefficient':float(weights[index]),
                    'fit_population':'all 7056 responses; demonstration artifact',
                    'interpretation':'Descriptive fitted character-ngram weight; not a causal explanation'})
    results_dir=Path(results_dir);results_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(results_dir/'demo_top_features.csv',index=False)
    return len(rows)
