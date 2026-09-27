"""Fixed before-test baseline specifications. No hyperparameter search."""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from .text import StyleFeatures

REPRESENTATIONS=('word_svm','char_svm','style_logit','dummy')

def make_transformer(name):
    if name=='word_svm':return TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=40000,sublinear_tf=True,strip_accents='unicode',dtype=np.float64)
    if name=='char_svm':return TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=3,max_features=60000,sublinear_tf=True,strip_accents='unicode',dtype=np.float64)
    if name=='style_logit':return Pipeline([('style',StyleFeatures()),('scale',StandardScaler())])
    if name=='dummy':return StyleFeatures()
    raise ValueError(f'Unknown representation: {name}')

def make_estimator(name,seed=20260927):
    if name in ('word_svm','char_svm'):return LinearSVC(C=1.0,dual='auto',tol=1e-4,max_iter=5000,random_state=seed)
    if name=='style_logit':return LogisticRegression(C=1.0,solver='lbfgs',max_iter=3000,random_state=seed)
    if name=='dummy':return DummyClassifier(strategy='most_frequent',random_state=seed)
    raise ValueError(f'Unknown representation: {name}')

def make_classifier(name,seed=20260927):
    return Pipeline([('features',make_transformer(name)),('classifier',make_estimator(name,seed))])
