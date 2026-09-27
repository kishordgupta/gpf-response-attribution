"""Closed-set demonstration. Scores are not probabilities or OOD confidence."""
import joblib
import numpy as np
from .data import MODEL_TO_COMPANY
from .text import clean_response,mask_model_names,CLEANING_VERSION

def predict_text(artifact_path,text,top_k=3):
    if not isinstance(text,str) or not text.strip():raise ValueError('Supply a nonempty response text')
    artifact=joblib.load(artifact_path)  # Load only artifacts you trust; joblib uses pickle.
    if artifact.get('cleaning_version')!=CLEANING_VERSION:raise ValueError('Artifact cleaning version mismatch')
    cleaned=clean_response(text)
    if not cleaned:raise ValueError('No substantive response remains after header cleaning')
    if artifact.get('masked_names'):cleaned=mask_model_names(cleaned)
    estimator=artifact['estimator'];scores=np.asarray(estimator.decision_function([cleaned]))[0]
    classes=np.asarray(estimator.classes_);order=np.argsort(-scores,kind='stable')[:max(1,min(int(top_k),len(classes)))]
    return {'target':artifact['target'],'predicted_label':str(classes[int(order[0])]),
        'top_predictions':[{'label':str(classes[i]),'margin':float(scores[i])} for i in order],
        'developer_company':MODEL_TO_COMPANY.get(str(classes[int(order[0])])) if artifact['target']=='model' else str(classes[int(order[0])]),
        'score_type':'uncalibrated decision-function margin, not probability',
        'scope':'Closed set of dataset labels only; no out-of-distribution credibility or authorship guarantee.',
        'cleaning_version':CLEANING_VERSION}
