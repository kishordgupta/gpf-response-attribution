"""Reproducible response-only attribution baselines for a fixed captured dataset."""
from .text import CLEANING_VERSION,clean_response,normalized_response_hash,mask_model_names
from .data import DATA_SHA256,MODEL_TO_COMPANY,EXCLUDED_CELLS,load_dataset
from .estimators import make_classifier
from .inference import predict_text
