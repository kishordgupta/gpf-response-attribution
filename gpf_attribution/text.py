"""Response-only text preparation and structural features."""
import hashlib
import re
import unicodedata
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

CLEANING_VERSION = 'response-only-v1'
_HEADER = re.compile(r'^\s*(?:Gemini said|Response(?:\s+\d+)?:)\s*\n+', re.I)
NAME_TERMS = (
    'Meta Llama Maverick 4', 'Meta Llama 3.3 Turbo', 'Claude Sonnet 4.5',
    'Claude Haiku 4.5', 'Gemini 2.5 Flash', 'Gemini 2.5 Pro', '3.5 Flash-Lite',
    'Mistral Medium 3', 'Mistral Large 3', 'GPT 5.4 Mini', 'GPT 5.4', 'Grok Fast',
    'ChatGPT', 'OpenAI', 'Anthropic', 'Google Gemini', 'Gemini', 'Claude',
    'Mistral', 'Llama', 'Grok', 'xAI', 'Google', 'Meta', 'Together AI', 'OpenRouter',
)
_NAME_PATTERN = re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(s) for s in sorted(NAME_TERMS,key=len,reverse=True))+r')(?!\w)', re.I)

def clean_response(text):
    """Remove exact leading display labels; normalize whitespace without adding data."""
    text = str(text).replace('\r\n','\n').replace('\r','\n')
    while True:
        stripped = _HEADER.sub('', text, count=1)
        if stripped == text: break
        text = stripped
    return re.sub(r'\s+', ' ', text).strip()

def normalized_response_hash(text):
    normalized = unicodedata.normalize('NFKC', clean_response(text)).casefold()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

def mask_model_names(text):
    """Literal model/vendor name masking; not removal of every indirect identifier."""
    return _NAME_PATTERN.sub('[MODEL_OR_COMPANY]', text)

STYLE_FEATURE_NAMES = [
    'words','characters','mean_word_length','std_word_length','unique_word_ratio',
    'sentence_endings','words_per_sentence','uppercase_letter_ratio','digit_ratio',
    'comma_per_1000_chars','colon_per_1000_chars','semicolon_per_1000_chars',
    'question_per_1000_chars','exclamation_per_1000_chars','quotes_per_1000_chars',
    'dash_per_1000_chars','parentheses_per_1000_chars','markdown_per_1000_chars',
]

class StyleFeatures(BaseEstimator, TransformerMixin):
    """Only numerical surface/length statistics of supplied response text."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        rows=[]
        for text in X:
            words=re.findall(r'\b\w+\b',text); n=max(len(words),1); chars=max(len(text),1)
            lengths=np.array([len(w) for w in words] or [0],dtype=float)
            letters=[c for c in text if c.isalpha()]
            sentence_ends=len(re.findall(r'[.!?]+(?:\s|$)',text))
            rates=[1000*sum(text.count(c) for c in chars_set)/chars for chars_set in
                   [',',':',';','?','!','"\'“”‘’','-–—','()[]','*#_`']]
            rows.append([len(words),len(text),float(lengths.mean()),float(lengths.std()),
                len(set(w.casefold() for w in words))/n,sentence_ends,n/max(sentence_ends,1),
                sum(c.isupper() for c in letters)/max(len(letters),1),
                sum(c.isdigit() for c in text)/chars,*rates])
        return np.asarray(rows,dtype=np.float64)
    def get_feature_names_out(self, input_features=None):
        return np.asarray(STYLE_FEATURE_NAMES,dtype=object)
