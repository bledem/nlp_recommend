import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# for tokenizer
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

first_n_words = 200

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z#+_:,.]')
PATTERN_S = re.compile("\'s") # remove multispaces like tab
PATTERN_RN = re.compile("\\r\\n")
PATTERN_PUNC = re.compile(r"[^\w\s]")
STOPWORDS = set(stopwords.words('english'))
MIN_WORDS = 1

def clean_text(text, only_symbols=False):
    """
        text: a string
        # TODO What is doing spacy
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = re.sub(PATTERN_S, ' ', text)
    text = re.sub(PATTERN_RN, ' ', text)
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub(' ', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    if not only_symbols:
        text = re.sub(PATTERN_PUNC, ' ', text)
        text = text.replace('x', ' ')
        text = re.sub(r'\W+', ' ', text) #removing non words character
        text = re.sub(r"\d+", " ", text)
    return text

def tokenizer(sentence, min_word=MIN_WORDS, lemmatize=True, stopwords=STOPWORDS):
    # Remove short words (under 3 characters) from the tokens
    if lemmatize:
        stemmer = WordNetLemmatizer()
        tokens = [stemmer.lemmatize(w) for w in word_tokenize(sentence) if (len(w)>min_word 
                                                        and w not in stopwords)]
    else:
        tokens = [w for w in word_tokenize(sentence) if (len(w)>min_word 
                                                        and w not in stopwords)]
    return tokens