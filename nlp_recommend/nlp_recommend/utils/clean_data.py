from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import nltk
# nltk.download('stopwords')

# for tokenizer
# nltk.download('punkt')
# nltk.download('wordnet')

first_n_words = 200

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z#+_:,.]')
PATTERN_S = re.compile("\'s")  # remove multispaces like tab
PATTERN_RN = re.compile("\\r\\n")
PATTERN_PUNC = re.compile(r"[^\w\s]")
PATTERN_BEG = re.compile(r":\s")  # remove space : space

STOPWORDS = set(stopwords.words('english'))
MIN_WORDS = 1


def clean_beginning(text):
    match = PATTERN_BEG.match(text)
    if match:
        if match.start() == 0:
            text = PATTERN_BEG.sub('', text)
    return text


def clean_text(text, only_symbols=False):
    """
        text: a string
        # TODO What is doing spacy

        return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = re.sub(PATTERN_S, ' ', text)
    text = re.sub(PATTERN_RN, ' ', text)
    # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing.
    text = BAD_SYMBOLS_RE.sub(' ', text)

    if not only_symbols:
        text = re.sub(PATTERN_PUNC, ' ', text)
        text = text.replace('x', ' ')
        text = re.sub(r'\W+', ' ', text)  # removing non words character
        text = re.sub(r"\d+", " ", text)
    return text


def tokenizer(sentence, min_word=MIN_WORDS, lemmatize=True, stopwords=STOPWORDS):
    # Remove short words (under 3 characters) from the tokens
    if lemmatize:
        stemmer = WordNetLemmatizer()
        tokens = [stemmer.lemmatize(w) for w in word_tokenize(sentence) if (len(w) > min_word
                                                                            and w not in stopwords)]
    else:
        tokens = [w for w in word_tokenize(sentence) if (len(w) > min_word
                                                         and w not in stopwords)]
    return tokens


def format_text(list_lines):
    pattern = re.compile('\[|\]')
    list_lines = list_lines[0].split(', ')
    one_text = ' '.join(list_lines)
    one_text = one_text.replace('\' \'', ' ')
    list_lines = one_text.split('. ')
    nb_lines = len(list_lines)
    # remove header and footer
    twenty_per = int(0.2 * nb_lines)
    reduced = list_lines[twenty_per:-2*twenty_per]
    reduced = [sentence + '.' for sentence in reduced if (
        len(sentence) > 12 and not 'chapter' in sentence.lower())]
    reduced = [sentence.capitalize().strip()
               for sentence in reduced if not pattern.search(sentence)]
    return reduced
