"""
Module to Preprocess Data.
"""
import re
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import string
russian_stopwords = set(stopwords.words('russian'))
russian_stemmer = SnowballStemmer("russian")
from bs4 import BeautifulSoup
from tqdm import tqdm


def preprocess_text(text, lemmatize=True):
    """
    This function is used to preprocess russian language text.

    :param text: text to preprocces
    :type text: str

    :return: list of preprocessed tokens
    :rtype: list of str

    Pipeline:
        - remove proper names
        - сonvert to lowercase
        - remove punctutaion
        - remove english text
        - remove months
        - remove digits
        - remove one character words (1)
        - tokenize the string
        - remove russian stopwords
        - stemmatize using SnowballStemmer for russian language from nltk lib
        - (1)
    """
    dtext = text
    dtext = re.sub(r'[A-Z|А-Я]+\w*', '', dtext)  # removing names surnames and cities
    dtext = dtext.lower()  # to lower case
    dtext = re.sub(rf'[{string.punctuation}№–]+', ' ', dtext)  # removing punctuation
    dtext = re.sub(r'[a-z]+', ' ', dtext)  # removing english text

    dtext = re.sub(
        r'(январ|феврал|март|апрел|май|июн|июл|август|сентябр|октябр|ноябр|декабр)(\w*)', ' ',
                      dtext)  # removing months

    dtext = re.sub(r'\d', '', dtext)  # replacing digits
    dtext = re.sub(r' \w ', ' ', dtext)  # replacing one character words
    dtext = word_tokenize(dtext)  # tokenizing
    dtext = [w for w in dtext if w not in russian_stopwords]  # removing stopwords
    if lemmatize:
        dtext = [russian_stemmer.stem(w) for w in dtext]
    dtext = [w for w in dtext if len(w) > 1]

    return dtext


def build_count_dict(elements_l):
    """
    :param elements_l: list of elements to count
    :type elements_l: list of str
    :return: dict of counts of elements in elements_l list
    :rtype: dict of str: int
    """
    count_dict = {}
    for el in elements_l:
        count_dict[el] = count_dict.get(el, 0) + 1
    return count_dict


def preprocess_data(data):
    """
    Wrapper for .preprocess_text() function to preprocess text collection.

    :param data: dict with keys specifyings categories of texts and values as
        list of raw texts of given category
    :type data: dict of lists of str

    :return: dict with keys specifyings categories of texts and values as
        list of preprocessed texts of given category in the format of dictinoary
            with keys as tokens and values as counts of tokens in text
    :rtype: dict[str:list[dict[str:int]]]
    """
    pdata = defaultdict(list)  # preprocessed data
    for disp_cat in tqdm(data.keys()):
        for cat_data in data[disp_cat]:
            _, _, text = cat_data
            ptext = preprocess_text(text)  # preprocessed text
            ptext_counts = build_count_dict(ptext)
            pdata[disp_cat].append(ptext_counts)

    return pdata


def make_corpus(data, dispute_categories):
    """
    This function is designed to preprocess and combine texts from same category
    into one text making a corpus of texts where each text represents some
    category.

    :param data: dict with keys specifyings categories of texts and values as
        list of raw texts of given category
    :type data: dict of lists of strings

    :param dispute_categories: list of some keys from data dict
    :type dispute_categories: list of str

    :return: list of texts each representing specific category. Indexation is
        according to dispute_categories list indexation.
    :rtype: list of str
    """
    corpus = []
    for disp_cat in tqdm(dispute_categories):
        document = ''
        for cat_data in data[disp_cat]:
            _, _, text = cat_data
            dtext = preprocess_text(text)
            dtext = ' '.join(dtext)
            document += dtext
            document += ' '
        corpus.append(document)
    return corpus


def make_word_counts_per_category(data, dispute_categories):
    """
    This function is designed to count number of entries of each token in
        each category (union of texts of one category)
    :param data: dict with keys specifyings categories of texts and values as
        list of texts of given category
    :type data: dict of lists of str

    :param dispute_categories: list of some keys from data dict
    :type dispute_categories: list of str

    :return: list of dictionaries with token counts in category. Indexation is
        according to dispute_categories list indexation.
    :rtype: list of dict str: int
    """
    counts_all = []
    for disp_cat in tqdm(dispute_categories):
        counts = {}
        for cat_data in data[disp_cat]:
            _, _, text = cat_data
            dtext = preprocess_text(text)
            for token in dtext:
                counts[token] = counts.get(token, 0) + 1
        counts_all.append(counts)
    return counts_all


def make_vocabulary(data):
    """
    This function is designed to extract all unique tokens from a raw text
    collection with their counts. Includes preprocessing of texts.

    :param data: dict with keys specifyings categories of texts and values as
        list of raw texts of given category
    :type data: dict of lists of str

    :return: dict of unique preprocessed tokens(stems) with their counts in the
        whole collection.
    :rtype: dict of str: int
    """
    vocab = {}
    for disp_cat in tqdm(data.keys()):
        for cat_data in data[disp_cat]:
            _, _ , text = cat_data
            dtext = preprocess_text(text)
            for w in dtext:
                vocab[w] = vocab.get(w, 0) + 1
    return vocab


def extract_casenumber(doc):
    """
    Extracting raw casenumber from a decoded document string

    # Arguments
        doc: string, document string

    # Returns
        string, case_number of document
        bool, True if succeeded to extract case_number
    """
    ext1 = re.findall('Дело\s*№\s*.+', doc)
    if len(ext1) > 0:
        case = ext1[0]
        return case, True
    ext3 = re.findall('[Дд]ел[оау].+\d.+', doc)
    if len(ext3) > 0:
        case = ext3[0]
        return case, True

    ext2 = re.findall('№.+', doc)
    if len(ext2) > 0:
        case = ext2[0]
        return case, True
    else:
        return '', False


def process_raw_casenumber(casenumber):
    """
    Processing raw casenumber string

    # Arguments
        casenumber: string, casenumber string

    # Returns
        string, preprocessed casenumber
        bool, True if succeeded to preprocess case_number
    """
    match = re.match('([Дд]ел[оау])?' +
                                    '\s*' +
                                    '(№)?' +
                                    '\s*' +
                                    '([АA]\d+П?|СИП|ВАС)' +
                                    '\s*' +
                                    '(-|–|\?)' +
                                    '\s*' +
                                    '(\d[\d-]+)' +
                                    '(ИП)?' +
                                    '\s*' +
                                    '(/)' +
                                    '\s*' +
                                    '(\d+)', casenumber
    )

    if not match:
        return '', False

#     return  match.groups()
    case_number_proc = list(
        filter(lambda x: x is not None, match.groups()[2:])
    ) # skipping No and Дело groups

    case_number_proc[-1] = case_number_proc[-1][:4]
    case_number_proc = ''.join(case_number_proc)
    case_number_proc = ( case_number_proc
                            .replace('?', '-')
                            .replace('–', '-')
                       )

    return case_number_proc, True


def make_xml_string(date, court, text, casenumber, category):
    return f'''<html><body>
                <court>{court}</court>
                <casenumber>{casenumber}</casenumber>
                <category>{category}</category>
                <date>{date}</date>
                <p><span>{text}</span></p>
                <\body><\html>'''


def process_xml_string(doc):
    """
    For autokad decoded document strings
    return court, casenumber, category, date, text

    # Arguments
        doc: string, xml string of document
    """
    s = BeautifulSoup(doc, 'html.parser')
    doc_tags = [
        s.court, s.casenumber, s.category, s.date, s.span
        ]
    doc_info = [tag.text if tag else None for tag in doc_tags]
    court, casenumber, category, date, text = doc_info
    return court, casenumber, category, date, text
