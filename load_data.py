"""
Module to load data from disk. Auxillary functions.
"""
import numpy as np
import pandas as pd
import glob
import os
import gzip
import json
from tqdm import tqdm


# from bs4 import BeautifulSoup
# from navec import Navec


def all_entries_in(folder):
    """
    Returns all entries of given folder (or directory in other words)
    """
    return glob.glob(
        os.path.join(folder, '*')
    )


def mkdir(dir_):
    if not os.path.exists(dir_):
        os.mkdir(dir_)


def yield_document(read_dir):
    """
    For a  dir with folder structure [reg_folder --> year_folder --> doc] yields
        each document

    # Arguments
        read_dir: str, directory to parse
    """
    decode_errors_count = 0
    reg_folders = list(all_entries_in(read_dir))
    for region_folder in tqdm(reg_folders):
        for year_folder in all_entries_in(region_folder):
            for doc_xml_gz_path in all_entries_in(year_folder):
                try:
                    doc_xml_gz = gzip.open(doc_xml_gz_path).read().decode('utf8')
                    yield doc_xml_gz
                except UnicodeDecodeError:
                    decode_errors_count += 1
    print('Number of decode errors ::', decode_errors_count)


def yield_document2(read_dir):
    """
    For a  dir with folder structure [reg_folder --> year_folder --> doc] yields
        each document.

    Same as yield_document function, but without counting decode errors.

    # Arguments
        read_dir: str, directory to parse
    """
    reg_folders = list(all_entries_in(read_dir))
    for region_folder in (reg_folders):
        for year_folder in all_entries_in(region_folder):
            for doc_xml_gz_path in all_entries_in(year_folder):
                doc_xml_gz = gzip.open(doc_xml_gz_path).read().decode('utf8')
                yield doc_xml_gz_path, doc_xml_gz


def yield_preprocessed_json_from_disk(read_dir):
    """
    This function helps to parse data storage format. Parses the preprocessed
      jsons of documents and yield their text and casenumber

    # Arguments
        read_dir: str, directory to parse

    # Yields
        x_str: str, preprocessed document string
        y_str: str, document casenumber
    """
    reg_folders = list(all_entries_in(read_dir))
    for region_folder in tqdm(reg_folders):
        for year_folder in all_entries_in(region_folder):
            for doc_json_path in all_entries_in(year_folder):
                with open(doc_json_path, 'r') as f:
                    xy = json.load(f)
                    x_str = xy['x']
                    y_casenumber = xy['y']
                yield x_str, y_casenumber


def read_preprocessed_from_disk_to_numeric_to_ram(read_dir, embedding_dir,
    emb_dim=300):
    from navec import Navec
    """
    Applies Navec embedding for document texts from read_dir storage using
        yield_preprocessed_json_from_disk function to parse storage.

    # Arguments
        read_dir: str, directory to parse
        embedding_dir: str, directory where embedding model is stored
        emb_dim: int, embedding dimension for tokens

    # Returns
        X: np.array, array of shape (?, emb_dim) with document embeddings
        Y: np.array, array of strings of documents casenumbers
    """
    navec = Navec.load(embedding_dir)  # model to embed texts
    X, Y = [], []
    # print(read_dir)
    for x_str, y_casenumber in yield_preprocessed_json_from_disk(read_dir):
        x_numeric = np.zeros(emb_dim)
        counts = 0
        for token in x_str.split(' '):
            if token in navec:
                x_numeric += navec[token]
                counts += 1

        if counts:
            x_numeric = x_numeric / counts

        X.append(x_numeric)
        Y.append(y_casenumber)
    return np.vstack(X), np.array(Y)


def read_from_disk_preprocess_save(read_dir, save_dir, lemmatize):
    from bs4 import BeautifulSoup
    """
    Reads raw documents from given read_dir, extracts texts and their labels,
        lemmatize texts if Flag lemmatize is True and saves document in json
        format with tags "x" and "y"
    # Arguments:
        read_dir: str, directory to parse
        save_dir: str, directory to save
        lemmatize: bool, if True the document texts will be lemmatized
    """
    mkdir(save_dir)
    reg_folders = list(all_entries_in(read_dir))
    for region_folder in tqdm(reg_folders):
        for year_folder in all_entries_in(region_folder):
            for doc_xml_gz_path in all_entries_in(year_folder):
                doc_xml_gz = gzip.open(doc_xml_gz_path)
                s = BeautifulSoup(doc_xml_gz.read().decode('utf8'),
                    'html.parser')
                doc_xml_gz.close()
                casenumber = s.casenumber
                if casenumber is None:
                    casenumber = s.case_number
                if casenumber:
                    casenumber = casenumber.text
                    x = s.text
                    x = ' '.join(preprocess_text(s.text, lemmatize=lemmatize))
                    y = casenumber
                    residual, source_name = os.path.split(doc_xml_gz_path)
                    residual, year = os.path.split(residual)
                    residual, region = os.path.split(residual)

                    region_folder = os.path.join(save_dir, region)
                    mkdir(region_folder)
                    year_folder = os.path.join(region_folder, year)
                    mkdir(year_folder)
                    destination = os.path.join(
                        year_folder, f'{source_name.split(".")[0]}.json'
                    )
                    with open(destination, 'w') as f:
                        json.dump({'y': y, 'x' : x}, f, ensure_ascii=False)


def extract_data_from_disk(read_dir):
    from bs4 import BeautifulSoup
    """
    Reads documents from read_dir storage and returns document texts X and
      document casenumbers Y
    # Arguments:
        read_dir: str, directory to parse

    # Returns:
        X: list[str], texts of documents
        Y: list[str], casenumbers of documents
    """
    X, Y = [], []
    reg_folders = list(all_entries_in(read_dir))
    for region_folder in tqdm(reg_folders):
        for year_folder in all_entries_in(region_folder):
            for doc_xml_gz_path in all_entries_in(year_folder):
                doc_xml_gz = gzip.open(doc_xml_gz_path)
                s = BeautifulSoup(doc_xml_gz.read().decode('utf8'),
                'html.parser')
                doc_xml_gz.close()
                casenumber = s.casenumber
                if casenumber is None:
                    casenumber = s.case_number
                if casenumber:
                    casenumber = casenumber.text

                    x = s.text
                    y = casenumber
                    X.append(x)
                    Y.append(y)
    return X, Y


def get_old_case_to_category_mapping(
        read_dir="raw_data/33m-russian-courts-cases-by-suvorov/разметка/"):

    """
    Reads a directory where a lot of csv files with document labels are stored
        and converts those to a dictionary that maps casenumber to CategoryID

    # Arguments:
        read_dir: str, directory to parse

    # Returns:
        dict[str, int], a dictionary that is a mapping from casenumber of
          document to its assigned category_id
    """
    if not read_dir.endswith("/"):
        read_dir = read_dir + "/"

    mapping = {}
    for f_name in glob.glob(f'{read_dir}*.csv'):
        mapping_subset = dict(
            pd.read_csv(f_name)[['Number', 'CategoryID']]
            .values
        )
        mapping.update(mapping_subset)

    return mapping
