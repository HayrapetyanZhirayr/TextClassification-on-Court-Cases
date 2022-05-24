"""
Module for taxonomy relied functions. Extracting data and creating taxonomies.
"""
import numpy as np
import pandas as pd

def preprocess_taxonomy(taxonomy_df_raw):
    """
    This function is designed to preprocess a pandas dataframe consisting of
        taxonomy of the categories.
    :param taxonomy_df_raw: pandas DataFrame of taxonomy with columns
        ['Нумерация с учётом Level 0', '"Правильная" нумерация',
        'Index (по КАД)',	'Level 0 = вид спора',	'Level 1 = категории дела',
        'Level 2',	'Level 3',	'Level 4',	'Category']
    :type taxonomy_df_raw: pandas.core.frame.DataFrame

    :return: preproccesed taxonomy DataFrame
    :rtype: pandas.core.frame.DataFrame

    Pipeline:
        - extracting label of each node (e.g. Споры о заключении ..)
        - extracting level of each node (e.g. 2.1.1.3)
        - adding depth column (depth of taxonomy node)
    """
    taxonomy_df = (
        taxonomy_df_raw.fillna('')
        .assign(
            label = lambda df: df.iloc[:, 3:].sum(axis=1)
                # .str.replace('(', '', regex=False)
                # .str.replace(')', '', regex=False)
                ,
            level = lambda df: (df['Index (по КАД)'].str.strip('.') + '.')
                .replace('.', np.nan)
        )
    )[['level', 'label']].dropna(axis=0).reset_index(drop=True)

    taxonomy_df.loc[taxonomy_df['level'] == '25.1.3.1.', 'level'] = '25.3.1.'

    taxonomy_df = (
        taxonomy_df.assign(
            depth = lambda df: df['level']
                .str.strip('.')
                .str.split('.')
                .apply(len)
        )
    )
    return taxonomy_df


def extract_taxonomy_df(dispute_categories):
    """
    This function is designed to transform list of categories into detalized
        pandas DataFrame of categories.

    :param dispute_categories: list of categories (e.g. [2.1.1.3. Споры о .., ])
    :type dispute_categories: list of str

    :return: pandas DataFrame with columns ['level', 'label', 'depth']
    :rtype: pandas.core.frame.DataFrame
    """
    data = []
    for cat in dispute_categories:
        dot_split = cat.split('. ')  # separating level(index) from label(name)
        index = dot_split.pop(0)
        depth = len(index.split('.'))
        index += '.'
        if len(dot_split) > 1:  #  if there was a adot in label string
            label = '. '.join(dot_split)
        else:
            label = dot_split[0]
        data.append([index, label, depth])
    return pd.DataFrame(data, columns=['level', 'label', 'depth'])


def get_categories_from_taxonomy(taxonomy_df):
    """
    Extacting list of categories with their names from taxonomy_df

    :param taxonomy_df: pandas datFrame with categories description,
        column 'level' is the index of the category (e.g. 2.1.1.3)
        column 'label' is the name of the category (e.g. Споры о заключении ...)
    :type taxonomy_df: pd.core.frame.DataFrame

    :return: list of categories (e.g. ['2.1.1.3 Споры о заключении ..., '])
    :rtype: list of str
    """
    return list(
        map(
            lambda x: x[0].strip('.') + '. '+ x[1].strip(),
            taxonomy_df[['cCode', 'Descr']].values
        )
    )


def get_labels(sample_cat_list, taxonomy_df):
    """
    Mapping sample string categories to integer labels.

    :param sample_cat_list: list of string categories of the sample objects
    :type sample_cat_list: list of str
    :param taxonomy_df: pandas DataFrame with categories description
    :type taxonomy_df: pd.core.frame.DataFrame

    :return: list of integer categories
    :rtype: list of int
    """
    labels = []
    categs = get_categories_from_taxonomy(taxonomy_df)
    categs_dict = dict(zip(categs, range(len(categs))))
    for i in range(len(sample_cat_list)):
        cat = sample_cat_list[i]
        labels.append(categs_dict[cat])

    return labels


def get_first_parent_label(sample_label_list, taxonomy_df):
    """
    Mapping sample labels into their first parent(root descendant) labels.

    :param sample_label_list: list of int categories of the sample objects
    :type sample_cat_list: list of int
    :param taxonomy_df: pandas datFrame with categories description
    :type taxonomy_df: pd.core.frame.DataFrame

    :return: list of integer categories of objects parents
    :rtype: list of int
    """
    label_to_par_index = (taxonomy_df['level'].str.split('.')
        .map(lambda x: (x[0])+'.').to_dict()
    )
    index_to_label = dict(zip(taxonomy_df['level'], taxonomy_df.index))
    par_labels = []
    for label in sample_label_list:
        par_index = label_to_par_index[label]
        par_label = index_to_label[par_index]
        par_labels.append(par_label)

    return par_labels
