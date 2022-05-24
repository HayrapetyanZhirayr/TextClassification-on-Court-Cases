# TextClassification-on-Court-Cases

This repo provides different models for text classification.

In the case of the data used in this project (Russian Court Cases), a certain structure was defined between the categories of texts -- a hierarchy.(or in other words a 
tree where one categoy may be more general then the other one but still cover it, in other words, parent-child relations were defined.) So the repo also provides tools 
for convenient usage of hierarchical structures and tools for checking the correctness of the list of elements from this structure. (e.g. whether they are ordered by DFS
or not)

Using torch, several classification models have been implemented, two of which are described in the google-ml tips https://developers.google.com/machine-learning/guides/text-classification,
but not implemented in torch. 
These two models TF IDF + ANOVA TOP K SELECTION + MLP and Separable Convolutions Neural Network gave the best classification accuracy. 
The fundamental difference between these two models is that the last one takes into account the words order. It is a sequential model with embedding of words.

In most models TFIDF measure was used for vectorization of texts except the SepConvolutionNN, where vectorized text is presented as a sequence of integers with a 
defined vocabulary mapping from words to those integers and backwards.

Also an algorithm of choosing the most relevant features from an itemset was proposed by HSE Prof. Boris Mirkin https://www.hse.ru/en/staff/bmirkin. 

**DESCRIPTION OF FILES**

All Modules have a small description inside., alternative one is given here for a better understanding

models.py :: Implementation of different used models including SepCNN from google-ml tips.

mirkin_LS_method.py :: Mirkin's method for choosing a maximally contributing itemset from given set of elements utilizing iterative
alternative least_squares method.     


-- taxonomies (hierarchical structures)

taxonomy_utils.py :: functions for extracting a taxonomy structury from csv table and other useful utils
taxonomy_eda.ipynb :: exploratory data analysis for russian court categories structure
dfs_sort_utils.py :: script for checking correctness of order of list of tree nodes (dfs or not)
                  

ast_local.py :: a vectorizer based on Annotated Suffix Trees
tf_keras_utils.py :: a sequential vectorizer. This one uses tensorflow keras.

-- utils

load_data.py :: utils for loading data
data_utils.py :: utils for preprocessing data
torch_utils.py :: torch wrappers for training NN
date_utils.py :: utils for date formats

measure_Hypothesis.ipynb :: analyzing a hypothesis on classification based only on relevance of category name to document text    
data_manipulations.ipynb :: auxiliary notebook used to changed data saving formats and other related things.

-- different models prototyping notebooks.
models_prototypingSelectK_Tfidf.ipynb      
models_prototypingSepCNN.ipynb            
models_prototypingSMALL_FIXED_TFIDF.ipynb 
models_prototypingSMALL_MIRKIN.ipynb      
models_prototypingSMALL_NAVEC.ipynb      
