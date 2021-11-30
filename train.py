import numpy as np
import os
import re
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import string

import pickle
import nltk
import sys

train_file = sys.argv[1]

def concat(f):
    f_concat = [""]*f.shape[0]
    for i in range(f.shape[0]):
        f[i, 1] = "" if (type(f[i, 1])==float) else f[i,1]
        f_concat[i] = f[i, 0] + f[i, 1]
    return f_concat

# read input
f = pd.read_csv(train_file, header = None,  encoding='ISO-8859-1') 
f = f.to_numpy()[1:, :]

data_size = f.shape[0]

dict_labels = {"Meeting":1, "For":2, "Selection": 3, "Policy":4, "Recruitment": 5, "Assessment":6, "Other":7}

f_concat = concat(f)
labels = []
for i in range(f.shape[0]):
    l = re.split(r'[\s,.""''*]+', f[i][2])
    labels.append(dict_labels[l[0]])

vectorizer = TfidfVectorizer(stop_words='english', smooth_idf = True, sublinear_tf=True, min_df=6)

X = vectorizer.fit_transform(f_concat).toarray()
lr_model = LogisticRegression(random_state=1, class_weight="balanced", solver='liblinear', C=11).fit(X, labels)

Pkl_Filename = "email_model"

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(vectorizer, file)
    pickle.dump(lr_model, file)