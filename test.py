import numpy as np
import os
import re
import pandas as pd
import nltk
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import string
import sys

def concat(f):
    f_concat = [""]*f.shape[0]
    for i in range(f.shape[0]):
        f[i, 1] = "" if (type(f[i, 1])==float) else f[i,1]
        f_concat[i] = f[i, 0] + f[i, 1]
    return f_concat

test_file = sys.argv[1]
f_test = pd.read_csv(test_file, header = None,  encoding='ISO-8859-1') 
f_test = f_test.to_numpy()[1:, :] 
test_size = f_test.shape[0]

f_test_concat = concat(f_test[:,0:2])

Pkl_Filename = "email_model"

# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:  
    vectorizer = pickle.load(file)
    model = pickle.load(file)

X_test = vectorizer.transform(f_test_concat).toarray()

Y_svm = model.predict(X_test)

out_name = sys.argv[2]
out_f = open(out_name, "w") 
for i in range(len(Y_svm)):
    out_f.write(str(Y_svm[i])+'\n')

out_f.close()


