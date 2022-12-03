# not needed as of now , things appended in csv_read.py

import pandas as pd
from ast import literal_eval
import json
import numpy as np

paired_cfg_filepath = "test_true_pairs.csv"
i2id_file_path = "../../ecsTest/word2id.json"
df = pd.read_csv(paired_cfg_filepath)
npy_file = "../../ecsTest/embedding_matrix.npy"

# --------------------------------------------------
# To identify max number of features in the paired ids
elements_len = []
print(len(df))

for i in range(0,len(df)):
    temp = []
    for element in literal_eval(df.iloc[i][2]) :
        # print(element)
        # print(len(element))
        temp.append(len(element))
    elements_len.append(max(temp))
    
print(elements_len)
print(len(elements_len))
print(max(elements_len))

# -------------------------------
# load i2id json file
# i2id_f = open(i2id_file_path, 'r')
# i2id = json.load(i2id_f)
# i2id_f.close()
# # _--------------------

        
     
# f = np.load(npy_file)
# print(f.shape)
# print(np.min(f))
# print(f[22].shape)
