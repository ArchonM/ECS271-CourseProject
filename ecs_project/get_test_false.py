# Read csv files

import pandas as pd
import json
import networkx as nx 
from networkx import json_graph 
from ast import literal_eval
import numpy as np

csv_path = "../../ecsTest/ecs_project/csv_files/"
cfg_csv_filename = "lstm_cfg.csv"
test_pair_csv_filename = "test_pairs.csv"
train_pair_csv_filename = "train_pairs.csv"
csv_output_file = "final.csv"
i2id_file_path = "../../ecsTest/word2id.json"
embedding_vec_file_path = "../../ecsTest/embedding_matrix.npy"

df_cfg = pd.read_csv(csv_path + cfg_csv_filename)
df_test_pairs = pd.read_csv(csv_path + test_pair_csv_filename)
df_train_pairs = pd.read_csv(csv_path + train_pair_csv_filename)

df_cfg_shape = df_cfg.shape
df_test_pairs_shape = df_test_pairs.shape
df_train_pairs_shape = df_train_pairs.shape

print("df_cfg_shape : ", df_cfg_shape, " ; Coloumns : ", list(df_cfg.columns))
print("df_test_pairs_shape :", df_test_pairs_shape, " ; Coloumns : ", list(df_test_pairs.columns))
print("df_train_pairs_shape : ", df_train_pairs_shape, " ; Coloumns : ", list(df_train_pairs.columns))

print("**************")
list_test_true_pairs = literal_eval(df_test_pairs.loc[0]['true_pair']) # Contains the list of ids of true_pairs (from test dataset)
print("list_test_true_pairs len : ", len(list_test_true_pairs))

list_test_false_pairs = literal_eval(df_test_pairs.loc[0]['false_pair']) # Contains the list of ids of false_pairs (from test dataset)
print("list_test_false_pairs len : ", len(list_test_false_pairs))

list_train_true_pairs = []
list_train_false_pairs = []

# These are for training pairs, too long , comment when not needed
# for i in range(0, df_train_pairs_shape[0]) :
#     list_train_true_pairs.extend(literal_eval(df_train_pairs.loc[i]['true_pair'])) # Contains the list of ids of true_pairs (from test dataset)

#     list_train_false_pairs.extend(literal_eval(df_train_pairs.loc[i]['false_pair'])) # Contains the list of ids of false_pairs (from test dataset)

# print("list_train_true_pairs len : ", len(list_train_true_pairs))
# print("list_train_false_pairs len : ", len(list_train_false_pairs))

# Just to test
# ele = list_test_true_pairs[3][0]
# print(ele)
# val = df_cfg.loc[df_cfg['id'] == ele]['lstm_cfg']
# print(val)
# print(type(val))
# print(len((val)))
# lstm_cfg_feature = val.iloc[len(val)-1]
# print("###########")
# j_load = json.loads(lstm_cfg_feature)
# print(j_load)
# print(type(j_load))
# print("********")
#--------------------------------------------
# Convert instruction to id

# load i2id json file
i2id_f = open(i2id_file_path, 'r')
i2id = json.load(i2id_f)
i2id_f.close()

def convert_to_ids(instructions_list):
        ret_array = []
        # For each instruction we add +1 to its ID because the first
        # element of the embedding matrix is zero
        for x in instructions_list:
            if x in i2id:
                ret_array.append(i2id[x] + 1)
            elif 'X_' in x:
                # print(str(x) + " is not a known x86 instruction")
                ret_array.append(i2id['X_UNK'] + 1)
            elif 'A_' in x:
                # print(str(x) + " is not a known arm instruction")
                ret_array.append(i2id['A_UNK'] + 1)
            else:
                # print("There is a problem " + str(x) + " does not appear to be an asm or arm instruction")
                ret_array.append(i2id['X_UNK'] + 1)
        return ret_array
#--------------------------------------------
# Normalize
max_instructions = 100

def normalize(f):
        f = np.asarray(f[0:max_instructions])
        length = f.shape[0]
        if f.shape[0] < max_instructions:
            f = np.pad(f, (0, max_instructions - f.shape[0]), mode='constant')
        return f
#----------------------------------------------
# Load embedding vec
embedding_vec = np.load(embedding_vec_file_path)
print(embedding_vec.shape)

def convert_to_vec(i2id_list):
    ret_vec_arr = []
    for id in i2id_list:
        ret_vec_arr.append(embedding_vec[id])
    return ret_vec_arr
        
    
#--------------------------------------------
# Actual logic
# feature extraction for pairs in test_pairs_true_labels. Same could be applied to others pairs i.e test_false_pairs, train_true_pairs, train_false_pairs.
test_true_paired_features = [] # contains list of mnemonic features from two ids in pair
test_true_paired_feature_id = []
test_true_paired_feature_vec = []
test_true_id_pair_list = []
for pairs in list_test_false_pairs :
    pair_feature_list = []
    pair_feature_id_list = []
    pair_feature_vec_list = []
    test_true_id_pair_list.append(pairs)
    # print(pairs)
    for ids in pairs :
        id_feature = []
        # print(ids)
        val = df_cfg.loc[df_cfg['id'] == ids]['lstm_cfg']
        lstm_cfg_feature = val.iloc[len(val)-1]
        lstm_cfg_feature_dict = json.loads(lstm_cfg_feature)
        # print(lstm_cfg_feature_dict)
        for nodes in lstm_cfg_feature_dict['nodes'] :
            # print(nodes)
            if nodes['features'] : 
                # print(nodes['features'])
                id_feature.extend(nodes['features'])
                # print(id_feature)
        pair_feature_list.append(id_feature)
        normalized_feature_ids = normalize(convert_to_ids(id_feature))
        pair_feature_id_list.append(normalized_feature_ids)
        pair_feature_vec_list.append(convert_to_vec(normalized_feature_ids))
        # print(pair_feature_list)
        # print("~~~~~~~~~~~~~~~~~")
    test_true_paired_features.append(pair_feature_list)
    test_true_paired_feature_id.append(pair_feature_vec_list)
    test_true_paired_feature_vec.append(pair_feature_vec_list)
    
    # print("************")
# print(test_true_paired_features)
# print(test_true_paired_feature_vec)
print(len(test_true_paired_features))
print(len(test_true_paired_feature_id))
print(len(test_true_id_pair_list))
assert(len(test_true_paired_features) == len(test_true_id_pair_list))
test_true_pair_labels = [1]*len(test_true_paired_features)

test_true_pair_dict = {'ids':test_true_id_pair_list , 'features': test_true_paired_features, 'feature_ids':test_true_paired_feature_id ,'labels': test_true_pair_labels}
test_true_pair_dict_vec = {'feature_vec' : test_true_paired_feature_vec ,'labels': test_true_pair_labels}
test_true_pair_df = pd.DataFrame(test_true_pair_dict)
test_true_pair_df_vec = pd.DataFrame(test_true_pair_dict_vec)
print(test_true_pair_df)
print(len(test_true_pair_df))
print(len(test_true_paired_feature_vec))
print(len(test_true_paired_feature_vec[0]))
print(len(test_true_paired_feature_vec[0][0]))
print((test_true_paired_feature_vec[0][0][0].shape))
test_true_pair_df.to_csv("test_false_pairs.csv", sep=',') # geneal info
test_true_pair_df_vec.to_csv("test_false_pairs_vec.csv", sep=',')


