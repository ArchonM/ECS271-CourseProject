# Read csv files

import pandas as pd
import json
import networkx as nx 
from networkx import json_graph 
from ast import literal_eval

csv_path = "../../ecsTest/ecs_project/csv_files/"
cfg_csv_filename = "lstm_cfg.csv"
test_pair_csv_filename = "test_pairs.csv"
train_pair_csv_filename = "train_pairs.csv"
csv_output_file = "final.csv"

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

# for i in range(0, df_train_pairs_shape[0]) :
#     list_train_true_pairs.extend(literal_eval(df_train_pairs.loc[i]['true_pair'])) # Contains the list of ids of true_pairs (from test dataset)

#     list_train_false_pairs.extend(literal_eval(df_train_pairs.loc[i]['false_pair'])) # Contains the list of ids of false_pairs (from test dataset)

print("list_train_true_pairs len : ", len(list_train_true_pairs))
print("list_train_true_pairs len : ", len(list_train_true_pairs))

# Just to test
ele = list_test_true_pairs[3][0]
print(ele)
val = df_cfg.loc[df_cfg['id'] == ele]['lstm_cfg']
print(val)
print(type(val))
print(len((val)))
lstm_cfg_feature = val.iloc[len(val)-1]
print("###########")
j_load = json.loads(lstm_cfg_feature)
print(j_load)
print(type(j_load))
print("********")

# Actual logic
paired_features = [] # contains list of mnemonic features from two ids in pair
for pairs in list_test_true_pairs :
    pair_feature_list = []
    for ids in pairs :
        id_feature = []
        val = df_cfg.loc[df_cfg['id'] == ids]['lstm_cfg']
        lstm_cfg_feature_dict = val.iloc[len(val)-1]
        for nodes in lstm_cfg_feature_dict[nodes] :
            if nodes[features] : 
                print(nodes[features])
                id_feature.extend(nodes[features])
                print(id_feature)
        pair_feature_list.append(id_feature)
        print(pair_feature_list)
    paired_features.append(pair_feature_list)
    
print(paired_features)
    



