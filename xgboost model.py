import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from skopt import BayesSearchCV
# pip install scikit-optimize

X_train = pd.read_pickle('X_train.pkl')
y_train = pd.read_pickle('y_train.pkl')
X_test = pd.read_pickle('X_test.pkl')
y_test = pd.read_pickle('y_test.pkl')


# flatten features
def getfeatures(data):
    feature1 = data.sum(axis=1).reshape(data.shape[0], 10000)
    feature2 = (data[:, 0, :, :] - data[:, 1, :, :]).reshape(data.shape[0], 10000)
    feature = np.hstack([feature1, feature2])
    return feature


X_train_flat = getfeatures(X_train)
X_test_flat = getfeatures(X_test)


# randomly permute data points
np.random.seed(1)
inds = np.random.permutation(X_train_flat.shape[0])
X_train_flat = X_train_flat[inds]
y_train = y_train[inds]

# model
xgb_model = XGBClassifier()

params = {
    'n_estimators': [100, 300, 500],
    'colsample_bytree': [0.9, 0.8, 0.7],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 5, 6, 7],
    'min_child_weight': [1, 2, 3, 4],
    'gamma': [0.1, 0.2, 0.3],
    'subsample': [0.8, 0.7, 0.6],
    'reg_lambda': [0.05, 0.1, 1],
    'reg_alpha': [0.05, 0.1, 1]
}

xgb_opt = BayesSearchCV(estimator=xgb_model, search_spaces=params,
                        scoring='accuracy', random_state=0, n_jobs=-1,
                        cv=3, n_iter=5)

xgb_opt.fit(X_train_flat, y_train)

# report the best result
print(xgb_opt.best_score_)
print(xgb_opt.best_params_)
xgb_opt.best_params_.to_csv("xgb_params.csv")

y_pred = xgb_opt.predict(X_test)
y_pred.to_csv("y_pred_xgb.csv")
