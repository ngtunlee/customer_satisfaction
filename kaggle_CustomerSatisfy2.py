# remove everything: %reset -f   

##############################################################################
#                             Data PreProcessing                             #
##############################################################################

import pandas as pd
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# check
len(list(train))
train["TARGET"].value_counts()
list( set( train.columns ) - set( train._get_numeric_data().columns ) ) # non-numeric columns
train.columns[ train.shape[0] - train.count() != 0 ] # no missing data
test.columns[ test.shape[0] - test.count() != 0 ] # no missing data

# (train & test) remove ID column 
train.drop(["ID"], axis = 1, inplace = True)
test_ID = test["ID"]
test.drop(["ID"], axis = 1, inplace = True)

# (train & test) remove columns with one unique value
unique_num = train.apply(pd.Series.nunique)
cols_to_drop = unique_num[unique_num == 1].index
train.drop(cols_to_drop, axis=1, inplace = True)
test.drop(cols_to_drop, axis=1, inplace = True)
del unique_num, cols_to_drop

# (train & test) impute NA w/ median
## no missing data

# (train) remove duplicated rows 
train.drop_duplicates(inplace = True)

# (train) separate predictors & responses
y_train = train["TARGET"]
train.drop("TARGET", axis = 1, inplace = True)

# (train) remove highly correlated columns
import numpy as np
cor_thresh = 0.9
corr_matrix = train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [col for col in upper.columns if any(upper[col] > cor_thresh)]
train.drop(train[to_drop], axis=1, inplace = True)
del corr_matrix, upper, cor_thresh

# (test) remove columns + impute outliers w/ training's median 
test.drop(test[to_drop], axis=1, inplace = True)
del to_drop

# (train & test) standardize 
mean = train.mean(axis = 0) # return a vector of column (axis = 0) means
stdev = train.std(axis = 0) # return a vector of column (axis = 0) std dev
train -= mean
train /= stdev
test -= mean
test /= stdev
del mean, stdev

import gc
gc.collect()

##############################################################################
#                                  Random Forest                             #
##############################################################################

train = train.to_numpy()
y_train = y_train.to_numpy(dtype = "float32")
test = test.to_numpy()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

rf_mod = RandomForestClassifier(criterion = "gini", 
                                random_state = 42,
                                max_features = "auto",
                                # oob_score = True,
                                bootstrap = True)

# random forest parameters
params = {
    'n_estimators': [1000, 2000, 3000], # default 100 
    'min_samples_leaf': [3, 5, 10] # default 1
 }

# grid search cv parameters
grid_search = GridSearchCV(rf_mod, 
                           param_grid = params, 
                           scoring = "roc_auc",
                           n_jobs = 2,
                           refit = True,
                           cv = 3, 
                           verbose = 3, 
                           return_train_score = False)

grid_search.fit(train, y_train)
grid_search.best_estimator_ 
grid_search.best_score_  # OOB score = 0.82765

# use OOB score to compare models without CV
from itertools import product
params_arr = np.array( list( product( *params.values() ) ) )
rf_OOB = []
for prm in params_arr:
     rf_mod = RandomForestClassifier(criterion = "gini", 
                                random_state = 42,
                                max_features = "auto",
                                oob_score = True,
                                n_estimators = prm[0],
                                min_samples_leaf = prm[1],
                                bootstrap = True) 
     rf_mod.fit(train, y_train)
     rf_OOB.append(rf_mod.oob_score_)
params_arr[ rf_OOB.index( max(rf_OOB) ) ]
     

test_pred = grid_search.predict_proba(test)
test_pred = test_pred[ : , 1]
res = pd.concat( [ test_ID, pd.Series( test_pred ) ], axis = 1)
res.columns = ["ID", "TARGET"]
res.to_csv('rf_try1.csv', header = True, index = False)

##############################################################################
#                                   Neural Network                           #
##############################################################################

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop, adam
from sklearn.metrics import roc_auc_score

## far fewer instances of TARGET = 1
## use stratified sampling

train = pd.concat( [train, y_train], axis = 1)
from sklearn.model_selection import StratifiedShuffleSplit
## StratifiedShuffleSplit.split generate training and test set indices
split = StratifiedShuffleSplit(n_splits = 1,
                               test_size = 0.2, 
                               random_state = 42)
for train_ind, val_ind in split.split(train, train["TARGET"]):
    strat_train = train.iloc[train_ind]
    strat_val = train.iloc[val_ind]

train.drop("TARGET", axis = 1, inplace = True)
y_strat_train = strat_train["TARGET"].copy().to_numpy(dtype="float32")
y_strat_val = strat_val["TARGET"].copy().to_numpy(dtype="float32")
strat_train = strat_train.to_numpy()
strat_val = strat_val.to_numpy()
strat_train = np.delete(strat_train, -1, axis=1)
strat_val = np.delete(strat_val, -1, axis=1)

import gc
gc.collect()

# try shallow
mod = Sequential()
mod.add( Dense( 512, activation = 'relu', 
                       input_shape = (strat_train.shape[1],) ) )
mod.add( Dense( 1, activation = 'sigmoid' ) )
mod.compile(optimizer = 'rmsprop', 
            loss = 'binary_crossentropy',
            metrics = ['accuracy'])

hist = mod.fit(x = strat_train,
               y = y_strat_train,
               epochs = 20,
               batch_size = 1024,
               validation_data = (strat_val, y_strat_val))

hist_dict = hist.history #history object
loss_val = hist_dict['loss']
validate_loss_val = hist_dict['val_loss']

epochs = range(1, len(hist_dict['acc']) + 1)
import matplotlib.pyplot as plt
%config InlineBackend.close_figures=False # keep figures open in pyplot
plt.clf() # CLEAR previous figures!!!

# plot training and validation losses vs epochs
plt.plot(epochs, 
         loss_val, 
         'bo', # blue dot
          label = 'Training Loss')
plt.plot(epochs,
         validate_loss_val,
         'b', # blue solid line
         label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot training and validation accuracy vs epochs
plt.clf() # CLEAR previous figures!!!
acc_val = hist_dict['acc']
validate_acc_val = hist_dict['val_acc']

plt.plot(epochs, 
         acc_val, 
         'bo', # blue dot
         label = 'Training Accuracy')
plt.plot(epochs,
         validate_acc_val,
         'b', # blue solid line
         label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# use epoch = 5
mod = Sequential()
mod.add( Dense( 512, activation = 'relu', 
                       input_shape = (strat_train.shape[1],) ) )
mod.add( Dense( 1, activation = 'sigmoid' ) )
mod.compile(optimizer = 'rmsprop', 
            loss = 'binary_crossentropy',
            metrics = ['accuracy'])
mod.fit(train.to_numpy(), np.asarray(y_train).astype('float32'), 
        epochs = 5, batch_size = 1024)

test_pred = mod.predict(test.to_numpy())
res = pd.concat( [ test_ID, 
                  pd.Series( test_pred.reshape(test_pred.shape[0]) ) ], 
      axis = 1)
res.columns = ["ID", "TARGET"]
res.to_csv('nn_try5.csv', header = True, index = False)

##############################################################################
#                                         SVM                                #
##############################################################################

train = train.to_numpy()
y_train = y_train.to_numpy(dtype = "float32")
test = test.to_numpy()

from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

svm_mod = LinearSVC(random_state = 42, max_iter = 10000, tol = 0.001)

params = {'C': [1, 5, 10, 50, 100]}

grid_search = GridSearchCV(svm_mod, 
                           param_grid = params, 
                           scoring = "accuracy",
                           n_jobs = 2,
                           refit = True,
                           cv = 3, 
                           verbose = 3, 
                           return_train_score = False)

grid_search.fit(train, y_train)
grid_search.best_estimator_ 
grid_search.best_score_  # try1 score 0.9601

CCcv = CalibratedClassifierCV( base_estimator = LinearSVC(C = 1, 
                                                          dual=False,
                                                          max_iter = 10000, 
                                                          tol = 0.001),
                               method = 'isotonic', cv = 5)
CCcv.fit(train, y_train)
test_pred = CCcv.predict_proba(test)[ : , 1]
res = pd.concat( [ test_ID, 
                  pd.Series( test_pred.reshape(test_pred.shape[0]) ) ], 
      axis = 1)
res.columns = ["ID", "TARGET"]
res.to_csv('svm_try1.csv', header = True, index = False)

##############################################################################
#                                     Boosting                               #
##############################################################################

train = train.to_numpy()
y_train = y_train.to_numpy(dtype = "float32")
test = test.to_numpy()

# Extreme Gradient Boosting
from xgboost import XGBClassifier
from scipy.stats import uniform, randint
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

xgb_mod = XGBClassifier(objective="binary:logistic", 
                        booster='gbtree',
                        random_state=42, 
                        eval_metric="auc")

# xgboost parameters
params = {
    "gamma": uniform(0, 0.3), # default 0
    "learning_rate": uniform(0.1, 0.25), # default 0.1 
    "max_depth": randint(1, 4), # default 6
    "n_estimators": randint(100, 1000) # default 100
}

# randomized search cv parameters
rand_search = RandomizedSearchCV(xgb_mod, 
                                 param_distributions = params, 
                                 # scoring = "roc_auc",
                                 random_state = 42, 
                                 n_iter = 40, # Number of sampled parameters 
                                 cv = 3, 
                                 verbose = 1, 
                                 n_jobs = 2, 
                                 refit = True,
                                 return_train_score = False)

rand_search.fit(train, y_train)
rand_search.best_estimator_ 
rand_search.best_score_ 

test_pred = rand_search.predict_proba(test)
test_pred = test_pred[ : , 1]
res = pd.concat( [ test_ID, pd.Series( test_pred ) ], axis = 1)
res.columns = ["ID", "TARGET"]
res.to_csv('xgboost_try2.csv', header = True, index = False)
