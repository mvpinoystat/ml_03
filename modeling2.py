#!/home/pinoystat/Documents/python/environment/datascience/bin/python
from preprocessing import *

from sklearn.ensemble import BaggingClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier



from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import sys 
import joblib

import scipy.stats as st
from itertools import combinations

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
X = train.drop("Exited", axis = 1).copy()
y = train.Exited.copy()

del train


#Add a seed for random state:
seed = 11


# Cross validation function
def cross_validation(model, X,y):
    print("[modeling status]: Starting Stratified Shuffle Split CV")
    print("shape of X: {}".format(X.shape))
    scores = []
    
    # splitter = StratifiedShuffleSplit(n_splits = 5,test_size = 0.5, random_state = seed)
    # for i, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
    #     model.fit(X[train_idx], y[train_idx])
    #     score_a = roc_auc_score(y[test_idx], model.predict_proba(X[test_idx])[:,1])
    #     print("Split {}: roc_auc_score: {}".format(i, score_a))
    #     scores.append(score_a)

    risk = RepeatedStratifiedKFold(n_repeats = 2, n_splits = 4, random_state = seed)
    for i, (train_idx, test_idx) in enumerate(risk.split(X, y)):
        model.fit(X[train_idx], y[train_idx])
        score_a = roc_auc_score(y[test_idx], model.predict_proba(X[test_idx])[:,1])
        print("Split {}: roc_auc_score: {}".format(i, score_a))
        scores.append(score_a)
    
    print("ROC_AUC_SCORE:")
    print("mean   : {}".format(np.mean(scores)))
    print("sd     : {}".format(np.std(scores, ddof = 1)))
    print("minimum: {}".format(np.min(scores)))
    print("maximum: {}".format(np.max(scores)))
    return np.mean(scores)
    

def executeModeling(cross_val = 1):

    
    # cat_params ={'iterations': 1553, 'learning_rate': 0.03599865703788584, 'max_depth': 4, 
    #                                   'subsample': 0.3578020123439845, 'colsample_bylevel': 0.3230349174459384, 
    #                                   'min_data_in_leaf': 69, 'scale_pos_weight' :3.72592422897397, 
    #                                  'random_state' : seed, 'verbose': 0} 

    #old optimized:
    # cat_params = {'iterations': 830, 'learning_rate': 0.08238714339235984, 'depth': 5,
    #               'l2_leaf_reg': 0.8106903985997884, 'random_state': 42, 'verbose': 0 }
    # 
    # model_cat = CatBoostClassifier(**cat_params)

    #from old optimized:
    # model_cat = CatBoostClassifier(**{'iterations': 830, 'learning_rate': 0.08238714339235984, 'depth': 5,
    #                              'l2_leaf_reg': 0.8106903985997884, 'random_state': 42, 'verbose': 0}) 

    # lgbm_params = {'n_estimators':1343, 'learning_rate' :0.00455180128034201, 
    #               'num_leaves': 74, 'bagging_freq' :1, 'subsample': 0.952508138936973,
    #               'colsample_bytree':0.608434289915229,
    #               'min_data_in_leaf':9,'verbosity':-1, }
    # 
    # lgbm_params = {'n_estimators': 960, 'learning_rate': 0.031725771326186744, 'max_depth': 8, 'min_child_samples': 8, 
    #                'subsample': 0.7458307885861184, 'colsample_bytree': 0.5111460378911089, 'random_state': 42, 'verbose' : -1}
    # model_lgbm = LGBMClassifier(**lgbm_params)
    #from old optimized:
    # model_lgbm = LGBMClassifier(**{'n_estimators': 960, 'learning_rate': 0.031725771326186744, 'max_depth': 8, 'min_child_samples': 8, 
    #                            'subsample': 0.7458307885861184, 'colsample_bytree': 0.5111460378911089, 'random_state': 42, 'verbose' : -1})

    
    
    

    # The voting classifier
    # model = VotingClassifier(estimators = [('model_cat', model_cat), ('model_lgbm', model_lgbm), ('model_xgb', model_xgb)], 
    #                          voting = 'soft', n_jobs = -1, weights = [6,7,4])
    
    # 20 n_estimators are ok
    # bagging_params = {'n_estimators' : 10, 'max_features':0.994442536958991, 'max_samples':0.681344396406371,
    #                   'random_state': seed} # LB 86755`
    # bagging_params = {'n_estimators' : 10, 'max_features':0.4, 'max_samples':0.6,
    #                   'random_state': seed} # LB 0.87198
    # bagging_params = {'n_estimators' : 50, 'max_features':0.8, 'max_samples':0.3,
    #                   'random_state': seed} # LB 0.8554
    # bagging_params = {'n_estimators' : 10, 'max_features':0.8, 'max_samples':0.3,
    #                   'random_state': seed} # LB 0.86572
    # bagging_params = {'n_estimators' : 10, 'max_features':0.4, 'max_samples':0.3,
    #                   'random_state': seed} # LB 0.87154
    # bagging_params = {'n_estimators' : 5, 'max_features':0.4, 'max_samples':0.6,
    #                   'random_state': seed} # 0.86657
    # bagging_params = {'n_estimators' : 20, 'max_features':1, 'max_samples':0.8,
    #                   'random_state': seed} 
    # model = BaggingClassifier(estimator = base_model, **bagging_params)

    #Note: Jan 24. This has a cv of about 0.015 higher than the LB score. This is the baseline model. XGBoost + Bagging. 
    #Modify now the features. 
    decision_tree_params = {'criterion': 'gini', 'max_depth': 13, 'min_samples_split': 20, 
                            'min_samples_leaf': 19, 'max_features': 'sqrt', 'min_weight_fraction_leaf': 0.0005376505605196474,
                           'random_state': seed}

    model_tree = DecisionTreeClassifier(**decision_tree_params)
    bagging_params = {'n_estimators' : 10, 'max_features':0.4, 'max_samples':0.6,
                      'random_state': seed} # LB 0.87198
    model_1 = BaggingClassifier(estimator = model_tree, **bagging_params)

    hist_gradient_params ={'learning_rate': 0.07630809870711368, 'max_iter': 181, 'max_leaf_nodes': 20,
                           'max_depth': 47, 'min_samples_leaf': 50, 'class_weight':{0:1, 1:3.72592422897397},
                          'random_state' : seed} 

    model_2 = HistGradientBoostingClassifier(**hist_gradient_params)

    xgb_params = {'min_child_weight': 14, 'n_estimators': 908, 'learning_rate': 0.0404153299899635, 
                  'max_depth': 6, 'gamma': 3.68287903858741, 'subsample': 0.814134730061779, 'eta': 0.326592004072874,
                  'colsample_bytree':0.523114265345016, 'scale_pos_weight' : 3.72592422897397}

    model_3= xgb.XGBClassifier(**xgb_params,random_state = seed)

    ada_params = {'learning_rate':0.0382276386033583, 'n_estimators': 50} 
    
    model_4 = AdaBoostClassifier(**ada_params, random_state = seed)
    # The voting classifier
    # model = VotingClassifier(estimators = [('model_1', model_1), ('model_2', model_2), 
    #                                        ('model_3', model_3), ('model_4', model_4)], 
    #                          voting = 'soft', n_jobs = -1)


    model = StackingClassifier( estimators = [('model_1', model_1), ('model_2', model_2), 
                                              ('model_4', model_4)], final_estimator = model_3,
                               n_jobs = -1)


    
    # ***********************
    
    

    if(cross_val == 1):
        cross_validation(model, train_processed, y)
        
    # Submit predictions:
    print("[modeling status]: Model fitting for submission.")
    model.fit(train_processed,y)
    print("[modeling status]: Model fitting completed")
    
    predictions = model.predict_proba(test_processed)[:,1]
    submission = pd.DataFrame()
    submission['id'] = test.id 
    submission['Exited'] = predictions 
    print("Submission:")
    print(submission.head(3))
    submission.to_csv('submission.csv', index = False)
    

def executeFeatureSearch():
    xgb_params = {'min_child_weight': 14, 'n_estimators': 908, 'learning_rate': 0.0404153299899635, 
                  'max_depth': 6, 'gamma': 3.68287903858741, 'subsample': 0.814134730061779, 'eta': 0.326592004072874,
                  'colsample_bytree':0.523114265345016, 'scale_pos_weight' : 3.72592422897397}

    model= xgb.XGBClassifier(**xgb_params,random_state = seed)
    print("Testing changes in features.")
    print("Baseline CV = 0.927324")
    cross_validation(model, train_processed, y)
    
    
# Execution:
if __name__ == "__main__":
    print("[modeling status]: Started Loading pre-processed features.")
    train_processed = joblib.load("train_processed.pkl")
    test_processed = joblib.load("test_processed.pkl")
    print("[modeling status]: Loading pre-processed features is complete.")
    do_crossvalidation = 0
    if(int(sys.argv[1]) == 1):
        do_crossvalidation = 1 
        executeModeling(do_crossvalidation)
    if(int(sys.argv[1]) == 0):
        do_crossvalidation = 0
        executeModeling(do_crossvalidation)
    if(int(sys.argv[1]) == 3):
        executeFeatureSearch()
        
        
    
