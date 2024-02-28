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
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

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


    # model = StackingClassifier( estimators = [('model_1', model_1), ('model_2', model_2), 
    #                                           ('model_4', model_4)], final_estimator = model_3,
    #                            n_jobs = -1)


    model = ComboModel()
    
    
    
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


class ComboModel(BaseEstimator, TransformerMixin):
    def __init__(self, n_models = 8):
        self.xgb_params = {'min_child_weight': 14, 'n_estimators': 908, 'learning_rate': 0.0404153299899635, 
                  'max_depth': 6, 'gamma': 3.68287903858741, 'subsample': 0.814134730061779, 'eta': 0.326592004072874,
                  'colsample_bytree':0.523114265345016, 'scale_pos_weight' : 3.72592422897397}


        self.n_models = n_models
        self.models= []
        for i in range(n_models):
            self.models.append(xgb.XGBClassifier(**self.xgb_params,random_state = seed))

        self.skf = StratifiedKFold(n_splits = n_models, random_state = 42, shuffle = True)


    def fit(self, X, y):
        y = np.array(y)
        for i, (train_idx, test_idx) in enumerate(self.skf.split(X,y)):
            self.models[i].fit(X[train_idx], y[train_idx])

        return self

    def predict_proba(self, X):
        container = []
        for i in range(len(self.models)):
            if(i == 0):
                a = self.models[i].predict_proba(X)[:,1]
            else:
                b = self.models[i].predict_proba(X)[:,1]
                a = np.c_[a,b]
        weights = np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        c = np.dot(a, weights)
        
        c2 = np.c_[1-c, c]
        return c2
            
            
        
            
            
        
            
        
        
    
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
        
        
    
