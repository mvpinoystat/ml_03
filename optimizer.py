#!/home/pinoystat/Documents/python/environment/datascience/bin/python
from preprocessing import *

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier



from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
import sys 
import joblib

import scipy.stats as st
from itertools import combinations

#optuna
import optuna
import logging
import time


# Cross validation function
def cross_validation(model, X,y):
    print("[modeling status]: Starting Repeated Stratified K Fold")
    print("shape of X: {}".format(X.shape))
    scores = []
    
    risk = RepeatedStratifiedKFold(n_repeats = 2, n_splits =4 )
    cv = risk.split(X,y)
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

def optimizeCAT(trial):
    print("Optimizing CatBoost Classifier")
    #Note :Use scale_pos_weight = 3.72592422897397 # the ratio between n_negative / n_positive in train dataset

    iterations = trial.suggest_int('iterations',1000,2000)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    subsample = trial.suggest_float('subsample', 0.10,1)
    colsample_bylevel = trial.suggest_float('colsample_bylevel', 0.10,1)
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 1,100)

    
    model = CatBoostClassifier(iterations = iterations, learning_rate = learning_rate, depth = max_depth,
                               subsample = subsample, colsample_bylevel = colsample_bylevel,
                               min_data_in_leaf = min_data_in_leaf,scale_pos_weight =3.72592422897397,
                               random_state = seed, verbose = 0) 
    
     
    return cross_validation(model, train_processed, y)

def optimizeLGBM(trial):
    print("Optimizing LightGBM")
    #Note :Use scale_pos_weight = 3.72592422897397 # the ratio between n_negative / n_positive in train dataset

    lgbm_params = {'n_estimators':trial.suggest_int('n_estimators', 1000,2000),
                  'learning_rate' : trial.suggest_float('learning_rate', 0.001, 0.1, log = True),
                  'num_leaves': trial.suggest_int('num_leaves', 2, 2**10),
                  'bagging_freq' :1, 'subsample': trial.suggest_float('subsample', 0.05,1),
                  'colsample_bytree': trial.suggest_float('colsample_bylevel', 0.05,1),
                  'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1,100),
                  'verbosity':-1, }
    model = LGBMClassifier(**lgbm_params)
     
    return cross_validation(model, train_processed, y)

def optimizeXGB(trial):
    # Optimization results:
    # 0.8884459103842713
    # {'learn_rate': 0.08273439310845147, 'max_depth': 4, 'sub_sample': 0.7056046376538588, 'eta': 0.05373853565091731}
    print("Optimizing XGBoost Classifier")
    #Note :Use scale_pos_weight = 3.72592422897397 # the ratio between n_negative / n_positive in train dataset

    min_child_weight = trial.suggest_int('min_child_weight', 1,20)
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    gamma = trial.suggest_float('gamma', 0,5)
    subsample = trial.suggest_float('subsample', 0.5,1)
    eta = trial.suggest_float('eta', 0.01,0.5)
    colsample_bytree =trial.suggest_float('colsample_bytree',  0.4, 0.9)
    
    
    
    
    model = xgb.XGBClassifier(seed = seed, objective = 'binary:logistic',eval_metric = "auc",
                              min_child_weight = min_child_weight, n_estimators = n_estimators, learning_rate = learning_rate, 
                             max_depth = max_depth, gamma = gamma, eta = eta, colsample_bytree = colsample_bytree,
                             scale_pos_weight = 3.72592422897397)
     
    return cross_validation(model, train_processed, y)



def optimizeVotingWeights(trial):
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
    model = VotingClassifier(estimators = [('model_1', model_1), ('model_2', model_2), 
                                           ('model_3', model_3), ('model_4', model_4)], 
                             voting = 'soft', n_jobs = -1, 
                            weights = [trial.suggest_int('wt_model_1', 1,8), 
                                       trial.suggest_int('wt_model_2', 1,8), 
                                       trial.suggest_int('wt_model_3', 1,8), 
                                       trial.suggest_int('wt_model_4', 1,8) ]
                            )
    # weight result = [1,1, 8, 2]
    return cross_validation(model, train_processed, y)
    

def optimizeTree(trial):
    print("Optimizing Decision Tree Classifier")

    model = DecisionTreeClassifier(criterion = trial.suggest_categorical("criterion", ['gini', 'log_loss']),
                                   max_depth = trial.suggest_int('max_depth', 1,50),
                                   min_samples_split = trial.suggest_int('min_samples_split',2,40 ),
                                   min_samples_leaf = trial.suggest_int('min_samples_leaf',1,20),
                                   max_features = trial.suggest_categorical('max_features', ['sqrt','log2']),
                                   min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 0, 0.5),
                                   random_state = seed
                                  )
                              
    return cross_validation(model, train_processed, y)

def optimizeBaggingTree(trial):
    print("Optimizing Bagging Classifier on Decision Tree")
    tree_params = {'criterion': 'gini', 'max_depth': 13, 'min_samples_split': 20, 
                   'min_samples_leaf': 19, 'max_features': 'sqrt', 'min_weight_fraction_leaf': 0.0005376505605196474}

    base_model = DecisionTreeClassifier(**tree_params)

    # The Bagging classifier
    model = BaggingClassifier(n_estimators = trial.suggest_int("n_etimators", 10,20),
                              max_samples = trial.suggest_float("max_samples", 0.30,1),
                              random_state = seed,n_jobs = -1)
                              
    return cross_validation(model, train_processed, y)
    

def optimizeHistGradient(trial):
    print("Optimizing HistGradientClassifier")
    # Note: Optimized parameter:
    # 0.9271139087146164
    # {'learning_rate': 0.07630809870711368, 'max_iter': 181, 'max_leaf_nodes': 20, 'max_depth': 47, 'min_samples_leaf': 50}

    model = HistGradientBoostingClassifier(
        loss = 'log_loss', learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1),
        max_iter = trial.suggest_int('max_iter', 50, 200),
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 20,60),
        max_depth = trial.suggest_int('max_depth', 6,50),
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 20, 50),
        class_weight = {0:1, 1:3.72592422897397}
    )
                              
    return cross_validation(model, train_processed, y)


def optimizeAdaBoost(trial):
    print("Optimizing AdaBoostClassifier with Tree as base")
    tree_params = {'criterion': 'gini', 'max_depth': 13, 'min_samples_split': 20, 
                   'min_samples_leaf': 19, 'max_features': 'sqrt', 'min_weight_fraction_leaf': 0.0005376505605196474}

    base_model = DecisionTreeClassifier(**tree_params)
    #ada boost params:
    # ada_params = {'learning_rate':0.0382276386033583, 'n_estimators': 50} 

    model = AdaBoostClassifier(estimator = base_model, 
                               n_estimators = trial.suggest_int('n_estimators', 20, 50),
                               learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1),
                               random_state = seed)
                              
    return cross_validation(model, train_processed, y)


def optimizedByOptuna():
    if(sys.argv[1] == 'lgbm'):
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        study_name = "lgbm_study"  # Unique identifier of the study.
        storage_name = "sqlite:///{}.db".format(study_name)
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction = 'maximize',
                                   load_if_exists = True)
        
        myFunc = optimizeLGBM
        
    if(sys.argv[1] == 'cat'):
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        study_name = "cat_study"  # Unique identifier of the study.
        storage_name = "sqlite:///{}.db".format(study_name)
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction = 'maximize',
                                   load_if_exists = True)
        myFunc =optimizeCAT
    
    if(sys.argv[1] == 'xgb'):
        
        study_name = "xgb_study"  # Unique identifier of the study.
        storage_name = "sqlite:///{}.db".format(study_name)
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction = 'maximize',
                                   load_if_exists = True)
    
        myFunc = optimizeXGB
    
    if(sys.argv[1] == 'weight'):
        
        study_name = "weight_study"  # Unique identifier of the study.
        storage_name = "sqlite:///{}.db".format(study_name)
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction = 'maximize',
                                   load_if_exists = True)
    
        myFunc = optimizeVotingWeights
    if(sys.argv[1] == 'bagging_tree'):
        
        study_name = "bagging_tree"  # Unique identifier of the study.
        storage_name = "sqlite:///{}.db".format(study_name)
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction = 'maximize',
                                   load_if_exists = True)
    
        myFunc = optimizeBaggingTree
    
    if(sys.argv[1] == 'tree'):
        
        study_name = "decision_tree_study"  # Unique identifier of the study.
        storage_name = "sqlite:///{}.db".format(study_name)
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction = 'maximize',
                                   load_if_exists = True)
    
        myFunc = optimizeTree
    
    if(sys.argv[1] == 'hist_gradient'):
        
        study_name = "histgradient_study"  # Unique identifier of the study.
        storage_name = "sqlite:///{}.db".format(study_name)
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction = 'maximize',
                                   load_if_exists = True)
    
        myFunc = optimizeHistGradient
    
    if(sys.argv[1] == 'ada_boost'):
        
        study_name = "ada_boost_study"  # Unique identifier of the study.
        storage_name = "sqlite:///{}.db".format(study_name)
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction = 'maximize',
                                   load_if_exists = True)
    
        myFunc = optimizeAdaBoost
        
    study.optimize(myFunc, n_trials = 30)
    print("Optimization results:")
    print(study.best_value)
    print(study.best_params)




if __name__ == '__main__':
    seed = 11
    target_algo = sys.argv[1]
    
    print("[modeling status]: optimizing {}.".format(target_algo))
    
    print("[modeling status]: Started Loading pre-processed features.")
    train_processed = joblib.load("train_processed.pkl")
    test_processed = joblib.load("test_processed.pkl")
    train = pd.read_csv("train.csv")
    y = train.Exited.copy()
    del train

    optimizedByOptuna()
    
    print("[modeling status]: Optimization script completed")