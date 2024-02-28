#!/home/pinoystat/Documents/python/environment/datascience/bin/python
from preprocessing import *

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
import sys 
import joblib

import scipy.stats as st
from itertools import combinations

#optuna
import optuna

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
X = train.drop("Exited", axis = 1).copy()
y = train.Exited.copy()


#Add a seed for random state:
seed = 11


# Cross validation function
def cross_validation(model, X,y):
    print("status: Starting Stratified Shuffle Split CV")
    print("shape of X: {}".format(X.shape))
    scores = []
    
    splitter = StratifiedShuffleSplit(n_splits = 5,test_size = 0.5, random_state = seed)
    for i, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
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
    

# Pipelines:

#Make pipes which uses Kmean to learn from data
def makeComboPipes():
    # categorical data are Geography and Gender
    # Numerical data are Age, Tenure, Balance,NumOfProducts,HasCrCard, IsActiveMember, EstimatedSalary
    cat_list =['Geography', 'Gender', 'Surname'] 
    num_list =['Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'CreditScore']
    combo_num_list = list(combinations(num_list, 3))
    pipe_list = []
    for cat in cat_list:
        for num in combo_num_list:
            num_name = num[0]+num[1]+num[2]
            pipe_list.append(("{}_{}".format(cat,num_name),
                              make_pipeline(ComboCatNumeric(cat_columns = [cat], num_columns = list(num), n_clusters = 6))))
    return pipe_list
    
tx_list = [
    ('new_features', make_pipeline(FeatureCreation1())),
    ('new_features2', make_pipeline(FeatureCreation2())),
    ('geo', make_pipeline(TargetTransformer(columnName = "Geography"))),#Good
    ('gender', make_pipeline(TargetTransformer(columnName = "Gender"))),#Good
    ('age', make_pipeline(PassThrough("Age"))),#Good. Already Optimized
    ('active', make_pipeline(TargetTransformer(columnName = "IsActiveMember"))),#Good
    ('products',make_pipeline(PassThrough(columnName = "NumOfProducts"))),#Very Good, high increase in score
    # Use below the Surname but with the combos as well.
    ('Surname', make_pipeline(TargetTransformer(columnName = "Surname"))),#Good
    
    # the combos:
    ('combo_num2', ComboNumeric(column_names = ['HasCrCard', 'NumOfProducts','CreditScore'], n_clusters = 4)),# Small but ok. Good at 4 clusters
    ('combo_num3', ComboNumeric(column_names = ['Age', 'NumOfProducts', 'Tenure'], n_clusters = 4 )),# Very small. Good at n_clusters = 4 
    ('combo_num4', ComboNumeric(column_names = ['CreditScore', 'NumOfProducts','Balance'], n_clusters = 8)),# .88728 at 8 
    #Below combo catnum1 5 clusters gives 0.887475
    ('combo_catnum1',ComboCatNumeric(cat_columns = ['Geography', 'Gender'],num_columns = ['Age', 'IsActiveMember'],n_clusters = 5)),
    #Pass through
    ('credit_score', make_pipeline(PassThrough("CreditScore"))),# Now ok using xgboost !!!
    ('tenure', make_pipeline(PassThrough("Tenure"))),# Now ok using xgboost !!!
    #Additional cookery:
    ('est_salary', make_pipeline(SalaryTransformer())),# Ok now with xgboost
]

# Append the new ones:
# only combo features:
tx_list = []
new = makeComboPipes()
for i in new:
    tx_list.append(i)


combo_pipe = FeatureUnion(transformer_list = tx_list)

final_pipe = Pipeline([('combo', combo_pipe)])


print("status: Pipeline connections completed.")

#For checking of individual feature
def featureCheck():
    # print("Checking correlation of the features")
    # results = pd.DataFrame(train_processed).corr() 
    # print(results)
    
    model = xgb.XGBClassifier(random_state= seed, n_estimators = 20)
    # model = BaggingClassifier(estimator = base_model, n_estimators = 50, max_samples = 0.3, 
    #                            random_state = seed, n_jobs = -1)
    # model = BaggingClassifier(n_estimators = 50, random_state = seed)
    cross_validation(model, train_processed, y)

    
def executeModeling():
    # Optimization results for GBC: 
    # 0.8884940463823603
    # {'learning_rate': 0.07818096030027812, 'subsample': 0.7037395929141526, 'max_depth': 8, 'n_estimators': 268}
    model1 = GradientBoostingClassifier(n_estimators = 268,learning_rate= 0.07818096030027812,
                                        subsample = 0.7037395929141526, max_depth= 8,random_state = seed)
    # XGBOOST
    # Optimization results:
    # 0.8886433163450297
    # {'learn_rate': 0.060536825278028394, 'max_depth': 6, 'sub_sample': 0.3283313513985284, 'eta': 0.12062817949364188}
    #test base:
    # {'learn_rate': 0.08273439310845147, 'max_depth': 4, 'sub_sample': 0.7056046376538588, 'eta': 0.05373853565091731}
    model2 = xgb.XGBClassifier(seed = seed, learning_rate = 0.08273439310845147, max_depth = 4, subsample = 0.7056046376538588,
                               eta =0.05373853565091731 , objective = 'binary:logistic')


    # Optimization results:
    # 0.8869376310837369
    # {'n_estimators': 295, 'criterion': 'gini', 'max_depth': 10}
    # From public notebook:
    model 3 = RandomForestClassifier(**{'n_estimators': 425, 'max_depth': 13, 'min_samples_split': 41, 'min_samples_leaf': 2, 
                                        'max_features': 'sqrt', 'bootstrap': False, 'criterion': 'entropy', 'random_state': 42})
    # model3 = RandomForestClassifier(n_estimators = 295, criterion = "gini", max_depth = 10, random_state = seed) 

    # The voting classifier
    # model = VotingClassifier(estimators = [('model1', model1), ('model2', model2),('model3', model3)], voting = 'soft')

    # Optimization results: XGB + Bagging
    # 0.8846404593786978
    # {'learn_rate': 0.05914792458107898, 'max_depth': 5, 'sub_sample': 0.38486008137857625, 
    # 'eta': 0.17736276347643476, 'n_estimators': 90, 'max_samples': 0.5960329016991794}
    # 
    # Using a bagging classifier: 
    # base_model = xgb.XGBClassifier(random_state= seed, n_estimators = 20)
    # model3 = BaggingClassifier(n_estimators = 50, random_state = seed)

    # The voting classifier
    model = VotingClassifier(estimators = [('model1', model1), ('model2', model2), ('model3', model3)], voting = 'soft',
                            n_jobs = -1)
    
    cross_validation(model, train_processed, y)
    # Submit predictions:
    print("status: Model fitting for submission.")
    model.fit(train_processed,y)
    print("status: Model fitting completed")
    
    predictions = model.predict_proba(test_processed)[:,1]
    submission = pd.DataFrame()
    submission['id'] = test.id 
    submission['Exited'] = predictions 
    print("Submission:")
    print(submission.head(3))
    submission.to_csv('submission.csv', index = False)
    
    

    

# Optuna Optimization


def optimizeForest(trial):
    print("Optimizing RandomForest")
    ns = trial.suggest_int('n_estimators', 50, 300)
    criterion = trial.suggest_categorical('criterion', ['gini', 'log_loss', 'entropy'])
    md = trial.suggest_int("max_depth", 3, 10)
    
    model = RandomForestClassifier(n_estimators = ns, criterion = criterion, max_depth = md, random_state = seed, n_jobs = -1) 
    
    return cross_validation(model, train_processed, y)
 
    
def optimizeGBC(trial):
    print("Optimizing GradientBoostingClassifier")
    lr = trial.suggest_float("learning_rate", 0.00001, .1)
    ss = trial.suggest_float("subsample", 0.10, 1)
    md = trial.suggest_int("max_depth", 3,10),
    ns = trial.suggest_int("n_estimators", 10, 300)
    
    model = GradientBoostingClassifier(loss = "log_loss",learning_rate = lr, subsample = ss, 
                          n_estimators = ns, random_state = seed) 
    return cross_validation(model, train_processed, y)
    
def optimizeXGB(trial):
    # Optimization results:
    # 0.8884459103842713
    # {'learn_rate': 0.08273439310845147, 'max_depth': 4, 'sub_sample': 0.7056046376538588, 'eta': 0.05373853565091731}
    print("Optimizing XGBoost Classifier")

    lr = trial.suggest_float('learn_rate', 0.001, 0.1)
    md = trial.suggest_int('max_depth', 3, 10)
    ss = trial.suggest_float('sub_sample', 0.2,1)
    et = trial.suggest_float('eta', 0.01,0.2)
    n_est = trial.suggest_int('n_estimators', 10, 100)
    ms = trial.suggest_float('max_samples', 0.10, 1)
    
    base_model = xgb.XGBClassifier(seed = seed, objective = 'binary:logistic',
                              learning_rate = lr, max_depth = md, subsample = ss, eval_metric = "auc",
                              eta = et, n_estimators = n_est )
     
    model = BaggingClassifier(estimator = base_model, max_samples = ms, n_estimators = 20, random_state = seed, n_jobs = -1)
    return cross_validation(model, train_processed, y)
# ***********************************


    
def optimizedByOptuna():
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction = "maximize", sampler=sampler)
    myFunc = optimizeXGB
    if(sys.argv[2] == 'gbc'):
        myFunc = optimizeGBC
    if(sys.argv[2] == 'forest'):
        myFunc = optimizeForest
    if(sys.argv[2] == 'xgb'):
        myFunc = optimizeXGB
    study.optimize(myFunc, n_trials = 40)
    print("Optimization results:")
    print(study.best_value)
    print(study.best_params)
    
    
# Execution:
if(int(sys.argv[1]) == 1):
    #preprocess X
    print("status: On-going Fit-transform on train and test set.")
    train_processed = final_pipe.fit_transform(X,y)
    test_processed = final_pipe.transform(test)
    print("status: Preprocessing: Fit-transform on train and test set completed")
    joblib.dump(train_processed, "train_processed.pkl")
    joblib.dump(test_processed,"test_processed.pkl")
    executeModeling()
    
if(int(sys.argv[1]) == 0):
    print("status: Started Loading pre-processed features.")
    train_processed = joblib.load("train_processed.pkl")
    test_processed = joblib.load("test_processed.pkl")
    print("status: Loading pre-processed features is complete.")
    executeModeling()
# Optuna optimization:
if(int(sys.argv[1]) == 2):
    # Optuna optimized:
    print("status: Started Loading pre-processed features.")
    train_processed = joblib.load("train_processed.pkl")
    test_processed = joblib.load("test_processed.pkl")
    print("status: Loading pre-processed features is complete.")
    optimizedByOptuna()

if(int(sys.argv[1]) == 3):
    #Feature Checkup:
    #preprocess X
    print("status: On-going Fit-transform on train and test set.")
    train_processed = final_pipe.fit_transform(X,y)
    test_processed = final_pipe.transform(test)
    print("status: This checks the individual feature contribution to roc_auc_score.")
    print("status: Preprocessing: Fit-transform on train and test set completed")
    joblib.dump(train_processed, "train_processed.pkl")
    joblib.dump(test_processed,"test_processed.pkl")
    featureCheck()
    
     
    
            
            


