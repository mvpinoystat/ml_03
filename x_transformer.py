#!/home/pinoystat/Documents/python/environment/datascience/bin/python

from preprocessing import *
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer
from sklearn.impute import SimpleImputer

import sys 
import joblib

# Pipelines:
def make_pipes():
    print("[modeling status]: Defining the pipelines.")
    # categorical data are Geography and Gender
    # Numerical data are Age, Tenure, Balance,NumOfProducts,HasCrCard, IsActiveMember, EstimatedSalary
    # getting list of the numerical and categorical columns
    
    # num = X.select_dtypes(include=['int64', 'float64']).columns
    # col = X.select_dtypes(include=['object']).columns
    
    # cat_list =['Geography', 'Gender'] 
    # num_list =['Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'CreditScore']
    # numerical_pipeline = Pipeline(steps = [('si',SimpleImputer(strategy = "median")),('ss',StandardScaler())])
    # categorical_pipeline = Pipeline(steps = [('imputer', SimpleImputer(strategy='most_frequent')),
    #                                          ('hot', OneHotEncoder())])
    # ct = ColumnTransformer(transformers = [('num', numerical_pipeline, num_list),('cat', categorical_pipeline, cat_list)],
    #                        remainder = "drop")
    # # included the CustomerID now
    # ctsurid = ColumnTransformer(transformers = [('te', TargetEncoder(), ["Surname","CustomerId"])], remainder = "drop")
    # tx_list = [
        # 
        # ('combo_catnum1',ComboCatNumeric(cat_columns = ['Geography','Gender'],num_columns = ['Age', 'IsActiveMember'],n_clusters = 5)),
    # ]
    
    # remove highly correlated columns:
    # combo_pipe = Pipeline(steps =[('drop_duplicates',DropDuplicates()), 
    #                               ('Fe', FeatureUnion(transformer_list = tx_list))]) 

    tx_list = [('Age',AgeTransformer()),#OK
               ('tenure', TenureTransformer()),#OK
               ('balance', BalanceTransformer()),#OK
               ('salary', SalaryTransformer()),#OK
               ('GeoGraphy', GeographyTransformer()),#OK
               ('HasCrCard', HasCrCardTransformer()),#OK
               ('NumOfProducts', NumOfProductsTransformer()),#Very OK
               ('Gender', GenderTransformer()),#OK
               ('Surname', SurnameTransformer()),#OK
               ('Id', CustomerIdTransformer()),#OK
               ('IsActiveMember', IsActiveMemberTransformer()),#OK
               # ('Credit', CreditScoreTransformer()),#OK
               # The above has CV of 0.9273244
               # ('FeatureEngineering', SimpleTransformer())
              ]
    print("[modeling status]: Completed the pipelines.")
    return  Pipeline([('fe', FeatureUnion(transformer_list = tx_list)), ('sp',SplineTransformer())])


if __name__ == '__main__':
    
    #preprocess X
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    X = train.drop('Exited', axis = 1).copy()
    y = train.Exited.copy()
    del train
    final_pipe = make_pipes() 
    print("[modeling status]: On-going Fit-transform on train set.")
    final_pipe.fit(X,y)
    train_processed = final_pipe.transform(X)
    print("[modeling status]: On-going transformation on test set.")
    test_processed = final_pipe.transform(test)
    print("[modeling status]: Preprocessing: Fit-transform on train and test set completed")
    joblib.dump(train_processed, "train_processed.pkl")
    joblib.dump(test_processed,"test_processed.pkl")
    print("[modeling status]: Saving processed train and test set on disk is complete.")


