#!/home/pinoystat/Documents/python/environment/datascience/bin/python
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, TargetEncoder, StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

'''
Age Transformer is OK now
'''
class AgeTransformer(BaseEstimator, TransformerMixin):

    def fit(self,X,y):
        return self

    def ageDivider(self, x):
        code = 0
        if x >= 20:
            code = 1
        if x >= 25:
            code = 2
        if x >= 30:
            code = 3
        if x >= 35:
            code = 4
        if x >= 40:
            code = 5
        if x >= 45:
            code = 6
        if x >= 50:
            code = 7
        if x >= 55:
            code = 8
        if x >= 60:
            code = 9
        return code


    def transform(self,X):
        #put the z score:
        age = X.Age.apply(lambda s: self.ageDivider(s))
        combo = np.c_[age, X.Age]
        return combo 
        


# Transformer for CreditScore
# This transformer is optimized 
# Note: This lowerse the auc_roc_score
class CreditScoreTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.targetEncoder = TargetEncoder(target_type = 'binary', random_state = 11)

    def fit(self, X, y):
        a = X.CreditScore.apply(lambda f: self.creditDivider(f))
        b = np.array(a).reshape(-1,1)
        c = self.targetEncoder.fit(b,y)
        return self

    def transform(self,X):
        a = X.CreditScore.apply(lambda f: self.creditDivider(f))
        b = np.array(a).reshape(-1,1)
        return self.targetEncoder.transform(b) 

    def creditDivider(self, x):
        code = 0
        if x > 450:
            code = 1
        if x > 500:
            code = 2
        if x > 550:
            code = 3
        if x > 600:
            code = 4
        if x > 650:
            code = 5
        if x > 700:
            code = 6
        if x > 750:
            code = 7
        if x > 800:
            code = 8
        


# This transformer as low contrib overall
class TenureTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    def transform(self,X):
        return np.array(X.Tenure).reshape(-1,1)


class BalanceTransformer(BaseEstimator, TransformerMixin):
    
    def fit(self, X,y):
        return self

    def transform(self, X):
        a = np.array(X.Balance.apply(lambda r : 0 if(r < 1) else 1)).reshape(-1,1)
        # b = np.array(X.Balance).reshape(-1,1)
        return np.c_[a] 

    


# Optimized Transformer for Estimated Salary
# Do not use. This currently lowers the score
class SalaryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.targetEncoder = TargetEncoder(target_type = 'binary', random_state = 11)
        self.bins = [0,22231.68, 44451.78, 66671.88, 88891.98,111112.08, 133332.18,
                     155552.28, 177772.38, 200000]
        self.standardScaler = StandardScaler()
    def fit(self, X, y):
        a = pd.cut(X.EstimatedSalary, bins = self.bins)
        b = np.array(a).reshape(-1,1)
        self.targetEncoder.fit(b, y)
        self.standardScaler.fit(np.array(X.EstimatedSalary).reshape(-1,1))
        return self

    def transform(self, X):
        a = pd.cut(X.EstimatedSalary, bins = self.bins)
        b = np.array(a).reshape(-1,1)
        # b1 =self.targetEncoder.transform(b)
        b2 = self.standardScaler.transform(np.array(X.EstimatedSalary).reshape(-1,1))
        return b2 


class GeographyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.targetEncoder = TargetEncoder(target_type = 'binary', random_state = 11)
        self.labelEncoder = LabelEncoder()
        
    def fit(self, X, y):
        self.targetEncoder.fit(np.array(X.Geography).reshape(-1,1), y)
        self.labelEncoder.fit(np.array(X.Geography))
        return self

    def transform(self, X):
        a = self.targetEncoder.transform(np.array(X.Geography).reshape(-1,1))
        b = self.labelEncoder.transform(np.array(X.Geography))
        return a 
        

class HasCrCardTransformer(BaseEstimator, TransformerMixin):
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #a = self.targetEncoder.transform(np.array(X.HasCrCard).reshape(-1,1))
        # return np.c_[a,np.array(X.HasCrCard).reshape(-1,1)]
        return np.array(X.HasCrCard).reshape(-1,1)
        
class NumOfProductsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.targetEncoder = TargetEncoder(target_type = 'binary', random_state = 11)
        
    def fit(self, X, y):
        self.targetEncoder.fit(np.array(X.NumOfProducts).reshape(-1,1), y)
        return self

    def transform(self, X):
        a = self.targetEncoder.transform(np.array(X.NumOfProducts).reshape(-1,1))
        # return np.c_[a,np.array(X.NumOfProducts).reshape(-1,1))
        return np.array(X.NumOfProducts).reshape(-1,1)
        
class GenderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.targetEncoder = TargetEncoder(target_type = 'binary', random_state = 11)
        self.labelEncoder = LabelEncoder()
        
    def fit(self, X, y):
        self.targetEncoder.fit(np.array(X.Gender).reshape(-1,1), y)
        self.labelEncoder.fit(X.Gender)
        return self

    def transform(self, X):
        # a = self.targetEncoder.transform(np.array(X.Gender).reshape(-1,1))
        b = self.labelEncoder.transform(X.Gender)
        return b.reshape(-1,1) 



class SurnameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.targetEncoder = TargetEncoder(target_type = 'binary', random_state = 11)
        self.labelEncoder = LabelEncoder()
        
    def fit(self, X, y):
        self.targetEncoder.fit(np.array(X.Surname).reshape(-1,1), y)
        a = X.Surname.apply(lambda q: q[0])
        self.labelEncoder.fit(a)
        return self

    def transform(self, X):
        a = self.targetEncoder.transform(np.array(X.Surname).reshape(-1,1))
        b = X.Surname.apply(lambda q: q[0])
        c = self.labelEncoder.transform(b)
        return np.c_[a,c.reshape(-1,1)]
        
class CustomerIdTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.targetEncoder = TargetEncoder(target_type = 'binary', random_state = 11)
        
    def fit(self, X, y):
        self.targetEncoder.fit(np.array(X.CustomerId).reshape(-1,1), y)
        return self

    def transform(self, X):
        a = self.targetEncoder.transform(np.array(X.CustomerId).reshape(-1,1))
        return a 

class IsActiveMemberTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.targetEncoder = TargetEncoder(target_type = 'binary', random_state = 11)
        self.labelEncoder = LabelEncoder()
        
    def fit(self, X, y):
        self.targetEncoder.fit(np.array(X.IsActiveMember).reshape(-1,1), y)
        self.labelEncoder.fit(X.IsActiveMember)
        return self

    def transform(self, X):
        a = self.targetEncoder.transform(np.array(X.IsActiveMember).reshape(-1,1))
        b = self.labelEncoder.transform(X.IsActiveMember)
        # # return np.c_[a,b.reshape(-1,1), np.array(X.IsActiveMember).reshape(-1,1)]
        return np.array(X.IsActiveMember).reshape(-1,1)

# combination of categorical and Numeric data:
class ComboCatNumeric(BaseEstimator, TransformerMixin):
    def __init__(self, cat_columns = [], num_columns = [], n_clusters =6, random_state=11):
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.standardScaler = StandardScaler()
        self.oneHotEncoder = OneHotEncoder(sparse_output =False)
        self.kmeans = KMeans(n_clusters = n_clusters, n_init = 10, random_state = self.random_state, max_iter = 50)
        self.targetEncoder = TargetEncoder(target_type = 'binary', random_state = self.random_state)

    def fit(self, X, y):
        a1 = []
        a2 = []
        if(len(self.cat_columns) > 0):
            self.oneHotEncoder.fit(X[self.cat_columns])
            a1 = self.oneHotEncoder.transform(X[self.cat_columns])
        if(len(self.num_columns) > 0):
            a2 = np.array(X[self.num_columns])
            a2 = self.standardScaler.fit_transform(a2)
        if(len(a1) > 0 and len(a2) > 0):
            b1 = np.c_[a1,a2]
        if(len(a1) > 0 and len(a2) < 1):
            b1 = a1
        if(len(a1) < 1 and len(a2) > 0):
            b1 = a2
            
        self.kmeans.fit(b1)
        return self

    def transform(self, X):
        a1 = []
        a2 = []
        if(len(self.cat_columns) > 0):
            a1 = self.oneHotEncoder.transform(X[self.cat_columns])
        if(len(self.num_columns) > 0):
            a2 = np.array(X[self.num_columns])
            a2 = self.standardScaler.transform(a2)
        if(len(a1) > 0 and len(a2) > 0):
            b1 = np.c_[a1,a2]
        if(len(a1) > 0 and len(a2) < 1):
            b1 = a1
        if(len(a1) < 1 and len(a2) > 0):
            b1 = a2
            
        b2 = self.kmeans.predict(b1)
        return b2.reshape(-1,1)

# *************************************

        


class DropDuplicates(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop_duplicates()
    

class RemoveColumn(BaseEstimator, TransformerMixin):
    def __init__(self, column_number):
        self.column_number = column_number
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        a = np.delete(arr = X, obj = self.column_number, axis = 1)
        return a

    
    
class SimpleTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.sc = StandardScaler()
        pass

    def fit(self, X, y):
        involvement = np.array(X.Tenure * (X.NumOfProducts + X.HasCrCard)).reshape(-1,1) 
        self.sc.fit(involvement)
        return self

    def transform(self,X):
        involvement = np.array(X.Tenure * (X.NumOfProducts + X.HasCrCard)).reshape(-1,1) 
        a = self.sc.transform(involvement)
        return involvement 

