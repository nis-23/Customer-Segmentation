

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
import seaborn as sns
import warnings



class ProcessData():
    
    
    def __init__(self, customer_data):
        
        self.customer_data = customer_data
        
    def drop_total_relationship_count(self, customer_data):
        
        customer_data.drop('total_relationship_count', axis = 1, inplace=True)
        
        return customer_data
    
    def add_avg_transaction_value(self, customer_data):
        
        customer_data['avg_transaction_value'] = customer_data['total_trans_amount']/customer_data.total_trans_count
        customer_data.drop(['total_trans_amount','total_trans_count'], axis = 1, inplace = True)
        customer_data.avg_transaction_value = np.log(customer_data.avg_transaction_value)
        
        return customer_data
    
    def label_gender(self, customer_data):
        
        from sklearn.preprocessing import   LabelEncoder
        label_encoder = LabelEncoder()
        customer_data['gender'] = label_encoder.fit_transform(customer_data.gender)
        
        return customer_data
    
    def transform_marital_status(self, customer_data):
        
        customer_data['Married'] = customer_data.marital_status.apply(lambda x: 1 if x == "Married" else 0)
        customer_data.drop('marital_status', axis=1, inplace=True)
        
        return customer_data 
    
    
    def transform_dependant_count(self, customer_data):
        
        customer_data['> 2 Dependants'] = customer_data.dependent_count.apply(lambda x : 1 if x > 2 else 0 )
        customer_data.drop('dependent_count', axis=1, inplace=True)
        
        return customer_data
    
    
    def transform_education_level(self, customer_data):
        
        el_map = {
        "Uneducated": 1,
        "High School": 2,
        "College": 3,
        "Graduate": 4,
        "Post-Graduate": 5, 
        "Doctorate": 6
                    }

        customer_data.education_level = customer_data['education_level'].map(el_map)
        customer_data.education_level.value_counts().sort_index()

        # 
        customer_data.education_level = customer_data.education_level.apply(lambda x: 1 if x > 3 else 0 )
        
        return customer_data
    
    
    def transform_mob(self, customer_data):
        
        customer_data['MOB > 3Y'] = customer_data.months_on_book.apply(lambda x: 1 if x > customer_data.months_on_book.quantile(.50) else 0) 
        customer_data.drop('months_on_book', axis = 1, inplace= True)
        
        return customer_data
    
    
    def transform_months_inactive(self, customer_data):
        
        customer_data[" Inactive > 2 Months"] =  customer_data.months_inactive_12_mon.apply( lambda x  : 1 if x > 3 else 0 )
        customer_data.drop('months_inactive_12_mon', axis=1, inplace= True)
        
        return customer_data
    
     
    
    
    def main(self):
        
        from functools import reduce 
        
        obj = ProcessData(self.customer_data)
        function_list = [getattr(obj, method) for method in dir(obj) if callable(getattr(obj, method)) and not method.startswith("__") and method != 'main']

        
        processed_data = reduce(lambda data, func: func(data), function_list, self.customer_data)
        
        return processed_data
    
        
        
        
        
        

        
    
        
         



def standardize_data(customer_data_processed):
    
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(customer_data_processed)
    X = pd.DataFrame(X_scaled)
    
    return X 