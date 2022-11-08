#importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats
from scipy.stats import kstest

#class for reading the data in the CSV file
class CSV_Data():
    #its constructor
    def __init__(self,datafilepath):
        #pandas library used for reading a csv file into a DataFrame
        self.df =  pd.read_csv(datafilepath) 
    
    #method named return_df
    def return_df(self):
        return self.df 
        #when called returns  self.df 


#method named distribution_fitting to call path of csv file and perform distribution fitting
def distribution_fitting(datafilepath,col,title): 
    mCSV_Data = CSV_Data(datafilepath)
    df = mCSV_Data.return_df()
    
    #parameters obtained for the fitting
    dist = getattr(stats, 'norm')
    parameters = dist.fit(df[col])
    #print(parameters)
	
    print(stats.kstest(df['idle_delta'], "norm", parameters))
    #print(stats.kstest(df['internal_delta'], "norm", parameters))
	#print(stats.kstest(df['internalhard_delta'], "norm", parameters))

def main():
    #Method instances are created and arguments are passed respectively  
    mdistribution_fitting=distribution_fitting("datasets/HSCC-2023-Case1.csv",'idle_delta','idle_delta original')
    #mdistribution_fitting=distribution_fitting("datasets/HSCC-2023-Case2.csv", 'internal_delta','internal_delta original')
    #mdistribution_fitting=distribution_fitting("datasets/HSCC-2023-Case3.csv", 'internalhard_delta','internalhard_delta original')
    

# the main program starts here, where we call the main routine
if (__name__ =='__main__'):
    main()