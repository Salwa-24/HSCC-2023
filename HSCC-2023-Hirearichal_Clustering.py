"""
HSCC-2023 Hirearichal Clustering

Description:
This code runs the hierarchichal clustering on our collected raw dataset, and gives insights about the outliers. 


Developer(s):    Salwa Sayeedul Hasan                                     Date: 14/08/2022
                 Mohamed Fazil Hussain
                 
"""

# Importing the required libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

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


#another class named Hierarchical 
class Hierarchical:
    #with its constructor
    def __init__(self, df2,index_col_name):
        """init function

        Args:
            datafilepath (str): csv file path of datasets\n
            index_col_name (str): index column name.
        """
        self.df = df2

    #formal method called hierarchical_clustering created
    def plot_hierarchical_clustering(self,datasetscolumns, casee ):
        clustertype=''
       # dataset = self.df[datasetscolumns].values
        X= self.df[datasetscolumns].values
        if (clustertype == "hierarchical"):
            model_zero = AgglomerativeClustering()
            model_zero.fit(X)
        #print(X)

        dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
        plt.title(casee)
        #plt.scatter(X[:,:], X[:,:])
        plt.scatter(X[:,0], X[:,1])

        #axhline() is used to add horizontal line across the axis.
        plt.axhline(y=0.0575, color='r', linestyle='--')       
        #plt show() looks for all currently active figure objects , and open the interactive windows and displays the figure    
        plt.show()
        
        #using agglomerative clustering i.e. individual data points clustered together, 
        # with 2 number of clusters
        model = AgglomerativeClustering(affinity='euclidean', linkage='ward')

        #fit_predict() method is used to fit and perform predictions over data 
        model.fit_predict(X)
        labels=model.fit_predict(X) 
        #print(labels)


def main():
    # m is added as prefix to give the instance name to the particular class instance
    mCSV_Data = CSV_Data("datasets/HSCC-2023-rawdata.csv")

     # instance method mCSV_Data returns the  df2
    csv_300upload_df2 = mCSV_Data.return_df()

    #instancing the class Hierarchical by passing the csv_300upload_df2
    mHierarchical = Hierarchical(csv_300upload_df2,'No.') 

    #Method instances are created and arguments are passed respectively
    mHierarchical.plot_hierarchical_clustering(['idle_delta', 'internal_delta'], 'Control vs Minimal Scans')
    mHierarchical.plot_hierarchical_clustering(['idle_delta', 'internalhard_delta'], 'Control vs HardEnd Scans')
    mHierarchical.plot_hierarchical_clustering(['internalhard_delta', 'internal_delta'], 'Minimal vs HardEnd Scans')

    
# the main program starts here, where we call the main routine
if (__name__ =='__main__'):
    main()






 



