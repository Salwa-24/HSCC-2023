"""
HSCC-2023 k-Means Clustering

Description:
This file conatins the code for performing the k-Means clustering on our repective three cases datasets. 
Along with this we even plot QQ-plot and perform KS-Test to test the normality of the datasets.

Developer(s):    Salwa Sayeedul Hasan                                     Date: 14/02/2022
                 Mohamed Fazil Hussain
"""

# Importing the required libraries

from cProfile import label
from dateutil.parser import parse 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.stats import kurtosis, skew
import statistics
from sklearn.cluster._kmeans import KMeans
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as PathEffects
import math
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import statsmodels.api as sm
from scipy.stats import norm

#class for reading the data in the CSV file
class CSV_Data():
    #its constructor
    def __init__(self,datafilepath):
        #pandas library used for reading a csv file into a DataFrame
        self.df =  pd.read_csv(datafilepath) 
      #  self.df2 = pd.read_csv(datafilepath) 
    
    #method named return_df
    def return_df(self):
        return self.df 
        #when called returns  self.df 

#class for Kmeans
class Kmeans:
    """Class that creates individual, combined and cluster graphs and prints stats of datasets from given csv file
    """
    #its constructor
    def __init__(self, df,index_col_name):
        """init function

        Args:
            datafilepath (str): csv file path of datasets/n
            index_col_name (str): index column name.
        """
        #pandas library used for reading a csv file into a DataFrame
        self.df = df


    #method named plot_kmeans_clusters with its arguments
    def plot_kmeans_clusters(self, datasetscolumns, colors=[], title="", xlabel='', ylabel='', clustertype=''):
        """plots cluster graphs and prints k-mean distance between cluster centers of gven dataset columns

        Args:
            datasetscolumns (list): array of dataset column names from csv file
            colors (list, optional): array of color strings. Defaults to [].
            title (str, optional): title for the graph. Defaults to "".
            xlabel (str, optional): lable for x-axis. Defaults to ''.
            ylabel (str, optional): lable for y-axis. Defaults to ''.
        """

        #subplots() used to position the figure in a specified grid
        fig, ax = plt.subplots(figsize=(11, 6))
        scatter_zero = self.df[datasetscolumns].values
        if (clustertype == "kmeans"):
            model_zero = KMeans(n_clusters=2)
            model_zero.fit(scatter_zero)

            # cmap = ListedColormap(["blue", "purple", "green"])
            #cmap is a colormap instance
            cmap=ListedColormap(colors)

            #plt.scatter is used to observe relationship between variables and dots are used to represent the relationship between them 
            plt.scatter(scatter_zero[:, 0], scatter_zero[:, 1],c=model_zero.labels_, cmap=cmap)

            # plot the cluster centers
            txts = []
            cluster_points = []
            print(str.join(" vs ", datasetscolumns))
            for ind, pos in enumerate(model_zero.cluster_centers_):
                print("Cluster Point:", pos)
                cluster_points.append(pos)
                txt = ax.text(pos[0], pos[1], 'cluster %i /n (%.3f,%.3f)' %
                        (ind, pos[0], pos[1]), fontsize=12, zorder=100)
                txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="aquamarine"), PathEffects.Normal()])
                txts.append(txt)
            #  sqrt (  (x2-x1)^2 +  (y2-y1)^2)

            #by using 2 cluster points , distance is calculated
            point1 = cluster_points[0]
            point2 = cluster_points[1]
            x_com = (point2[0] - point1[0])
            x_com *= x_com
            y_com = point2[1] - point1[1]
            y_com *= y_com
            print( x_com,y_com)
            distance = math.sqrt((x_com + y_com))
            print("Distance between K-mean centers: ", round(distance, 4))
            zero_mean = np.mean(model_zero.cluster_centers_, axis=0)
            txt = ax.text(zero_mean[0], zero_mean[1], 'point set zero', fontsize=15)
            txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="violet"), PathEffects.Normal()])
            txts.append(txt)
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            plt.title(title, fontsize=17, fontweight='bold')
            #showing which points belong in cluster 0 and cluster 1
            y_predicted = model_zero.fit_predict(self.df[:])
            print(y_predicted)
            plt.show()


#method named qq_plot() with its arguments
def qq_plot(datafilepath,colname,title):
    
        data= pd.read_csv(datafilepath)
        datalist =  data[colname].tolist()
        #print(datalist)
        dataarray =  np.array(datalist)
        #print(dataarray)

        np.random.seed(0)
        data = np.random.normal(0,1, 1000)
        #print(type(data))
        #print(datalist)

        fig = sm.qqplot(dataarray, line='45')
        plt.title(title, fontsize=17, fontweight='bold')
        plt.show()

#method named distribution_fitting to call path of csv file and perform distribution fitting
def distribution_fitting(datafilepath,colname,title): 
    mCSV_Data = CSV_Data(datafilepath)
    df = mCSV_Data.return_df()
    
    #parameters obtained for the fitting
    dist = getattr(stats, 'norm')
    parameters = dist.fit(df[colname])
    #print(parameters)
	
    stats.kstest(df['idle_delta'], "norm", parameters)
    #stats.kstest(df['internal_delta'], "norm", parameters)
	#stats.kstest(df['internalhard_delta'], "norm", parameters)

        
#Method named title_name with parameters which has the filepath , title and col names 
def title_data(datafilepath,title,col1,col2):
    
    mCSV_Data = CSV_Data(datafilepath)
    modified_df = mCSV_Data.return_df()
    mKmeans = Kmeans(modified_df,'No.')
    mKmeans.plot_kmeans_clusters([col1,col2],colors=['green','blue'], title=title, xlabel="k-mean center", ylabel="k-mean cluster", clustertype="kmeans")


def main():

  #Method instances are created and arguments are passed respectively  
   
   mtitle_data=title_data("datasets/HSCC-2023-rawdata.csv","Control vs Minimal Scan with Outlier Data",'idle_delta', 'internal_delta' )
   mtitle_data=title_data("datasets/HSCC-2023-Case1.csv","Control vs Minimal Scan without Outlier Data",'idle_delta', 'internal_delta')

   mtitle_data=title_data("datasets/HSCC-2023-rawdata.csv","Control vs HardEnd Scan with Outlier Data",'idle_delta', 'internalhard_delta' )
   mtitle_data=title_data("datasets/HSCC-2023-Case2.csv","Control vs HardEnd Scan without Outlier Data",'idle_delta', 'internalhard_delta')
   
   mtitle_data=title_data("datasets/HSCC-2023-rawdata.csv","Minimal vs HardEnd Scan with Outlier Data",'internal_delta', 'internalhard_delta' )
   mtitle_data=title_data("datasets/HSCC-2023-Case3.csv","Minimal vs HardEnd Scan without Outlier Data", 'internal_delta', 'internalhard_delta')


   mqq_plot=qq_plot('datasets/HSCC-2023-rawdata.csv','idle_delta','Control_Policy_Data_(case1)')
   mqq_plot=qq_plot('datasets/HSCC-2023-rawdata.csv','internal_delta','Minimal_Policy_Data_(case2)')
   mqq_plot=qq_plot('datasets/HSCC-2023-rawdata.csv','internalhard_delta','HardEnd_Policy_Data_(case3)')

   #Method instances are created and arguments are passed respectively  
   mdistribution_fitting=distribution_fitting("datasets/HSCC-2023-Case1.csv",'idle_delta','idle_delta original')
   #mdistribution_fitting=distribution_fitting("datasets/HSCC-2023-Case2.csv", 'internal_delta','internal_delta original')
   #mdistribution_fitting=distribution_fitting("datasets/HSCC-2023-Case3.csv", 'internalhard_delta','internalhard_delta original')
   
    
# the main program starts here, where we call the main routine
if (__name__ =='__main__'):
    main()
