from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans,DBSCAN
from sklearn.mixture import GaussianMixture
# -*- coding: utf-8 -*-
#%matplotlib inline
import pandas as pd
import numpy as np
import math
from pandas import DataFrame
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from pandas import DataFrame as df

binning = pd.read_csv('for_grade.csv',encoding='euc-kr')
xy = pd.read_csv('total_binning.csv',encoding='euc-kr')
#binning = binning.reset_index()
#print(binning.dtypes)
#binning['인적등급'].astype('int')
print(binning.head(5))
print(binning.dropna())
model = KMeans(n_clusters= 5)#KMeans n의 수를 5로 하면 정확도 1, 그 외의 수로 하면 정확도 거의 0에 수렴
dbscan = DBSCAN(min_samples=3)#DBScan
acc = AgglomerativeClustering(n_clusters= 5)#계층적 군집화
gmm=GaussianMixture(n_components= 5,covariance_type="full",tol = 0.001)

print("           1")
model.fit(binning)
dbscan.fit(binning)
acc.fit(binning)
gmm.fit(binning)
print("2")
y_predict_KM = model.fit_predict(binning)
y_predict_DB = dbscan.fit_predict(binning)
y_predict_AC = acc.fit_predict(binning)
y_predict_GMM = gmm.fit_predict(binning)

print(y_predict_KM)
print(y_predict_DB)
print(y_predict_AC)
print(y_predict_GMM)
binning['clustering_KM'] = y_predict_KM
binning['clustering_DB'] = y_predict_DB
binning['clustering_AC'] = y_predict_AC
binning['clustering_GMM'] = y_predict_GMM
print(binning)
formap_ = df(
    data={'Latitude': xy['Latitude'],
          'Longitude': xy['Longitude'],
          'clustering_KM': binning['clustering_KM'],
          'clustering_DB': binning['clustering_DB'],
          'clustering_AC': binning['clustering_AC'],
          'clustering_GMM': binning['clustering_GMM']},
    columns=['Latitude','Longitude','clustering_KM','clustering_DB','clustering_AC','clustering_GMM']

)

formap_.to_csv('for_map.csv',encoding='ms949')

binning.to_csv('clustering_result.csv',encoding='ms949')
print("ok")
