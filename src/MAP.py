import pandas as pd
import numpy as np
import math
from pandas import DataFrame
import folium
import base64
import json
from folium.plugins import MarkerCluster

result = pd.read_csv('for_map.csv',encoding='euc-kr')
name = pd.read_csv('건물등급_최종.csv',encoding='euc-kr')

print(result.head(5))


map_osm = folium.Map(location=[result['Latitude'][0],	result['Longitude'][0]],zoom_start=13)
mcg = folium.plugins.MarkerCluster(control = False)
map_osm.add_child(mcg)
g1 = folium.plugins.FeatureGroupSubGroup(mcg,'1등급')
map_osm.add_child(g1)
g2 = folium.plugins.FeatureGroupSubGroup(mcg,'2등급')
map_osm.add_child(g2)
g3 = folium.plugins.FeatureGroupSubGroup(mcg,'3등급')
map_osm.add_child(g3)
g4 = folium.plugins.FeatureGroupSubGroup(mcg,'4등급')
map_osm.add_child(g4)
g5 = folium.plugins.FeatureGroupSubGroup(mcg,'5등급')
map_osm.add_child(g5)

group1 =['1','2','3','4']
group2 = ['5','6','7','8']
group3 = ['9']

for i in range(len(result)):
   if result['clustering_AC'][i]==0:
    print("군집그룹 1")
    folium.Marker([result['Latitude'][i],result['Longitude'][i]],
                  popup=name['학교명'][i],icon=folium.Icon(icon='red')).add_to(g1)

   elif result['clustering_KM'][i] == 2:
    print("군집그룹 2")
    folium.Marker([result['Latitude'][i], result['Longitude'][i]],
                  popup=name['학교명'][i], icon=folium.Icon(color='red')).add_to(g2)

   elif result['clustering_KM'][i] == 3:
    print("군집그룹 3")
    folium.Marker([result['Latitude'][i], result['Longitude'][i]],
                  popup=name['학교명'][i], icon=folium.Icon(color='blue')).add_to(g3)
   elif result['clustering_KM'][i] == 4:
    print("군집그룹 4")
    folium.Marker([result['Latitude'][i], result['Longitude'][i]],
                  popup=name['학교명'][i], icon=folium.Icon(color='red')).add_to(g4)
   else :
    folium.Marker([result['Latitude'][i], result['Longitude'][i]],
                     popup=name['학교명'][i], icon=folium.Icon(color='blue')).add_to(g5)



folium.LayerControl(collapsed=False).add_to(map_osm)

map_osm.save('./map5.html')
print("ok")