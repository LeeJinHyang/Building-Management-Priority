# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from pandas import DataFrame as df

def change_building_grade(binning,building_binning_data):
    building_binning =[0 for i in range(len(building_binning_data))]
    for i in range(len(binning)):
        #print("i는", i, "  ", building_binning_data['등급'][i])
        if building_binning_data['등급'][i] == "A등급":
            building_binning[i] = 12
        elif building_binning_data['등급'][i] == "B등급":
            building_binning[i] = 14
        elif building_binning_data['등급'][i] == "C등급":
            building_binning[i] = 16
        elif building_binning_data['등급'][i] == "D등급":
            building_binning[i] = 18
        elif building_binning_data['등급'][i] == "E등급":
            building_binning[i] = 20
        elif building_binning_data['등급'][i] == "불명":
            building_binning[i] = 17
        else:
            building_binning[i] = 0

    return building_binning

binning =pd.read_csv("total_scope.csv",encoding='euc-kr')
building_binning_data = pd.read_csv("건물등급_최종.csv",encoding='euc-kr')
building_binning= change_building_grade(binning,building_binning_data)# 건물지표_xy좌표.csv
people_binning=binning['people_scope'] # 인적지표_xy좌표.csv
property_binning=binning['property_scope'] # 재산지표_xy좌표.csv
indirect_binning=binning['indirect_scope'] # 간접지표_xy좌표.csv
construction_binning=binning['construction_scope']# 공사지표_xy좌표.csv
disaster_binning=binning['disaster_scope'] # 재난지표_xy좌표.csv
result=pd.read_csv('데이터_최종.csv', encoding='euc_kr')

print(binning.head(5))
print(len(binning))
#max = np.max(binning['간접지표'])
#total = np.mean(binning['인구총합'])*len(binning)

building_binning_pre = sorted(building_binning)
people_binning_pre = sorted(binning['people_scope'])
property_binning_pre = sorted(binning['property_scope'])
indirect_binning_pre = sorted(binning['indirect_scope'])
construction_binning_pre = sorted(binning['construction_scope'])
disaster_binning_pre = sorted(binning['disaster_scope'])
people_max = np.max(binning['people_scope'])
property_max = np.max(binning['property_scope'])
print("people : ",people_max)
print("property : ",property_max)
people_bins = [people_binning_pre[0]-1,people_binning_pre[len(people_binning_pre)//10*2 ],people_binning_pre[len(people_binning_pre)//10*3 ],people_binning_pre[len(people_binning_pre)//10*4 ],people_binning_pre[len(people_binning_pre)//10*5 ],people_binning_pre[len(people_binning_pre)//10*6 ],people_binning_pre[len(people_binning_pre)//10*7 ],people_binning_pre[len(people_binning_pre)//10*8 ],people_binning_pre[len(people_binning_pre)//10*9 ],people_binning_pre[len(people_binning_pre)//10*10],people_max+1]
property_bins = [property_binning_pre[0]-1,property_binning_pre[len(property_binning_pre)//10*2 ],property_binning_pre[len(property_binning_pre)//10*3 ],property_binning_pre[len(property_binning_pre)//10*4 ],property_binning_pre[len(property_binning_pre)//10*5 ],property_binning_pre[len(property_binning_pre)//10*6 ],property_binning_pre[len(property_binning_pre)//10*7 ],property_binning_pre[len(property_binning_pre)//10*8 ],property_binning_pre[len(property_binning_pre)//10*9 ],property_binning_pre[len(property_binning_pre)//10*10],property_max+1]
indirect_bins = [-1,indirect_binning_pre[len(indirect_binning_pre)//10],1,indirect_binning_pre[len(indirect_binning_pre)//10*5 ],indirect_binning_pre[len(indirect_binning_pre)//10*6 ],indirect_binning_pre[len(indirect_binning_pre)//10*7 ],indirect_binning_pre[len(indirect_binning_pre)//10*8 ]]
construction_bins = [-1,0,1,2]
disaster_bins = [-1,0,1]

people_labels = [0,1,2,3,4,5,6,7,8,9]
property_labels = [0,1,2,3,4,5,6,7,8,9]
indirect_labels = [0,2,4,6,8,10]
construct_labels = [0,5,10]
disaster_labels = [0,10]

"""people_binning_pre = sorted(binning['people_scope'])
property_binning_pre = sorted(binning['property_scope'])
indirect_binning_pre = sorted(binning['indirect_scope'])
construction_binning_pre = sorted(binning['construction_scope'])
disaster_binning_pre = sorted(binning['disaster_scope'])
"""

result.loc[:, 'building_grade'] = pd.Series(building_binning, index=result.index)
result['people_grade'] = pd.cut(binning['people_scope'],bins=people_bins,labels=people_labels)
result['property_grade'] = pd.cut(binning['property_scope'],bins=property_bins,labels=property_labels)
result['indirect_grade'] = pd.cut(binning['indirect_scope'],bins=indirect_bins,labels=indirect_labels)
result['construction_grade'] = pd.cut(binning['construction_scope'],bins=indirect_bins,labels=indirect_labels)
result['disaster_grade'] = pd.cut(binning['disaster_scope'],bins=disaster_bins,labels=disaster_labels)

result.to_csv('total_binning.csv',encoding ='ms949')

forgrade_ = df(
    data={'building_grade':result['building_grade'],
    'people_grade':result['people_grade'],
    'property_grade':result['property_grade'],
    'indirect_grade':result['indirect_grade'],
    'construction_grade' : result['construction_grade'],
    'disaster_grade' : result['disaster_grade']       },
    columns = ['building_grade','people_grade','property_grade','indirect_grade','construction_grade','disaster_grade']
)

forgrade_.to_csv('for_grade.csv',encoding='ms949')

print("ok")

