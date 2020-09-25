# 공사지표 300m기준
import numbers
import math
from openpyxl import load_workbook
import pandas as pd
import numpy as np
from pandas import DataFrame

class GeoUtil: # x좌표 y좌표 km로 변환해서 x1 y1 x2 y2 차이 구하기
    """
    Geographical Utils
    """
    @staticmethod
    def degree2radius(degree):
        return degree * (math.pi / 180)

    @staticmethod
    def get_harversion_distance(x1, y1, x2, y2, round_decimal_digits=5):
        """
        경위도 (x1,y1)과 (x2,y2) 점의 거리를 반환
        Harversion Formula 이용하여 2개의 경위도간 거래를 구함(단위:Km)
        """
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return None

        R = 6371  # 지구의 반경(단위: km)
        dLon = GeoUtil.degree2radius(x2 - x1)
        dLat = GeoUtil.degree2radius(y2 - y1)

        a = math.sin(dLat / 2) * math.sin(dLat / 2) \
            + (math.cos(GeoUtil.degree2radius(y1)) \
               * math.cos(GeoUtil.degree2radius(y2)) \
               * math.sin(dLon / 2) * math.sin(dLon / 2))
        b = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return round(R * b, round_decimal_digits)


# 서울시청 126.97843, 37.56668
# 강남역   127.02758, 37.49794

#print(GeoUtil.get_harversion_distance(126.97843, 37.56668, 127.02758, 37.49794)) => 떨어진 거리가 나옴 ( 8.7(km) )

location=pd.read_csv('최종.csv', encoding='euc_kr') # 전주시 학교주소_도로명+한국시설안전공단.csv
building=pd.read_csv('최종.csv', encoding='euc-kr') # 건물지표_xy좌표.csv
people=pd.read_csv('최종.csv', encoding='euc-kr') # 인적지표_xy좌표.csv
property=pd.read_csv('재산지표_최종.csv', encoding='euc-kr') # 재산지표_xy좌표.csv
indirect=pd.read_csv('간접지표_최종.csv', encoding='euc-kr') # 간접지표_xy좌표.csv
construction=pd.read_csv('공사지표_최종.csv', encoding='euc-kr') # 공사지표_xy좌표.csv
disaster=pd.read_csv('재난지표_최종.csv', encoding='euc-kr') # 재난지표_xy좌표.csv


print((location.head(5)))
print(building.head(5))
print(people.head(5))
print(property.head(5))
print(indirect.head(5))
print(construction.head(5))
print(disaster.head(5))

lo_x = location['Latitude'] # (학교)건물 x좌표
lo_y = location['Longitude'] # (학교)건물 y좌표
building_x = building['Latitude'] # 간접지표 x좌표
building_y = building['Longitude'] # 간접지표 y좌표
people_x = people['Latitude'] # 간접지표 x좌표
people_y = people['Longitude'] # 간접지표 y좌표
indirect_x = indirect['Latitude'] # 간접지표 x좌표
indirect_y = indirect['Longitude'] # 간접지표 y좌표
construction_x = construction['Latitude'] # 간접지표 x좌표
construction_y = construction['Longitude'] # 간접지표 y좌표
disaster_x = disaster['Latitude'] # 간접지표 x좌표
disaster_y = disaster['Longitude'] # 간접지표 y좌표
property_x = property['Latitude'] # 간접지표 x좌표
property_y = property['Longitude'] # 간접지표 y좌표

#print(x[270], y[270])

#for i in range(0, len(location)):
#    print(GeoUtil.get_harversion_distance(lo_x[i], lo_y[i], 127.02758, 37.49794))

people_cnt=[] # 150m 반경내 간접지표 개수
indirect_cnt=[] # 150m 반경내 간접지표 개수
construction_cnt=[] # 150m 반경내 간접지표 개수
disaster_cnt=[] # 150m 반경내 간접지표 개수
property_cnt=[] # 150m 반경내 간접지표 개수

for k in range(len(location)): #cnt 초기화
    people_cnt.append(0)
    indirect_cnt.append(0)
    construction_cnt.append(0)
    disaster_cnt.append(0)
    property_cnt.append(0)

print(location)

data_lo =[]
for i in range(30001, 35001):
    print( i,"번째")
    for j in range(0, len(people)):
        if (GeoUtil.get_harversion_distance(lo_x[i], lo_y[i], people_x[j], people_y[j]))<=0.15: #0.15km 내
            people_cnt[i]+=people['총합'][i]
    for j in range(0, len(indirect)):
        if (GeoUtil.get_harversion_distance(lo_x[i], lo_y[i], indirect_x[j], indirect_y[j]))<=0.15: #0.15km 내
            indirect_cnt[i]+=1
    for j in range(0, len(construction)):
        if (GeoUtil.get_harversion_distance(lo_x[i], lo_y[i], construction_x[j], construction_y[j]))<=0.15: #0.15km 내
            construction_cnt[i]+=1
    for j in range(0, len(disaster)):
        if (GeoUtil.get_harversion_distance(lo_x[i], lo_y[i], disaster_x[j], disaster_y[j]))<=0.15: #0.15km 내
            disaster_cnt[i]+=1
    for j in range(0, len(property)):
        if (GeoUtil.get_harversion_distance(lo_x[i], lo_y[i], property_x[j], property_y[j]))<=0.15: #0.15km 내
            property_cnt[i]+=property['공시지가'][i]

    #print("cnt[",i,"]= ",cnt[i])

    #location['간접지표'].append(cnt[i])

#print(location)

location.loc[:, 'people_scope'] = pd.Series(people_cnt, index=location.index)
location.loc[:, 'property_scope'] = pd.Series(property_cnt, index=location.index)
location.loc[:, 'indirect_scope'] = pd.Series(indirect_cnt, index=location.index)
location.loc[:, 'construction_scope'] = pd.Series(construction_cnt, index=location.index)
location.loc[:, 'disaster_scope'] = pd.Series(disaster_cnt, index=location.index)

location.to_csv("30000_35001_scope.csv",encoding='ms949')

print("ok")



