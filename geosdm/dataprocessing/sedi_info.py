from glob import glob
import rasterio
import os
from rasterio.merge import merge
import rioxarray
import geopandas as gpd 
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from shapely.geometry import Point
from pyproj import Proj, transform

def merge_sedi_data(data_path: str):
    sq_path_list = glob(os.path.join(data_path,"rawdata","퇴적물","*.xlsx",))
    sq_list = [pd.read_excel(path,header=1) for path in sq_path_list]
    merge1_sq_df = pd.concat(sq_list)
    merge1_sq_df.index = range(len(merge1_sq_df))
    merge1_sq_df.columns = [name.replace(" ","") for name in merge1_sq_df.columns.tolist()]


    # shp파일 ;좌표계 설정해야함 -호소만 좌표계 다르므로 호소만 설정
    shp_list_path = glob(os.path.join(data_path,"rawdata","생물","shp-catchment","*.shp",))
    shp_list = [gpd.GeoDataFrame.from_file(path_shp,encoding = 'cp949') for path_shp in shp_list_path]
    sq_point_shp_1 = shp_list[4]
    sq_point_shp_2 = shp_list[5]

    P5181 = Proj(init='epsg:5181')
    P4326 = Proj(init='epsg:4326')
    new_point_list = [Point(transform(P4326, P5181, x, y)) for x,y in np.concatenate([sq_point_shp_2.geometry.x.values.reshape(-1,1), sq_point_shp_2.geometry.y.values.reshape(-1,1)],axis = 1).tolist()]
    sq_point_shp_2.geometry = new_point_list

    sq_point_shp = pd.concat([sq_point_shp_1,sq_point_shp_2],axis=0)
    sq_point_shp.index = range(len(sq_point_shp))

    shp_list2 = [gpd.GeoDataFrame.from_file(path_shp,encoding = 'cp949') for path_shp in glob(os.path.join(data_path,"pro1shp","CaseStudy_Result (2)","shp","*.shp",))]
    base_shp = shp_list2[7]

    #
    sq_point_shp.crs = {'init':'epsg:5181'}
    sq_point_shp = sq_point_shp.to_crs(base_shp.crs)


    # point 정보와 병합
    merge2_sq_df = pd.merge(merge1_sq_df,sq_point_shp, how='left', left_on='지점명', right_on='조사지점명')
    merge2_sq_df = gpd.GeoDataFrame(merge2_sq_df, geometry= 'geometry')
    merge2_sq_df = merge2_sq_df.loc[~ merge2_sq_df.geometry.isna()]
    merge2_sq_df.index = range(len(merge2_sq_df))

    # 속하는 폴리건 찾기
    # merge_point_df.to_csv(group_level[0] + '.csv', encoding='euc-kr')
    for_subbmatch_df = merge2_sq_df.drop_duplicates(subset = 'geometry')
    for_subbmatch_df.index = range(len(for_subbmatch_df))
    for i in range(len(for_subbmatch_df)):
        try:
            merge2_sq_df.loc[merge2_sq_df.지점명 == for_subbmatch_df.지점명[i],['SBSNCD','SBSNNM','BBSNCD','MBSNCD']] = base_shp[base_shp.contains(for_subbmatch_df.loc[i,'geometry'])].loc[:,['SBSNCD', 'SBSNNM', 'BBSNCD', 'MBSNCD']].values[0]

        except:
            # 포함되는 영역이 없는 지점명
            print(merge2_sq_df.loc[i,['지점명']].tolist())
    # 년월일 데이트타임 생성
    date_list = [str(merge2_sq_df.년[raw_ind]) +"-"+ str(merge2_sq_df.월[raw_ind]) +"-"+ str(merge2_sq_df.일[raw_ind]) for raw_ind in range(len(merge2_sq_df))]
    merge2_sq_df.loc[:,'일시'] = pd.to_datetime(date_list)

    # merge2_sq_df.to_csv(os.path.join(data_path,"merged","merged_1","sq_data.csv",), encoding = 'euc-kr')
    pickle.dump(merge2_sq_df, open(os.path.join(data_path,"merged","merged_1","sq_gdf.p",), "wb"))
