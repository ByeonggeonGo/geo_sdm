from . import ecological_info
from . import hydinfo
from . import landcover
from . import sedi_info
from . import water_quality_info
from . import weather

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
from rasterstats import zonal_stats

from copy import deepcopy

class DataContorller():

    def __init__(self, data_path: str, use_cleaned_dataset: bool):
        self._data_path = data_path
        # 데이터 불러오기
    
        ### dataset information ###
        #     * 데이터셋 가용기간
        # - 수리수문자료 2010 ~ 2020
        # - 기후자료 2010 ~ 2021
        # - 생물자료 2011 ~ 2021
        # - 수질자료 2010 ~ 2021 (api기준)
        # - 토지피복 (전국단위 2010년대말)
        # - 퇴적물 2015 - 2020

        # --> 2015 ~ 2020 자료 병합

        if use_cleaned_dataset:

            # 자료 5종류 + 생물측정망자료
            self._wq_gdf = pickle.load(open(os.path.join(self._data_path,"merged","merged_1","wq_gdf_newshp_3.p",), "rb")) 
            self._sq_gdf = pickle.load(open(os.path.join(self._data_path,"merged","merged_1","sq_gdf.p",), "rb")) 
            self._landcover_gdf = pickle.load(open(os.path.join(self._data_path,"merged","merged_1","landcover_gdf_2.p",), "rb"))
            # self._landcover_gdf = self._landcover_gdf.astype({"CAT_DID": "int64"})

            self._hyd_gdf = pickle.load(open(os.path.join(self._data_path,"merged","merged_1","hyd_gdf.p",), "rb")) 
            self._atm_gdf = pickle.load(open(os.path.join(self._data_path,"merged","merged_1","atm_gdf.p",), "rb")) 
            self._Family_gdf = pickle.load(open(os.path.join(self._data_path,"merged","merged_1","Family_gdf.p",), "rb"))
            self._Order_gdf = pickle.load(open(os.path.join(self._data_path,"merged","merged_1","Order_gdf.p",), "rb"))
            self._name_gdf = pickle.load(open(os.path.join(self._data_path,"merged","merged_1","학명_gdf.p",), "rb"))

            self._main_landcover_crs = self._landcover_gdf.crs



            #좌표계 설정
            self._atm_gdf.crs = {'init':'epsg:5181'}
            self._atm_gdf = self._atm_gdf.to_crs(self._main_landcover_crs)

            self._hyd_gdf.crs = {'init':'epsg:5181'}
            self._hyd_gdf = self._hyd_gdf.to_crs(self._main_landcover_crs)

            self._wq_gdf = self._wq_gdf.to_crs(self._main_landcover_crs)
            # datetime 설정
            # wq_gdf.연도 = pd.to_datetime(wq_gdf.WMYR,format="%Y").dt.year
            self._atm_gdf.일시 = pd.to_datetime(self._atm_gdf.일시)
            self._Family_gdf.조사년도 = pd.to_datetime(self._Family_gdf.조사년도,format="%Y")
            self._Order_gdf.조사년도 = pd.to_datetime(self._Order_gdf.조사년도,format="%Y")
            self._name_gdf.조사년도 = pd.to_datetime(self._name_gdf.조사년도,format="%Y")

        else:
            self.make_cleaned_dataset_for_allcategories()

    def make_cleaned_dataset_for_allcategories(self):

        ecological_info.reshape_eco_monitoring_data(self._data_path)
        hydinfo.merge_hyd_monitoring_n_coords(self._data_path)
        landcover.get_catchment_did_shp(self._data_path)
        sedi_info.merge_sedi_data(self._data_path)
        water_quality_info.get_wq_api_data(self._data_path)
        weather.merge_weather_data(self._data_path)

    
    def get_merged_set(self, summary_level: str):

        sp_gdf = self._Order_gdf if summary_level == 'Order' else self._Family_gdf if summary_level == 'Family' else self._name_gdf

        filter_sp_data = ecological_info.resample_ecological_data(sp_gdf)
        filter_atm_data = weather.resample_weather_data(self._atm_gdf)
        filter_wq_data = water_quality_info.resample_wq_data(self._wq_gdf)
        filter_hyd_data = hydinfo.resample_hyd_data(self._hyd_gdf)


        # 기상지점 할당
        for i in range(len(filter_sp_data)):
            # test.geometry[i].distance(atm_df_resampled.geometry)
            result = [filter_sp_data.geometry[i].distance(at_point) for at_point in filter_atm_data.geometry]
            min_ind = np.argmin(result)
            filter_sp_data.loc[i,'기상지점'] = filter_atm_data.iloc[min_ind].지점명# 수질정보 병합
        
        # 수질정보 병합
        tem_cols = filter_wq_data.columns.tolist()
        tem_cols.remove('geometry')

        merged_df = pd.merge(filter_sp_data, filter_wq_data.loc[:,tem_cols], how='left', left_on=['CAT_DID'], right_on=['CAT_DID'])
        merged_df = merged_df.drop_duplicates('조사지점')
        merged_df.index = range(len(merged_df))
        
        # 기상정보 병합
        tem_cols_2 = filter_atm_data.columns.tolist()
        tem_cols_2.remove('geometry')

        merged_df_2 = pd.merge(merged_df, filter_atm_data.loc[:,tem_cols_2], how='left', left_on=['기상지점'], right_on=['지점명'])

        #수리수문 병합
        tem_cols_3 = filter_hyd_data.columns.tolist()
        tem_cols_3.remove('관측소명')
        tem_cols_3.remove('geometry')

        merged_df_3 = pd.merge(merged_df_2, filter_hyd_data.loc[:,tem_cols_3], how='left', left_on=['CAT_DID'], right_on=['CAT_DID'])


        # 토지피복정보 병합
        tem_cols_4 = merged_df_3.columns.tolist()
        tem_cols_4.remove('지점명')

        merged_df_4 = pd.merge(merged_df_3.loc[:,tem_cols_4], self._landcover_gdf.loc[:,['cate_1','cate_2','cate_3','cate_4','cate_5','cate_6','cate_7','CAT_DID']], how='left', left_on=['CAT_DID'], right_on=['CAT_DID'])
        
        # 고도 및 기울기(지리정보) 병합
        # 좌표정보 dem자료로 바꾼 후 고도 등 머지
        ## dem에서 elevation 따오기
        dem_90_raster = rasterio.open(os.path.join(self._data_path,"rawdata","한반도","한반도90m_GRS80.img",))

        merged_df_4_gpd = gpd.GeoDataFrame(merged_df_4, geometry='geometry')
        merged_df_4_gpd.crs = self._landcover_gdf.crs
        merged_df_4_gpd = merged_df_4_gpd.to_crs(dem_90_raster.crs)
        # zonal_stats(vectors=merged_df_4_gpd,raster=masked_0_region, affine=dem_90_raster.transform,stats=["mean"],)
        elevation = zonal_stats(vectors=merged_df_4_gpd,raster=dem_90_raster.read(1), affine=dem_90_raster.transform,stats=["mean"],)
        elevation_list = [ele['mean'] for ele in elevation]
        merged_df_4_gpd.loc[:,'elevation'] = elevation_list
        ele = dem_90_raster.read(1)
        cellsize = 90.
        px, py = np.gradient(ele, cellsize)
        slope = np.sqrt(px ** 2 + py ** 2)
        slope_deg = np.degrees(np.arctan(slope))

        slope = zonal_stats(vectors=merged_df_4_gpd,raster=slope_deg, affine=dem_90_raster.transform,stats=["mean"],)
        slo_list = [slo['mean'] for slo in slope]
        merged_df_4_gpd.loc[:,'slope'] = slo_list

        # 머지지역중 동일 CAT-DID에 여러지점있는 경우가 있어서, 비슷할것이라 가정하고 중복제거
        merged_df_4_gpd = merged_df_4_gpd.drop_duplicates('조사지점')
        return merged_df_4_gpd
        

    def get_merged_set_year(self, summary_level: str):
        sp_gdf, wq_gdf, hyd_gdf, atm_gdf = self.get_filterd_df(summary_level)

        # 기상지점 할당
        temp_df = sp_gdf.loc[:,['조사지점', 'geometry']].drop_duplicates('조사지점')
        temp_df.index = range(len(temp_df))

        for i in range(len(temp_df)):
            # test.geometry[i].distance(atm_df_resampled.geometry)
            result = [temp_df.geometry[i].distance(at_point) for at_point in atm_gdf.loc[atm_gdf.year.values == 2010].drop_duplicates("지점명").geometry]
            min_ind = np.argmin(result)
            temp_df.loc[i,'기상지점'] = atm_gdf.loc[atm_gdf.year.values == 2010].drop_duplicates("지점명").iloc[min_ind].지점명# 수질정보 병합

        sp_gdf = pd.merge(sp_gdf, temp_df.loc[:,['조사지점', '기상지점']], how = 'left', on = '조사지점')

        merged_df = deepcopy(sp_gdf)
        for origin_df in [wq_gdf, hyd_gdf]:
            resam_df = get_resampled_df(origin_df)
            merged_df = pd.merge(merged_df, resam_df, how='left', left_on=['CAT_DID','조사년도', '조사회차'], right_on=['CAT_DID','조사년도', '조사회차'])
        merged_df = pd.merge(merged_df, self._landcover_gdf.loc[:,['cate_1','cate_2','cate_3','cate_4','cate_5','cate_6','cate_7','CAT_DID']], how='left', left_on=['CAT_DID'], right_on=['CAT_DID'])
        
        resam_df = get_resampled_df_atm(atm_gdf)
        atm_cols = [
            "지점명",
            '조사년도', '조사회차',
            "평균기온(°C)",
            "최저기온(°C)",
            "최고기온(°C)",
            "일강수량(mm)"
        ]

        merged_df = pd.merge(merged_df, resam_df.loc[:,atm_cols], how='left', left_on=['기상지점','조사년도', '조사회차'], right_on=['지점명','조사년도', '조사회차'])
        return merged_df

    def get_merged_set_quarter(self, summary_level: str):
        
        sp_gdf, wq_gdf, hyd_gdf, atm_gdf = self.get_filterd_df(summary_level)
        # 기상지점 할당
        temp_df = sp_gdf.loc[:,['조사지점', 'geometry']].drop_duplicates('조사지점')
        temp_df.index = range(len(temp_df))

        for i in range(len(temp_df)):
            # test.geometry[i].distance(atm_df_resampled.geometry)
            result = [temp_df.geometry[i].distance(at_point) for at_point in atm_gdf.loc[atm_gdf.year.values == 2010].drop_duplicates("지점명").geometry]
            min_ind = np.argmin(result)
            temp_df.loc[i,'기상지점'] = atm_gdf.loc[atm_gdf.year.values == 2010].drop_duplicates("지점명").iloc[min_ind].지점명# 수질정보 병합

        sp_gdf = pd.merge(sp_gdf, temp_df.loc[:,['조사지점', '기상지점']], how = 'left', on = '조사지점')

        merged_df = deepcopy(sp_gdf)
        for origin_df in [wq_gdf, hyd_gdf]:
            resam_df = get_resampled_df_season(origin_df)
            merged_df = pd.merge(merged_df, resam_df, how='left', left_on=['CAT_DID','조사년도', '조사회차'], right_on=['CAT_DID','조사년도', '조사회차'])
        merged_df = pd.merge(merged_df, self._landcover_gdf.loc[:,['cate_1','cate_2','cate_3','cate_4','cate_5','cate_6','cate_7','CAT_DID']], how='left', left_on=['CAT_DID'], right_on=['CAT_DID'])
        

        resam_df = get_resampled_df_season_atm(atm_gdf)
        atm_cols = [
            "지점명",
            '조사년도', '조사회차',
            "평균기온(°C)",
            "최저기온(°C)",
            "최고기온(°C)",
            "일강수량(mm)"
        ]

        merged_df = pd.merge(merged_df, resam_df.loc[:,atm_cols], how='left', left_on=['기상지점','조사년도', '조사회차'], right_on=['지점명','조사년도', '조사회차'])
        return merged_df

    def get_filterd_df(self, summary_level: str):
        # 데이터 불러오기
        hyd_cols = ['CAT_DID','측정시각', '수면폭', '단면적', '평균유속', '평균수심', '유량']
        wq_cols = [
            "CAT_DID",
            "WMYR",
            "WMOD",
            "ITEM_TEMP",
            "ITEM_PH",
            "ITEM_BOD",
            "ITEM_COD",
            "ITEM_SS",
            "ITEM_TN",
            "ITEM_TP",
            "ITEM_CLOA",
            "ITEM_EC",
            "ITEM_NO3N",
            "ITEM_NH3N",
            "ITEM_TOC",
        ]

        atm_cols = [
            "지점명",
            "geometry",
            "CAT_DID",
            "일시",
            "평균기온(°C)",
            "최저기온(°C)",
            "최고기온(°C)",
            "일강수량(mm)"
        ]
        Family_gdf = pickle.load(open(os.path.join(self._data_path,"merged","merged_1","Family_gdf.p",), "rb"))
        Order_gdf = pickle.load(open(os.path.join(self._data_path,"merged","merged_1","Order_gdf.p",), "rb"))
        name_gdf = pickle.load(open(os.path.join(self._data_path,"merged","merged_1","학명_gdf.p",), "rb"))


        sp_gdf = Order_gdf if summary_level == 'Order' else Family_gdf if summary_level == 'Family' else name_gdf


        # 자료 5종류 + 생물측정망자료
        wq_gdf = pickle.load(open(os.path.join(self._data_path,"merged","merged_1","wq_gdf_newshp_3.p",), "rb"))[wq_cols]
        hyd_gdf = pickle.load(open(os.path.join(self._data_path,"merged","merged_1","hyd_gdf.p",), "rb"))[hyd_cols]
        atm_gdf = pickle.load(open(os.path.join(self._data_path,"merged","merged_1","atm_gdf.p",), "rb")) [atm_cols]
        
        # 날짜 정리
        wq_gdf['Date'] = pd.to_datetime(wq_gdf['WMYR'].astype(str) + wq_gdf['WMOD'].astype(str), format='%Y%m')
        atm_gdf['일시'] = pd.to_datetime(atm_gdf['일시'].astype(str), format='%Y-%m-%d')
        atm_gdf['Date'] = atm_gdf['일시']
        hyd_gdf['Date'] = hyd_gdf['측정시각']

        # 안쓰는 시간정보 컬럼 제거
        wq_gdf = wq_gdf.drop(['WMYR', 'WMOD'], axis=1)
        atm_gdf = atm_gdf.drop(['일시'], axis=1)
        hyd_gdf = hyd_gdf.drop(['측정시각'], axis=1)

        # 년도 쿼터정보 할당
        hyd_gdf.loc[:,'quarter'] = hyd_gdf['Date'].dt.quarter
        hyd_gdf.loc[:,'year'] = hyd_gdf['Date'].dt.year

        wq_gdf.loc[:,'quarter'] = wq_gdf['Date'].dt.quarter
        wq_gdf.loc[:,'year'] = wq_gdf['Date'].dt.year

        atm_gdf.loc[:,'quarter'] = atm_gdf['Date'].dt.quarter
        atm_gdf.loc[:,'year'] = atm_gdf['Date'].dt.year

        # 수질데이터 정량한계미만 0으로 할당
        wq_gdf.replace('정량한계미만', 0, inplace=True)
        # 정량한계 미만 제외 후 dtype지정
        wq_gdf.loc[:, (wq_gdf.columns != 'CAT_DID') & (wq_gdf.columns != 'Date')] = wq_gdf.loc[:, (wq_gdf.columns != 'CAT_DID') & (wq_gdf.columns != 'Date')].astype(float)
        return sp_gdf, wq_gdf, hyd_gdf, atm_gdf


#봄철 측정인경우
def get_round_info_spring(row):
    if row['quarter'] == 1:
        return row['Date'].year
    else:
        return row['Date'].year + 1
    
#가을철 측정인경우
def get_round_info_autumn(row):
    if row['quarter'] == 4:
        return row['Date'].year + 1
    else:
        return row['Date'].year
    
#동일한 계절만 묶는 경우
def get_round_info(row):
    if row['quarter'] == 1:
        return 1
    elif row['quarter'] == 3:
        return 2
    else:
        return 0

def get_resampled_df(df):
    def get_resampled_df_by_round(df, measure_round):

        if measure_round == 1:
            df['조사년도'] = df.apply(get_round_info_spring, axis=1)
        elif measure_round == 2:
            df['조사년도'] = df.apply(get_round_info_autumn, axis=1)
        
        df = df.groupby(['CAT_DID','조사년도']).mean()
        cat_did = df.index.get_level_values('CAT_DID')
        meas_y = df.index.get_level_values('조사년도')


        df.reset_index(drop=True, inplace=True)
        df.insert(0, 'CAT_DID', cat_did)
        df.insert(1, '조사년도', meas_y)
        df.reset_index(drop=True, inplace=True)
        df = df.drop(['quarter', 'year'], axis=1)

        df.insert(df.columns.get_loc('조사년도') + 1, '조사회차', measure_round)

        return df
    df_list = []
    for i in [1,2]:
        df_tm = get_resampled_df_by_round(df, i)
        df_list.append(df_tm)
    final_df = pd.concat(df_list, axis = 0)
    final_df.index = range(len(final_df))
    return final_df

def get_resampled_df_season(df):

    df = df.groupby(['CAT_DID','year','quarter']).mean()
    df = df.reset_index()

    df['조사회차'] = df.apply(get_round_info, axis=1)
    df['조사년도'] = df['year']

    df = df.drop(['quarter', 'year'], axis=1)
    return df

def get_resampled_df_atm(df):
    def get_resampled_df_by_round(df, measure_round):

        if measure_round == 1:
            df['조사년도'] = df.apply(get_round_info_spring, axis=1)
        elif measure_round == 2:
            df['조사년도'] = df.apply(get_round_info_autumn, axis=1)
        
        df = df.groupby(['지점명','조사년도']).mean()
        cat_did = df.index.get_level_values('지점명')
        meas_y = df.index.get_level_values('조사년도')


        df.reset_index(drop=True, inplace=True)
        df.insert(0, '지점명', cat_did)
        df.insert(1, '조사년도', meas_y)
        df.reset_index(drop=True, inplace=True)
        df = df.drop(['quarter', 'year'], axis=1)

        df.insert(df.columns.get_loc('조사년도') + 1, '조사회차', measure_round)

        return df
    df_list = []
    for i in [1,2]:
        df_tm = get_resampled_df_by_round(df, i)
        df_list.append(df_tm)
    final_df = pd.concat(df_list, axis = 0)
    final_df.index = range(len(final_df))
    return final_df

def get_resampled_df_season_atm(df):

    df = df.groupby(['지점명','year','quarter']).mean()
    df = df.reset_index()

    df['조사회차'] = df.apply(get_round_info, axis=1)
    df['조사년도'] = df['year']

    df = df.drop(['quarter', 'year'], axis=1)
    return df