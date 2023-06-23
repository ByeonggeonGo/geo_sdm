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


class DataContorller():

    def __init__(self, data_path: str, use_cleaned_dataset: bool):
        self._data_path = data_path

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
        self._final_set = merged_df_4_gpd
        








