from glob import glob
import pandas as pd
import os
import pickle
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

def merge_weather_data(data_path: str):
    em_dataset_path_list = glob(os.path.join(data_path,"rawdata","기후","sensor_data","*.csv",))
    em_dataset_list = [pd.read_csv(path, encoding='cp949') for path in em_dataset_path_list]

    em_meta = pd.read_csv(glob(os.path.join(data_path,"rawdata","기후","meta","*.csv",))[0],encoding='cp949')
    em_df = pd.concat(em_dataset_list, axis = 0)

    from pyproj import Proj, transform
    geo_point = []
    P5181 = Proj(init='epsg:5181')
    P4326 = Proj(init='epsg:4326')
    for i in range(len(em_meta)):
        x,y = transform(P4326, P5181, em_meta.loc[i,'경도'], em_meta.loc[i,'위도'])
        geo_point.append(Point(x,y))
        

    em_meta['geometry'] = geo_point

    em_point_gdf = gpd.GeoDataFrame(em_meta, geometry= 'geometry')


    shp_list = [gpd.GeoDataFrame.from_file(path_shp,encoding = 'cp949') for path_shp in glob(os.path.join(data_path,"pro1shp","CaseStudy_Result (2)","shp","*.shp",))]
    base_shp = shp_list[7]

    # base_shp = shp_list[7]
    cat_did_shp = gpd.GeoDataFrame.from_file(os.path.join(data_path,"KRF_ver3_total","KRF_ver3_CATCHMENT.shp",),encoding = 'cp949')

    import pyproj
    
    from  shapely.ops import transform

    geo_point = []
    # P5181 = pyproj.CRS('EPSG:7019')
    P5181 = shp_list[7].crs
    P4326 = pyproj.CRS('EPSG:4326')

    project = pyproj.Transformer.from_crs(P4326, P5181, always_xy=True).transform
    # project = pyproj.Transformer.from_crs(P5181, P4326).transform

    for i in range(len(cat_did_shp)):
        tt = transform(project, cat_did_shp.loc[i,'geometry'])
        geo_point.append(tt)

    cat_did_shp['geometry'] = geo_point
    cat_did_shp.crs = {'init':'epsg:5181'}
    cat_did_shp = cat_did_shp.to_crs(shp_list[7].crs)



    em_point_gdf.crs = {'init':'epsg:5181'}
    em_point_gdf = em_point_gdf.to_crs(base_shp.crs)

    ### 포인트 속하는 코드 부여

    for i in range(len(em_point_gdf)):
        search_rs = cat_did_shp[cat_did_shp.contains(em_point_gdf.geometry[i])]
        try:
            em_point_gdf.loc[i,cat_did_shp.columns.tolist()[:-1]] = search_rs.iloc[0,:-1]
        except:
            pass

    ### 백령도 등 표준유역에 포함되지 않는 지점 제거 ###
    drop_ind = em_point_gdf.loc[em_point_gdf.CAT_DID.isna()].index.tolist()
    drop_point = em_point_gdf.loc[em_point_gdf.CAT_DID.isna()].지점.values.tolist()
    em_point_gdf = em_point_gdf.drop(drop_ind)
    em_point_gdf.index = range(len(em_point_gdf))

    drop_data_ind = []
    for i in drop_point:
        drop_data_ind.extend(em_df.loc[em_df.지점 == i].index.tolist())

    em_df = em_df.drop(drop_data_ind)
    em_df.index = range(len(em_df))
    #################################################

    for i in list(set(em_df.loc[:,'지점'].values.tolist())):
        l = len(em_df.loc[em_df.지점 == i])
        em_df.loc[em_df.지점 == i,em_point_gdf.columns.tolist()] = np.array(em_point_gdf.loc[(em_point_gdf.지점 == i),em_point_gdf.columns.tolist()].iloc[0,:].values.reshape(1,-1).tolist() * l, dtype=object)


    em_df = gpd.GeoDataFrame(em_df, geometry = 'geometry')
    pickle.dump(em_df, open(os.path.join(data_path,"merged","merged_1","atm_gdf.p",), "wb"))


def resample_weather_data(atm_gdf):

    remove_col = {'지점',
    '지점명',
    '일시',
    '기사',
    '안개 계속시간(hr)',
    '시작일',
    '종료일',
    '지점주소',
    '관리관서',
    '위도',
    '경도',
    '노장해발고도(m)',
    '기압계(관측장비지상높이(m))',
    '기온계(관측장비지상높이(m))',
    '풍속계(관측장비지상높이(m))',
    '강우계(관측장비지상높이(m))',
    'geometry',
    'OBJECTID',
    'CAT_ID',
    'CAT_DID',
    'BRU_X',
    'BRU_Y',
    'BLL_X',
    'BLL_Y',
    'Shape_Leng',
    'Shape_Area',
    'streamname',
    'BASIN_ID',
    'BASIN_NM',
    'MB_ID',
    'MB_NM',
    'SB_ID',
    'SB_NM',
    'CAT_SN',
    'CAT_DIV',
    'AREA',
    'PERI',
    'CAT_FLAG'
    }

    after_add = [
                'OBJECTID',
                'CAT_ID',
                'CAT_DID',
                'BRU_X',
                'BRU_Y',
                'BLL_X',
                'BLL_Y',
                'Shape_Leng',
                'Shape_Area',
                'streamname',
                'BASIN_ID',
                'BASIN_NM',
                'MB_ID',
                'MB_NM',
                'SB_ID',
                'SB_NM', 
                '지점명',
                'geometry']
    all_col = set(atm_gdf.columns)
    remain_col = list(all_col - remove_col) + ['일시']

    site_list = list(set(atm_gdf.지점명.values))
    df_bysite_list = []
    for s_name in site_list:
        t_df = atm_gdf.loc[atm_gdf.지점명 == s_name, remain_col].set_index('일시').resample(rule = '1Y').mean()
        t_df.loc[:,after_add] = np.array(atm_gdf.loc[atm_gdf.지점명 == s_name, after_add].iloc[0].tolist(), dtype = object)
        df_bysite_list.append(t_df)

    atm_df_resampled = pd.concat(df_bysite_list)
    atm_df_resampled.loc[:,'년'] = atm_df_resampled.index.year.tolist()
    atm_df_resampled.index = range(len(atm_df_resampled))
    atm_df_filter_1 = atm_df_resampled.loc[:, ['일강수량(mm)', '평균기온(°C)', '합계 일조시간(hr)', '지점명', 'geometry', '년', 'CAT_DID']]
    tem_cols = atm_df_filter_1.columns.tolist()
    tem_cols.remove('년')
    atm_df_filter_1 = atm_df_filter_1.loc[(atm_df_filter_1.년 >= 2015) & (atm_df_filter_1.년 <= 2020),tem_cols]
    
    mean_df = atm_df_filter_1.groupby('지점명').mean()
    q1_df = atm_df_filter_1.groupby('지점명').quantile([0.25]).set_index(mean_df.index)
    q1_df.columns = [name+"_q1" for name in q1_df.columns.to_list()]
    q3_df = atm_df_filter_1.groupby('지점명').quantile([0.75]).set_index(mean_df.index)
    q3_df.columns = [name+"_q3" for name in q3_df.columns.to_list()]

    atm_df_filter_2 = pd.concat([mean_df,q1_df,q3_df], axis = 1)
    atm_df_filter_2.loc[:,'지점명'] = atm_df_filter_2.index
    atm_df_filter_2.index = range(len(atm_df_filter_2))
    
    atm_df_filter_3 = pd.merge(atm_df_filter_2,atm_df_filter_1.drop_duplicates('지점명').loc[:, ['지점명', 'geometry']], how = 'left', on = '지점명')
    return atm_df_filter_3