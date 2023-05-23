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


def get_mosaic_raster(data_path: str):

    ###### mosaic 자료 merge and save #######

    landcover_path_list = glob(os.path.join(data_path,'rawdata',"토지피복","extracted","*.tif",))
    landcover_list = [rasterio.open(os.path.join(path_tif)) for path_tif in landcover_path_list]
    mosaic, out_trans = merge(landcover_list)

    with rasterio.open(os.path.join(data_path,'rawdata',"토지피복","mos.tif",),
    "w", driver ='Gtiff',count=1,
        height= mosaic.shape[1],
        width= mosaic.shape[2],
        transform= out_trans,
        crs= landcover_list[0].crs,
        dtype= landcover_list[0].dtypes[0]) as dest:
        dest.write(mosaic)

    
    ############ DEM 좌표계 안맞을 경우 ############
    rds = rioxarray.open_rasterio(os.path.join(data_path,'rawdata',"토지피복","mos.tif",))
    rds_5181 = rds.rio.reproject("EPSG:5181")
    rds_5181.rio.to_raster(os.path.join(data_path,'rawdata',"토지피복","mos_reproj.tif",))


def get_catchment_did_shp(data_path: str):
    shp_list = [gpd.GeoDataFrame.from_file(path_shp,encoding = 'cp949') for path_shp in glob(os.path.join(data_path,"pro1shp","CaseStudy_Result (2)","shp","*.shp",))]
    
    # base_shp = shp_list[7]
    cat_did_shp = gpd.GeoDataFrame.from_file(os.path.join(data_path,"KRF_ver3_total","KRF_ver3_CATCHMENT.shp",),encoding = 'cp949')

    import pyproj
    from shapely.geometry import Point
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
    # cat_did_shp.plot(ax = ax)

    if os.path.join(data_path,'rawdata',"토지피복","mos.tif",) not in glob(os.path.join(data_path,'rawdata',"토지피복","*.tif",)):
        get_mosaic_raster(data_path=data_path)


    land_c = rasterio.open(os.path.join(data_path,"rawdata","토지피복","mos_reproj.tif",))

    ### 폴리건별 각 카테고리 비율 구하기 ###
    total_raster_arr = land_c.read(1)

    masked_0_region = np.where(total_raster_arr == 0, 1, 0)
    masked_1_region = np.where(total_raster_arr == 1, 1, 0)
    masked_2_region = np.where(total_raster_arr == 2, 1, 0)
    masked_3_region = np.where(total_raster_arr == 3, 1, 0)
    masked_4_region = np.where(total_raster_arr == 4, 1, 0)
    masked_5_region = np.where(total_raster_arr == 5, 1, 0)
    masked_6_region = np.where(total_raster_arr == 6, 1, 0)
    masked_7_region = np.where(total_raster_arr == 7, 1, 0)

    from rasterstats import zonal_stats

    cate_ratio_list = []

    ratio_cate_0 = zonal_stats(vectors=cat_did_shp,raster=masked_0_region, affine=land_c.transform,stats=["mean"],)
    ratio_cate_1 = zonal_stats(vectors=cat_did_shp,raster=masked_1_region, affine=land_c.transform,stats=["mean"],)
    ratio_cate_2 = zonal_stats(vectors=cat_did_shp,raster=masked_2_region, affine=land_c.transform,stats=["mean"],)
    ratio_cate_3 = zonal_stats(vectors=cat_did_shp,raster=masked_3_region, affine=land_c.transform,stats=["mean"],)
    ratio_cate_4 = zonal_stats(vectors=cat_did_shp,raster=masked_4_region, affine=land_c.transform,stats=["mean"],)
    ratio_cate_5 = zonal_stats(vectors=cat_did_shp,raster=masked_5_region, affine=land_c.transform,stats=["mean"],)
    ratio_cate_6 = zonal_stats(vectors=cat_did_shp,raster=masked_6_region, affine=land_c.transform,stats=["mean"],)
    ratio_cate_7 = zonal_stats(vectors=cat_did_shp,raster=masked_7_region, affine=land_c.transform,stats=["mean"],)


    ratio_cate_0_tolist = []
    ratio_cate_1_tolist = []
    ratio_cate_2_tolist = []
    ratio_cate_3_tolist = []
    ratio_cate_4_tolist = []
    ratio_cate_5_tolist = []
    ratio_cate_6_tolist = []
    ratio_cate_7_tolist = []

    for i in range(len(ratio_cate_0)):
        val0 = ratio_cate_0[i]['mean']
        val1 = ratio_cate_1[i]['mean']
        val2 = ratio_cate_2[i]['mean']
        val3 = ratio_cate_3[i]['mean']
        val4 = ratio_cate_4[i]['mean']
        val5 = ratio_cate_5[i]['mean']
        val6 = ratio_cate_6[i]['mean']
        val7 = ratio_cate_7[i]['mean']


        ratio_cate_0_tolist.append(val0)
        ratio_cate_1_tolist.append(val1)
        ratio_cate_2_tolist.append(val2)
        ratio_cate_3_tolist.append(val3)
        ratio_cate_4_tolist.append(val4)
        ratio_cate_5_tolist.append(val5)
        ratio_cate_6_tolist.append(val6)
        ratio_cate_7_tolist.append(val7)


    cat_did_shp.loc[:,'cate_0'] = ratio_cate_0_tolist
    cat_did_shp.loc[:,'cate_1'] = ratio_cate_1_tolist
    cat_did_shp.loc[:,'cate_2'] = ratio_cate_2_tolist
    cat_did_shp.loc[:,'cate_3'] = ratio_cate_3_tolist
    cat_did_shp.loc[:,'cate_4'] = ratio_cate_4_tolist
    cat_did_shp.loc[:,'cate_5'] = ratio_cate_5_tolist
    cat_did_shp.loc[:,'cate_6'] = ratio_cate_6_tolist
    cat_did_shp.loc[:,'cate_7'] = ratio_cate_7_tolist

    pickle.dump(cat_did_shp, open(os.path.join(data_path,"merged","merged_1","landcover_gdf.p",), "wb"))

    landcshp_list = [gpd.GeoDataFrame.from_file(path_shp,encoding = 'cp949') for path_shp in glob(os.path.join(data_path,"rawdata","토지피복","세분류","extracted","*.shp",))]
    landcshp_list_filt = [geo_pd.to_crs(landcshp_list[0].crs) for geo_pd in landcshp_list]
    del landcshp_list
    test = pd.concat(landcshp_list_filt)
    del landcshp_list_filt
    cat_did_shp = pickle.load(open(os.path.join(data_path,"merged","merged_1","landcover_gdf.p",), "rb")) 
    test = test.to_crs(cat_did_shp.crs)
    test2 = cat_did_shp.loc[cat_did_shp.cate_0 != 0].overlay(test, how='intersection')

    for i in list(set(test2.CAT_DID.values)):
        now_cat = test2.loc[test2.CAT_DID == i]
        now_land_cat = cat_did_shp.loc[cat_did_shp.CAT_DID == i]
        new_cate_0 = 1 - sum(now_cat.area)/now_land_cat.area
        new_cate_1 = sum(now_cat.loc[now_cat.L1_CODE == '100'].area)/now_land_cat.area
        new_cate_2 = sum(now_cat.loc[now_cat.L1_CODE == '200'].area)/now_land_cat.area
        new_cate_3 = sum(now_cat.loc[now_cat.L1_CODE == '300'].area)/now_land_cat.area
        new_cate_4 = sum(now_cat.loc[now_cat.L1_CODE == '400'].area)/now_land_cat.area
        new_cate_5 = sum(now_cat.loc[now_cat.L1_CODE == '500'].area)/now_land_cat.area
        new_cate_6 = sum(now_cat.loc[now_cat.L1_CODE == '600'].area)/now_land_cat.area
        new_cate_7 = sum(now_cat.loc[now_cat.L1_CODE == '700'].area)/now_land_cat.area

        cat_did_shp.loc[cat_did_shp.CAT_DID == i,['cate_0','cate_1','cate_2','cate_3','cate_4','cate_5','cate_6','cate_7',]] = np.concatenate([new_cate_0.values,new_cate_1.values,new_cate_2.values,new_cate_3.values,new_cate_4.values,new_cate_5.values,new_cate_6.values,new_cate_7.values,])

        print(sum([new_cate_0,new_cate_1,new_cate_2,new_cate_3,new_cate_4,new_cate_5,new_cate_6,new_cate_7,]))
    pickle.dump(cat_did_shp, open(os.path.join(data_path,"merged","merged_1","landcover_gdf_2.p",), "wb"))

    return cat_did_shp
