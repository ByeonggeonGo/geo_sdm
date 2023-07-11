from glob import glob
import os
import geopandas as gpd 
import numpy as np
import pickle
import pandas as pd
import requests




def get_wq_api_data(data_path: str):

    # https://apis.data.go.kr/1480523/WaterQualityService/getWaterMeasuringList?numOfRows=1&pageNo=1&serviceKey=k5wXUhoJHwee1cncQCBmm81YbQ%2Bexttb0vdJcyF5GuGJn0mbGBNNL%2FER2VfkrJMlExfc%2BFZjPeRuOM2bvgDYyQ%3D%3D&resultType=json&ptNoList=3008A40
    WaterMeasuring_site_code_list_path = os.path.join(data_path,"rawdata","수질","국립환경과학원_openAPI_활용가이드_수질DB","물환경 수질측정망 운영결과 DB_물환경_코드_코드명.xlsx",)
    WaterMeasuring_site_code_df = pd.read_excel(WaterMeasuring_site_code_list_path, header= [1])
    WaterMeasuring_site_code_list = WaterMeasuring_site_code_df.지점코드.tolist()
    WaterMeasuring_site_code_arr = np.array(WaterMeasuring_site_code_list).reshape(-1,32)

   
    url = 'http://apis.data.go.kr/1480523/WaterQualityService/getWaterMeasuringList'
    key = "k5wXUhoJHwee1cncQCBmm81YbQ+exttb0vdJcyF5GuGJn0mbGBNNL/ER2VfkrJMlExfc+FZjPeRuOM2bvgDYyQ=="
    params ={'serviceKey' : key, 'pageNo' : '1', 'numOfRows' : '1000000', 'resultType' : 'JSON', 'ptNoList' : [], 'wmyrList' : '2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022', 'wmwkList' : '1,2,3,4,5,6,7,8,9,10,11,12' }

    df_list = []
    for i in range(len(WaterMeasuring_site_code_arr)):
        site_code_param = ",".join(WaterMeasuring_site_code_arr[i])
        params['ptNoList'] = site_code_param
        response = requests.get(url, params=params)

        data_dic = response.json()
        df_list.append(pd.DataFrame(data_dic['getWaterMeasuringList']['item']))
    all_watermeasure_df = pd.concat(df_list)


    ## 공간정보포털자료 좌표정보기준 코드로 매칭
    shp_list_path = glob(os.path.join(data_path,"rawdata","수질","updated_point","extracted","*.shp",))
    shp_list = [gpd.GeoDataFrame.from_file(path_shp,encoding = 'cp949') for path_shp in shp_list_path]
    wq_point_shp = pd.concat(shp_list, axis=0)
    wq_point_shp.index = range(len(wq_point_shp))
    merged_wq_df = pd.merge(all_watermeasure_df, wq_point_shp, how='left', left_on='PT_NO', right_on= '측정소코드')

    merged_wq_df = merged_wq_df.loc[~merged_wq_df.geometry.isna()]
    merged_wq_df.index = range(len(merged_wq_df))
    # pickle.dump(merged_wq_gdf, open(os.path.join(data_path,"merged","merged_1","wq_gdf_newshp_3.p",), "wb"))

    merged_wq_gdf = gpd.GeoDataFrame(merged_wq_df, geometry='geometry')
    cat_did_shp = gpd.GeoDataFrame.from_file(os.path.join(data_path,"KRF_ver3_total","KRF_ver3_CATCHMENT.shp",),encoding = 'cp949')
    merged_wq_gdf.crs = {'init':'epsg:5181'}
    merged_wq_gdf = merged_wq_gdf.to_crs(cat_did_shp.crs)
    for i in range((len(cat_did_shp))):
        merged_wq_gdf.loc[merged_wq_gdf.geometry[:].within(cat_did_shp.geometry[i]),'CAT_DID'] = cat_did_shp.loc[i,'CAT_DID']

    merged_wq_gdf = merged_wq_gdf.loc[~merged_wq_gdf.CAT_DID.isna()]
    merged_wq_gdf.index = range(len(merged_wq_gdf))
    pickle.dump(merged_wq_gdf, open(os.path.join(data_path,"merged","merged_1","wq_gdf_newshp_3.p",), "wb"))


def resample_wq_data(wq_gdf):
    col_names_dic= { 'ITEM_LVL' : '수위(m)',
    'ITEM_AMNT' : '유량(m3/sec)',
    'ITEM_TEMP' : '수온',
    'ITEM_PH' : 'pH', 
    'ITEM_DOC' : 'DO(mg/L)',
    'ITEM_BOD' : 'BOD(mg/L)',
    'ITEM_COD' : 'COD(mg/L)',
    'ITEM_SS' : 'SS(mg/L)',
    'ITEM_TCOLI' : 'T_coli(총대장균군수/100ml)',
    'ITEM_TN' : 'TN(mg/L)',
    'ITEM_TP' : 'TP(mg/L)',
    'ITEM_CD' : 'Cd(mg/L)',
    'ITEM_CN' : 'Cn(mg/L)',
    'ITEM_PB' : 'Pb(mg/L)',
    'ITEM_CR6' : 'Cr6(mg/L)',
    'ITEM_AS' : 'As(mg/L)',
    'ITEM_HG' : 'Hg(mg/L)',
    'ITEM_CU' : 'Cu(mg/L)',
    'ITEM_ABS' : 'abs(mg/L)',
    'ITEM_PCB' : 'pcb(mg/L)',
    'ITEM_OP' : '유기인(mg/L)',
    'ITEM_MN' : 'Mn(mg/L)',
    'ITEM_TRANS' : '투명도(mg/L)',
    'ITEM_CLOA' : 'clo-a(mg/L)',
    'ITEM_CL' : 'Cl(mg/L)',
    'ITEM_ZN' : 'Zn(mg/L)',
    'ITEM_CR' : 'Cr(mg/L)',
    'ITEM_FE' : 'Fe(mg/L)',
    'ITEM_PHENOL' : 'phenols(mg/L)',
    'ITEM_NHEX' : '노말헥산추출물질(mg/L)',
    'ITEM_EC' : 'EC(microS/cm)',
    'ITEM_TCE' : 'TCE(mg/L)',
    'ITEM_PCE' : 'PCE(mg/L)',
    'ITEM_NO3N' : 'NO3N(mg/L)',
    'ITEM_NH3N' : 'NH3N(mg/L)',
    'ITEM_ECOLI' : 'Ecoli(분원성대장균군수/100ml)',
    'ITEM_POP' : 'PO4P(mg/L)',
    'ITEM_DTN' : 'DTN(mg/L)',
    'ITEM_DTP' : 'DTP(mg/L)',
    'ITEM_FL' : 'F(mg/L)',
    'ITEM_COL' : '색도(도)',
    'ITEM_ALGOL' : 'ALGOL(mg/L)',
    'ITEM_CCL4' : 'CCl4(mg/L)',
    'ITEM_DCETH' : '1,2-다이클로로에탄(mg/L)',
    'ITEM_DCM' : '다이클로로메탄(mg/L)',
    'ITEM_BENZENE' : '벤젠(mg/L)',
    'ITEM_CHCL3' : '클로로포름(mg/L)',
    'ITEM_TOC' : 'TOC(mg/L)',
    'ITEM_DEHP' : 'DEHP(mg/L)',
    'ITEM_ANTIMON' : '안티몬(mg/L)',
    'ITEM_DIOX' : '1,4-다이옥세인(mg/L)',
    'ITEM_HCHO' : '포름알데히드(mg/L)',
    'ITEM_HCB' : 'HCB(mg/L)',
    'ITEM_NI' : 'Ni(mg/L)',
    'ITEM_BA' : 'Ba(mg/L)',
    'ITEM_SE' : 'Se(mg/L)',
    }


    wq_val_names = [
    '수온',
    'pH',
    'DO(mg/L)',
    'BOD(mg/L)',
    'COD(mg/L)',
    'SS(mg/L)',
    'TN(mg/L)',
    'TP(mg/L)',
    'clo-a(mg/L)',
    'EC(microS/cm)',
    'TOC(mg/L)',
    ]

    wq_gdf.rename(columns=col_names_dic, inplace=True)
    from copy import deepcopy
    filter_1_wq_data = deepcopy(wq_gdf.loc[:,['PT_NO', 'PT_NM', 'WMYR', 'WMOD', 'WMWK', 'WMCYMD','geometry', 'CAT_DID']+wq_val_names])
    filter_2_wq_data = filter_1_wq_data.loc[(pd.to_numeric(filter_1_wq_data.WMYR) <= 2020) & (pd.to_numeric(filter_1_wq_data.WMYR) >= 2015)]
    
    # 2015년 이후로 BOD 측정지점 970 절반만 남음 일단은 COD만 포함
    tem_cols = filter_2_wq_data.columns.to_list()
    tem_cols.remove('BOD(mg/L)')
    filter_3_wq_data = filter_2_wq_data.loc[:,tem_cols]
    filter_3_wq_data = filter_3_wq_data.dropna()
    filter_3_wq_data = filter_3_wq_data.applymap(lambda x: 0 if x == '정량한계미만' else x)
    filter_3_wq_data = filter_3_wq_data.astype(
        {
        'PT_NO':'string',
        'PT_NM':'string',
        'WMYR':'string',
        'WMOD':'string',
        'WMWK':'string',
        'WMCYMD':'string',
        'CAT_DID':'string',
        '수온':'float32',
        'pH':'float32',
        'DO(mg/L)':'float32',
        'COD(mg/L)':'float32',
        'SS(mg/L)':'float32',
        'TN(mg/L)':'float32',
        'TP(mg/L)':'float32',
        'clo-a(mg/L)':'float32',
        'EC(microS/cm)':'float32',
        'TOC(mg/L)':'float32', })

   
    mean_df = filter_3_wq_data.groupby('PT_NO').mean()
    q1_df = filter_3_wq_data.groupby('PT_NO').quantile([0.25]).set_index(mean_df.index)
    q1_df.columns = [name+"_q1" for name in q1_df.columns.to_list()]
    q3_df = filter_3_wq_data.groupby('PT_NO').quantile([0.75]).set_index(mean_df.index)
    q3_df.columns = [name+"_q3" for name in q3_df.columns.to_list()]

    filter_4_wq_data = pd.concat([mean_df,q1_df,q3_df], axis = 1)
    filter_4_wq_data.loc[:,'PT_NO'] = filter_4_wq_data.index
    filter_4_wq_data.index = range(len(filter_4_wq_data))

    filter_5_wq_data = pd.merge(filter_4_wq_data,filter_3_wq_data.drop_duplicates(['PT_NO']).loc[:,['PT_NO','PT_NM','geometry','CAT_DID',]],how='left', on = 'PT_NO')
    return filter_5_wq_data