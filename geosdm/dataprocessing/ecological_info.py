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


def reshape_eco_monitoring_data(data_path: str):
    ########## check! ##########
    category = '저서성 대형 무척추동물'  # 어류
    col_names = ['No','조사년도', '조사회차', '대권역', '중권역', '분류코드','조사지점', '학명', '국명', '개체밀도', 'category',] if category == '저서성 대형 무척추동물' else ['No','조사년도', '조사회차', '대권역', '중권역', '분류코드','조사지점', '학명', '국명', '개체밀도', 'category',]
    sp_raw_df_list = []
    sp_dataset_path_list = glob(os.path.join(data_path,"rawdata","생물","출현생물종_하천","*.xlsx",))

    #황박사님 매치 테이블 로드
    match_table = pd.read_csv(os.path.join(data_path,"rawdata","생물","sp_list_f.csv",), encoding='euc-kr')

    for i in sp_dataset_path_list:
        xl = pd.ExcelFile(i)
        sheet_names = xl.sheet_names

        for j in sheet_names:
            if j == category:
                temp_data = pd.read_excel(i, sheet_name= j)
                temp_data.loc[:,['category']] = j
                sp_raw_df_list.append(temp_data)
            else: pass
            # print(j,temp_data.columns)

    raw_data = pd.concat(sp_raw_df_list, axis = 0)
    raw_data.index = range(len(raw_data))
    raw_data.columns = ['No','조사년도', '조사회차', '대권역', '중권역', '분류코드','조사지점', '학명', '국명', '개체밀도', 'category',]

    # raw_data 중 이름 뒤에 스페이스 있는 경우 있으므로 제거
    def space_cut(nam):
        if nam[-1] == ' ':
            nam = nam[:-1]
        else:
            nam = nam
        return nam

    new_name = [space_cut(name) for name in raw_data.학명.values.tolist()]
    raw_data.학명 = new_name


    # raw_data, match_table merge
    merged_1_df = pd.merge(raw_data,match_table,how='left', left_on='학명', right_on='학명')
    # 원하는 수준으로 재정렬
    group_level = ['Order']
    base_col = ['조사년도', '조사회차', '대권역', '중권역','조사지점','개체밀도']

    target_df = merged_1_df.loc[:,base_col + group_level]

    year_list = list(set(target_df.조사년도.values))
    year_list.sort()
    n_list = list(set(target_df.조사회차.values))
    n_list.sort()

    ind_df_list = []
    for y in year_list:
        for n in n_list:
            temp_df = target_df.loc[(target_df.조사년도 == y) & (target_df.조사회차 == n)]
            temp_site_list = list(set(temp_df.조사지점.values))
            for s_n in temp_site_list:
                t_grouped_df = temp_df.loc[temp_df.조사지점 == s_n].groupby(group_level[0]).sum()
                names = t_grouped_df.index.tolist()
                nums = t_grouped_df.loc[:,'개체밀도'].values.tolist()

                cols = ['조사년도', '조사회차', '대권역', '중권역','조사지점'] + names
                vals = [y,n]+ temp_df.loc[temp_df.조사지점 == s_n,['대권역', '중권역']].iloc[0,:].values.tolist() +[s_n] + nums
                ind_df_list.append(pd.DataFrame([vals], columns= cols))


    reshaped_df = pd.concat(ind_df_list)
    # reshaped_df.to_csv('학명.csv', encoding='euc-kr')

    # 지점정보 매칭
    # 생물측정망, 수질, 퇴적물측정망 shp 불러오기
    shp_list_path = glob(os.path.join(data_path,"rawdata","생물","shp-catchment","*.shp",))
    shp_list = [gpd.GeoDataFrame.from_file(path_shp,encoding = 'cp949') for path_shp in shp_list_path]
    main_shp = pd.concat([shp_list[0], shp_list[1]], axis=0)
    main_shp.index = range(len(main_shp))

    shp_list2 = [gpd.GeoDataFrame.from_file(path_shp,encoding = 'cp949') for path_shp in glob(os.path.join(data_path,"pro1shp","CaseStudy_Result (2)","shp","*.shp",))]
    base_shp = shp_list2[7]

    P5181 = Proj(init='epsg:5181')
    P4326 = Proj(init='epsg:4326')
    new_point_list = [Point(transform(P4326, P5181, x, y)) for x,y in np.concatenate([main_shp.geometry.x.values.reshape(-1,1), main_shp.geometry.y.values.reshape(-1,1)],axis = 1).tolist()]
    main_shp.geometry = new_point_list
    main_shp.crs = {'init':'epsg:5181'}
    main_shp = main_shp.to_crs(base_shp.crs)


    # 병합
    merge_point_df = pd.merge(reshaped_df,main_shp, how='left', left_on='조사지점', right_on='조사지점명')
    merge_point_df = gpd.GeoDataFrame(merge_point_df, geometry= 'geometry')

    # # 매칭안된지점 정보 및 생물측정망 정보 저장
    # merge_point_df.loc[merge_point_df.geometry.isna()].to_csv(os.path.join(data_path,"nomatch_point.csv",), encoding='euc-kr')
    # main_shp.to_csv(os.path.join(data_path,"biopoint_info.csv",), encoding='euc-kr')

    #매칭안된 지점 제거
    merge_point_df = merge_point_df.loc[ ~ merge_point_df.geometry.isna()]

    # # 속하는 폴리컨 매칭
    wdf_path = glob(os.path.join(data_path,"rawdata","생물","평가항목점수","*.xlsx",))
    wdf = pd.read_excel(wdf_path[0])

    merge_point_df = pd.merge(merge_point_df,wdf, how='left', left_on=['조사년도','조사회차','조사지점'], right_on=['연도','회차','조사구간명'])
    merge_point_df.index = range(len(merge_point_df))


    # # test.loc[ ~ test.geometry.isna()]
    # for i in range(len(merge_point_df)):
    #     try:
    #         merge_point_df.loc[i,['SBSNCD','SBSNNM','BBSNCD','MBSNCD']] = base_shp[base_shp.contains(merge_point_df.geometry[i])].loc[:,['SBSNCD', 'SBSNNM', 'BBSNCD', 'MBSNCD']].values[0]
    #     except:
    #         # 포함되는 영역이 없는 지점
    #         print(merge_point_df.loc[i,['조사년도','조사회차','조사지점']].tolist())

    # merge_point_df.to_csv(group_level[0] + '.csv', encoding='euc-kr')
    for_subbmatch_df = merge_point_df.drop_duplicates(subset = 'geometry')
    for_subbmatch_df.index = range(len(for_subbmatch_df))
    for i in range(len(for_subbmatch_df)):
        try:
            merge_point_df.loc[merge_point_df.조사지점 == for_subbmatch_df.조사지점[i],['SBSNCD','SBSNNM','BBSNCD','MBSNCD']] = base_shp[base_shp.contains(for_subbmatch_df.loc[i,'geometry'])].loc[:,['SBSNCD', 'SBSNNM', 'BBSNCD', 'MBSNCD']].values[0]

        except:
            # 포함되는 영역이 없는 지점
            print(merge_point_df.loc[i,['조사년도','조사회차','조사지점']].tolist())

    # merge_point_df.to_csv(os.path.join(data_path,"merged","merged_1",group_level[0]+".csv",), encoding='euc-kr')
    pickle.dump(merge_point_df, open(os.path.join(data_path,"merged","merged_1",group_level[0]+"_gdf.p",), "wb"))
    # test = pickle.load(open(os.path.join(data_path,"merged","merged_1","sp_gdf.p",), "rb")) 
    # 원하는 수준으로 재정렬
    group_level = ['Family']
    base_col = ['조사년도', '조사회차', '대권역', '중권역','조사지점','개체밀도']

    target_df = merged_1_df.loc[:,base_col + group_level]

    year_list = list(set(target_df.조사년도.values))
    year_list.sort()
    n_list = list(set(target_df.조사회차.values))
    n_list.sort()

    ind_df_list = []
    for y in year_list:
        for n in n_list:
            temp_df = target_df.loc[(target_df.조사년도 == y) & (target_df.조사회차 == n)]
            temp_site_list = list(set(temp_df.조사지점.values))
            for s_n in temp_site_list:
                t_grouped_df = temp_df.loc[temp_df.조사지점 == s_n].groupby(group_level[0]).sum()
                names = t_grouped_df.index.tolist()
                nums = t_grouped_df.loc[:,'개체밀도'].values.tolist()

                cols = ['조사년도', '조사회차', '대권역', '중권역','조사지점'] + names
                vals = [y,n]+ temp_df.loc[temp_df.조사지점 == s_n,['대권역', '중권역']].iloc[0,:].values.tolist() +[s_n] + nums
                ind_df_list.append(pd.DataFrame([vals], columns= cols))


    reshaped_df = pd.concat(ind_df_list)
    # reshaped_df.to_csv('학명.csv', encoding='euc-kr')

    # 지점정보 매칭
    # 생물측정망, 수질, 퇴적물측정망 shp 불러오기
    shp_list_path = glob(os.path.join(data_path,"rawdata","생물","shp-catchment","*.shp",))
    shp_list = [gpd.GeoDataFrame.from_file(path_shp,encoding = 'cp949') for path_shp in shp_list_path]
    main_shp = pd.concat([shp_list[0], shp_list[1]], axis=0)
    main_shp.index = range(len(main_shp))

    shp_list2 = [gpd.GeoDataFrame.from_file(path_shp,encoding = 'cp949') for path_shp in glob(os.path.join(data_path,"pro1shp","CaseStudy_Result (2)","shp","*.shp",))]
    base_shp = shp_list2[7]

    P5181 = Proj(init='epsg:5181')
    P4326 = Proj(init='epsg:4326')
    new_point_list = [Point(transform(P4326, P5181, x, y)) for x,y in np.concatenate([main_shp.geometry.x.values.reshape(-1,1), main_shp.geometry.y.values.reshape(-1,1)],axis = 1).tolist()]
    main_shp.geometry = new_point_list
    main_shp.crs = {'init':'epsg:5181'}
    main_shp = main_shp.to_crs(base_shp.crs)


    # 병합
    merge_point_df = pd.merge(reshaped_df,main_shp, how='left', left_on='조사지점', right_on='조사지점명')
    merge_point_df = gpd.GeoDataFrame(merge_point_df, geometry= 'geometry')

    # # 매칭안된지점 정보 및 생물측정망 정보 저장
    # merge_point_df.loc[merge_point_df.geometry.isna()].to_csv(os.path.join(data_path,"nomatch_point.csv",), encoding='euc-kr')
    # main_shp.to_csv(os.path.join(data_path,"biopoint_info.csv",), encoding='euc-kr')

    #매칭안된 지점 제거
    merge_point_df = merge_point_df.loc[ ~ merge_point_df.geometry.isna()]

    # # 속하는 폴리컨 매칭
    wdf_path = glob(os.path.join(data_path,"rawdata","생물","평가항목점수","*.xlsx",))
    wdf = pd.read_excel(wdf_path[0])

    merge_point_df = pd.merge(merge_point_df,wdf, how='left', left_on=['조사년도','조사회차','조사지점'], right_on=['연도','회차','조사구간명'])
    merge_point_df.index = range(len(merge_point_df))


    # # test.loc[ ~ test.geometry.isna()]
    # for i in range(len(merge_point_df)):
    #     try:
    #         merge_point_df.loc[i,['SBSNCD','SBSNNM','BBSNCD','MBSNCD']] = base_shp[base_shp.contains(merge_point_df.geometry[i])].loc[:,['SBSNCD', 'SBSNNM', 'BBSNCD', 'MBSNCD']].values[0]
    #     except:
    #         # 포함되는 영역이 없는 지점
    #         print(merge_point_df.loc[i,['조사년도','조사회차','조사지점']].tolist())

    # merge_point_df.to_csv(group_level[0] + '.csv', encoding='euc-kr')
    for_subbmatch_df = merge_point_df.drop_duplicates(subset = 'geometry')
    for_subbmatch_df.index = range(len(for_subbmatch_df))
    for i in range(len(for_subbmatch_df)):
        try:
            merge_point_df.loc[merge_point_df.조사지점 == for_subbmatch_df.조사지점[i],['SBSNCD','SBSNNM','BBSNCD','MBSNCD']] = base_shp[base_shp.contains(for_subbmatch_df.loc[i,'geometry'])].loc[:,['SBSNCD', 'SBSNNM', 'BBSNCD', 'MBSNCD']].values[0]

        except:
            # 포함되는 영역이 없는 지점
            print(merge_point_df.loc[i,['조사년도','조사회차','조사지점']].tolist())

    # merge_point_df.to_csv(os.path.join(data_path,"merged","merged_1",group_level[0]+".csv",), encoding='euc-kr')
    pickle.dump(merge_point_df, open(os.path.join(data_path,"merged","merged_1",group_level[0]+"_gdf.p",), "wb"))
    # test = pickle.load(open(os.path.join(data_path,"merged","merged_1","sp_gdf.p",), "rb")) 
    # 원하는 수준으로 재정렬
    group_level = ['학명']
    base_col = ['조사년도', '조사회차', '대권역', '중권역','조사지점','개체밀도']

    target_df = merged_1_df.loc[:,base_col + group_level]

    year_list = list(set(target_df.조사년도.values))
    year_list.sort()
    n_list = list(set(target_df.조사회차.values))
    n_list.sort()

    ind_df_list = []
    for y in year_list:
        for n in n_list:
            temp_df = target_df.loc[(target_df.조사년도 == y) & (target_df.조사회차 == n)]
            temp_site_list = list(set(temp_df.조사지점.values))
            for s_n in temp_site_list:
                t_grouped_df = temp_df.loc[temp_df.조사지점 == s_n].groupby(group_level[0]).sum()
                names = t_grouped_df.index.tolist()
                nums = t_grouped_df.loc[:,'개체밀도'].values.tolist()

                cols = ['조사년도', '조사회차', '대권역', '중권역','조사지점'] + names
                vals = [y,n]+ temp_df.loc[temp_df.조사지점 == s_n,['대권역', '중권역']].iloc[0,:].values.tolist() +[s_n] + nums
                ind_df_list.append(pd.DataFrame([vals], columns= cols))


    reshaped_df = pd.concat(ind_df_list)
    # reshaped_df.to_csv('학명.csv', encoding='euc-kr')

    # 지점정보 매칭
    # 생물측정망, 수질, 퇴적물측정망 shp 불러오기
    shp_list_path = glob(os.path.join(data_path,"rawdata","생물","shp-catchment","*.shp",))
    shp_list = [gpd.GeoDataFrame.from_file(path_shp,encoding = 'cp949') for path_shp in shp_list_path]
    main_shp = pd.concat([shp_list[0], shp_list[1]], axis=0)
    main_shp.index = range(len(main_shp))

    shp_list2 = [gpd.GeoDataFrame.from_file(path_shp,encoding = 'cp949') for path_shp in glob(os.path.join(data_path,"pro1shp","CaseStudy_Result (2)","shp","*.shp",))]
    base_shp = shp_list2[7]

    P5181 = Proj(init='epsg:5181')
    P4326 = Proj(init='epsg:4326')
    new_point_list = [Point(transform(P4326, P5181, x, y)) for x,y in np.concatenate([main_shp.geometry.x.values.reshape(-1,1), main_shp.geometry.y.values.reshape(-1,1)],axis = 1).tolist()]
    main_shp.geometry = new_point_list
    main_shp.crs = {'init':'epsg:5181'}
    main_shp = main_shp.to_crs(base_shp.crs)


    # 병합
    merge_point_df = pd.merge(reshaped_df,main_shp, how='left', left_on='조사지점', right_on='조사지점명')
    merge_point_df = gpd.GeoDataFrame(merge_point_df, geometry= 'geometry')

    # # 매칭안된지점 정보 및 생물측정망 정보 저장
    # merge_point_df.loc[merge_point_df.geometry.isna()].to_csv(os.path.join(data_path,"nomatch_point.csv",), encoding='euc-kr')
    # main_shp.to_csv(os.path.join(data_path,"biopoint_info.csv",), encoding='euc-kr')

    #매칭안된 지점 제거
    merge_point_df = merge_point_df.loc[ ~ merge_point_df.geometry.isna()]

    # # 속하는 폴리컨 매칭
    wdf_path = glob(os.path.join(data_path,"rawdata","생물","평가항목점수","*.xlsx",))
    wdf = pd.read_excel(wdf_path[0])

    merge_point_df = pd.merge(merge_point_df,wdf, how='left', left_on=['조사년도','조사회차','조사지점'], right_on=['연도','회차','조사구간명'])
    merge_point_df.index = range(len(merge_point_df))


    # # test.loc[ ~ test.geometry.isna()]
    # for i in range(len(merge_point_df)):
    #     try:
    #         merge_point_df.loc[i,['SBSNCD','SBSNNM','BBSNCD','MBSNCD']] = base_shp[base_shp.contains(merge_point_df.geometry[i])].loc[:,['SBSNCD', 'SBSNNM', 'BBSNCD', 'MBSNCD']].values[0]
    #     except:
    #         # 포함되는 영역이 없는 지점
    #         print(merge_point_df.loc[i,['조사년도','조사회차','조사지점']].tolist())

    # merge_point_df.to_csv(group_level[0] + '.csv', encoding='euc-kr')
    for_subbmatch_df = merge_point_df.drop_duplicates(subset = 'geometry')
    for_subbmatch_df.index = range(len(for_subbmatch_df))
    for i in range(len(for_subbmatch_df)):
        try:
            merge_point_df.loc[merge_point_df.조사지점 == for_subbmatch_df.조사지점[i],['SBSNCD','SBSNNM','BBSNCD','MBSNCD']] = base_shp[base_shp.contains(for_subbmatch_df.loc[i,'geometry'])].loc[:,['SBSNCD', 'SBSNNM', 'BBSNCD', 'MBSNCD']].values[0]

        except:
            # 포함되는 영역이 없는 지점
            print(merge_point_df.loc[i,['조사년도','조사회차','조사지점']].tolist())

    # merge_point_df.to_csv(os.path.join(data_path,"merged","merged_1",group_level[0]+".csv",), encoding='euc-kr')
    pickle.dump(merge_point_df, open(os.path.join(data_path,"merged","merged_1",group_level[0]+"_gdf.p",), "wb"))
    # test = pickle.load(open(os.path.join(data_path,"merged","merged_1","sp_gdf.p",), "rb")) 



def resample_ecological_data(sp_gdf):
    # 퇴적물데이터 수질측정망이라는 컬럼있음 어디서 들어온건지 확인하고 체크하기(잘못병합됐을 수 있음)
    # cate_level = 'name'  #'Family', 'name','Order'
    # sp_gdf = Order_gdf if cate_level == 'Order' else Family_gdf if cate_level == 'Family' else name_gdf

    remove_col = {'조사년도',
    '조사회차',
    '대권역_x',
    '중권역_x',
    '조사지점',
    'field_1',
    '조사지점명',
    '분류번호',
    '대권역(No)',
    '대권역_y',
    '중권역(No)',
    '중권역_y',
    '주소',
    'N (도분초)',
    'E (도분초)',
    '조사기관',
    'WGS84_Y',
    'WGS84_X',
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
    'CAT_FLAG',
    'geometry',
    '수계',
    '연도',
    '회차',
    '분류코드',
    '조사구간명',
    '종횡사주_횟수_평가 ',
    '자연성_정도_평가 ',
    '유속_다양성_평가',
    '하천변_폭_평가 ',
    '저수로_하안_평가 ',
    '하안_재료_평가 ',
    '저질_상태_평가 ',
    '횡구조물_방해_평가 ',
    '제외지_토지이용_평가 ',
    '제내지_토지이용_평가',
    '조사불가 특이사항',
    'SBSNCD',
    'SBSNNM',
    'BBSNCD',
    'MBSNCD'
    }

    geo_col = ['종횡사주_횟수_평가 ',
            '자연성_정도_평가 ',
            '유속_다양성_평가',
            '하천변_폭_평가 ',
            '저수로_하안_평가 ',
            '하안_재료_평가 ',
            '저질_상태_평가 ',
            '횡구조물_방해_평가 ',
            '제외지_토지이용_평가 ',
            '제내지_토지이용_평가',]

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
                '조사지점',
                'geometry'] + geo_col
    
    all_col = set(sp_gdf.columns)
    remain_col = list(all_col - remove_col) + ['조사년도']

    site_list = list(set(sp_gdf.조사지점.values))
    df_bysite_list = []
    for s_name in site_list:
        t_df = sp_gdf.loc[sp_gdf.조사지점 == s_name, remain_col].set_index('조사년도').resample(rule = '1Y').sum()
        t_df.loc[:,after_add] = np.array(sp_gdf.loc[sp_gdf.조사지점 == s_name, after_add].iloc[0].tolist(), dtype = object)
        df_bysite_list.append(t_df)

    sp_df_resampled = pd.concat(df_bysite_list)
    sp_df_resampled.loc[:,'년'] = sp_df_resampled.index.year.tolist()
    sp_df_resampled.index = range(len(sp_df_resampled))
    # 2015~2020 3번이상 출현이면 출현
    filter_1_sp_df = sp_df_resampled.loc[(sp_df_resampled.년 >= 2015) & (sp_df_resampled.년 <= 2020), :]
    filter_2_sp_df = filter_1_sp_df.drop(geo_col, axis=1).groupby(['조사지점','년']).min().groupby('조사지점').sum()
    filter_3_sp_df = filter_2_sp_df.applymap(lambda x: 0 if x < 2 else 1)

    ### 조사지점 조사정보?는 평균으로
    geo_info_resmaple_df = filter_1_sp_df.loc[:,['조사지점','년'] + geo_col].groupby('조사지점').mean()
    geo_info_resmaple_df.loc[:,'조사지점'] = geo_info_resmaple_df.index
    geo_info_resmaple_df.index = range(len(geo_info_resmaple_df))
    geo_info_resmaple_df = geo_info_resmaple_df.loc[:, ['조사지점'] + geo_col]

    filter_3_sp_df = pd.merge(filter_3_sp_df, geo_info_resmaple_df, how='left', on='조사지점')

    tmp_cols = filter_3_sp_df.columns.tolist()
    for i in ['OBJECTID',
    'BRU_X',
    'BRU_Y',
    'BLL_X',
    'BLL_Y',
    'Shape_Leng',
    'Shape_Area',
    'streamname']:

        tmp_cols.remove(i)

    filter_4_sp_df = filter_3_sp_df.loc[:,tmp_cols]
    filter_4_sp_df.index = range(len(filter_4_sp_df))
    filter_5_sp_df = pd.merge(filter_4_sp_df, filter_1_sp_df.drop_duplicates('조사지점').loc[:,['CAT_DID', '조사지점', 'geometry']], how='left', on='조사지점')

    return filter_5_sp_df
