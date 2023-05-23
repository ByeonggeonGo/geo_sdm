import pandas as pd
from glob import glob
import os
import numpy as np
import geopandas as gpd
import pickle

def merge_hyd_monitoring_n_coords(data_path: str):
    col_dict = {'관측기기': '관측기기',
            '관측소명': '관측소명',
            '단 면 적': '단면적',
            '번호': '번호',
            '비      고': '비고',
            '수  위': '수위',
            '수 계 명': '수계명',
            '수 면 폭': '수면폭',
            '수위변화': '수위변화',
            '유   량': '유량',
            '측선수': '측선수',
            '측정년.월.일.시': '측정시각',
            '측정시각': '측정시각',
            '측정장비': '관측기기',
            '평균수심': '평균수심',
            '평균유속': '평균유속'}

    # col_ind = ['수계명', '관측소명', '측정시각', '수위', '수면폭', '단면적', '평균유속', '평균수심', '유량',]
    col_ind = ['수계명', '관측소명', '측정시각', '수면폭', '단면적', '평균유속', '평균수심', '유량',]
    dty_info = {
        '수계명' : 'string', 
        '관측소명': 'string', 
        '수면폭': 'float', 
        '단면적': 'float', 
        '평균유속': 'float', 
        '평균수심': 'float', 
        '유량': 'float',
    }

    check_c_list = ['수면폭', '단면적', '평균유속', '평균수심', '유량']

    hyd_dataset_path_list = glob(os.path.join(data_path,"rawdata","수리수문","filtered_1","*.xls",))
    hyd_dataset_path_list.extend(glob(os.path.join(data_path,"rawdata","수리수문","filtered_1","*.xlsx",)))
    hyd_dataset_list = [pd.read_excel(path,header=[3] ) if 'Unnamed: 1' in pd.read_excel(path, ).columns else pd.read_excel(path, ) for path in hyd_dataset_path_list]

    for i in range(len(hyd_dataset_list)):
        #컬럼 헤더 정리
        if 'Unnamed: 1' in hyd_dataset_list[i].columns:
            hyd_dataset_list[i] = pd.read_excel(hyd_dataset_path_list[i],header=[5] )
        
        #컬럼명 통일
        cols = []
        for k in hyd_dataset_list[i].columns:
            cols.append(col_dict[k.split('\n')[0]])
        hyd_dataset_list[i].columns = cols
        
        #필요없는 컬럼 제거 및 컬럼순서 통일
        hyd_dataset_list[i] = hyd_dataset_list[i][col_ind]

        #결측치 바로 이전행 값으로 대체, 수게명, 관측소명 통일 목적
        #이외의 컬럼에 대한 결측치또한 이전값으로 대체될 것이므로 미리 제거하는 등의 작업 고려해보기, 
        #확인결과 간혹 수위에 결측치 있음 평균수심은 결측치 없으므로 수위 제거해도  될듯
        hyd_dataset_list[i] = hyd_dataset_list[i].fillna(method='ffill')

        #자료형 통일
        for j in check_c_list:
            if hyd_dataset_list[i][j].dtype == 'object':
                #자료에 포함된 ","랑 "?"제거 
                hyd_dataset_list[i][j] = hyd_dataset_list[i][j].str.replace(',','',regex=True).str.replace('?','',regex=True).astype(float)
        
        if hyd_dataset_list[i]['측정시각'].dtype == 'object' or hyd_dataset_list[i]['측정시각'].dtype == 'str':
            # 중간중간 datetime 섞여있는것 일단 string으로 통일한후 한번에 datetime으로변경(안그러면 중간에 nan값 반환됨)
            hyd_dataset_list[i]['측정시각'] = hyd_dataset_list[i]['측정시각'].map(lambda x : str(x))
            for q in range(len(hyd_dataset_list[i])):
                hyd_dataset_list[i]['측정시각'] = hyd_dataset_list[i]['측정시각'].str.replace('.','-',regex=True)
                if len(hyd_dataset_list[i]['측정시각'][q].split('-')[0]) ==2:
                    hyd_dataset_list[i]['측정시각'][q] = '20' + hyd_dataset_list[i]['측정시각'][q]
                elif len(hyd_dataset_list[i]['측정시각'][q].split('-')[0]) !=4:
                    hyd_dataset_list[i]['측정시각'][q] = hyd_dataset_list[i]['측정시각'][q-1].split('-')[0] + '-' + hyd_dataset_list[i]['측정시각'][q].split('-')[1] + '-' + hyd_dataset_list[i]['측정시각'][q].split('-')[2]


        # hyd_dataset_list[i]['측정시각'] = pd.to_datetime(hyd_dataset_list[i]['측정시각'], format='%Y-%m-%d')
        hyd_dataset_list[i]['측정시각'] = pd.to_datetime(hyd_dataset_list[i]['측정시각'])
        hyd_dataset_list[i] = hyd_dataset_list[i].astype(dty_info)
    

    hyd_df_merged = pd.concat(hyd_dataset_list,axis=0)

    xy_dataset_path_list = glob(os.path.join(data_path,"rawdata","수리수문","filtered_2","*",))
    xy_dataset_list = [pd.read_excel(path,header=[2] ) if 'Unnamed: 1' in pd.read_excel(path, ).columns else pd.read_excel(path, ) for path in xy_dataset_path_list]

    for i in range(len(xy_dataset_list)):
        #컬럼 헤더 정리
        if 'Unnamed: 1' in xy_dataset_list[i].columns:
            xy_dataset_list[i] = pd.read_excel(xy_dataset_path_list[i],header=[3] )
        else:
            xy_dataset_list[i] = pd.read_excel(xy_dataset_path_list[i],header=[2,3,4,5] )

    
    #### x,y 좌표나와있는 컬럼 서치 및 좌표 안나온 데이터셋 제거 ####
    #### 좌표계 참고 WGS84 ####
    x_col_list = []
    xy_na_set_list = []
    for i in range(len(xy_dataset_list)):
        # print(xy_dataset_list[i].iloc[:,8:10].head())
        val_check = False
        for ind,k in enumerate(xy_dataset_list[i].columns):
            try:
                if '126' in xy_dataset_list[i][k][0] or '127' in xy_dataset_list[i][k][0] or '128' in xy_dataset_list[i][k][0]:
                    x_col_list.append(ind)
                    val_check = True
            except:
                pass
        if not val_check:
            xy_na_set_list.append(False)
        if val_check:
            xy_na_set_list.append(True)

    #좌표 안나와있는 데이터 제거
    filt_xy_dataset_list = []
    for i, val in enumerate(xy_na_set_list):
        if val:
            filt_xy_dataset_list.append(xy_dataset_list[i])
    ##### 좌표 지정컬럼 잘 찾나 확인
    # for i, indval in enumerate(x_col_list):
    #     print(filt_xy_dataset_list[i].iloc[0,indval])
    #     print(filt_xy_dataset_list[i].iloc[0,indval+1])


    ### 관측소명 - 좌표 데이터프레임 생성
    new_xy_dataset = {
        '수계명': [],
        '관측소명': [],
        'x': [],
        'y': [],
    }

    for i in range(len(filt_xy_dataset_list)):
        xind = x_col_list[i]
        yind = xind + 1
        wb_names = filt_xy_dataset_list[i].loc[:,['수 계 명']].values.tolist()
        wb_names = np.array(wb_names).reshape(1,-1)[0].tolist()
        names = filt_xy_dataset_list[i].loc[:,['관 측 소 명']].values.tolist()
        names = np.array(names).reshape(1,-1)[0].tolist()
        x = filt_xy_dataset_list[i].iloc[:,xind].values.tolist()
        y = filt_xy_dataset_list[i].iloc[:,yind].values.tolist()
        new_xy_dataset['수계명'].extend(wb_names)
        new_xy_dataset['관측소명'].extend(names)
        new_xy_dataset['x'].extend(x)
        new_xy_dataset['y'].extend(y)

    new_xy_df = pd.DataFrame(new_xy_dataset)
    new_xy_df = new_xy_df.drop_duplicates(keep='first',)

    # 스트링 다르게 입력되어있는 것 수정
    new_xy_df['x'] = new_xy_df['x'].apply(lambda x : x[0:3] + '-' + x[4:6] + '-' + x[7:9])
    new_xy_df['y'] = new_xy_df['y'].apply(lambda x : x[0:2] + '-' + x[3:5] + '-' + x[6:8])

    def dms_to_dd(dms_str):
        dms_list = dms_str.split('-')
        dd = float(dms_list[0]) + float(dms_list[1])/60 + float(dms_list[2])/3600
        return dd

    new_xy_df['x'] = new_xy_df['x'].apply(lambda x : dms_to_dd(x))
    new_xy_df['y'] = new_xy_df['y'].apply(lambda x : dms_to_dd(x))

    ### 직접 좌표찾은 표 추가 ###
    ###########################
    menual_xy_dataset_path_list = glob(os.path.join(data_path,"rawdata","수리수문","filtered_2","*좌표찾음.xls",))
    menual_xy_dataset_list = [pd.read_excel(path,header=[2] ) if 'Unnamed: 1' in pd.read_excel(path, ).columns else pd.read_excel(path, ) for path in menual_xy_dataset_path_list]

    for i in range(len(menual_xy_dataset_list)):
        #컬럼 헤더 정리
        if 'Unnamed: 1' in menual_xy_dataset_list[i].columns:
            menual_xy_dataset_list[i] = pd.read_excel(menual_xy_dataset_path_list[i],header=[3] )
        else:
            menual_xy_dataset_list[i] = pd.read_excel(menual_xy_dataset_path_list[i],header=[2,3,4,5])

    menual_xy_dataset = {
        '수계명': [],
        '관측소명': [],
        'x': [],
        'y': [],
    }

    for i in range(len(menual_xy_dataset_list)):
        xind = 11
        yind = xind + 1
        wb_names = menual_xy_dataset_list[i].loc[:,['수 계 명']].values.tolist()
        wb_names = np.array(wb_names).reshape(1,-1)[0].tolist()
        names = menual_xy_dataset_list[i].loc[:,['관 측 소 명']].values.tolist()
        names = np.array(names).reshape(1,-1)[0].tolist()
        x = menual_xy_dataset_list[i].iloc[:,xind].values.tolist()
        y = menual_xy_dataset_list[i].iloc[:,yind].values.tolist()
        menual_xy_dataset['수계명'].extend(wb_names)
        menual_xy_dataset['관측소명'].extend(names)
        menual_xy_dataset['x'].extend(x)
        menual_xy_dataset['y'].extend(y)

    menual_xy_df = pd.DataFrame(menual_xy_dataset)
    menual_xy_df = menual_xy_df.drop_duplicates(keep='first',)
    #########################
    #########################

    new_xy_df = pd.concat([new_xy_df,menual_xy_df], axis=0)

    new_xy_df = new_xy_df.astype({'수계명': 'string','관측소명': 'string','x': 'float','y': 'float',})
    new_xy_df.index = range(len(new_xy_df))

    from pyproj import Proj, transform

    P5181 = Proj(init='epsg:5181')
    P4326 = Proj(init='epsg:4326')

    geo_point_x = []
    geo_point_y = []

    for i in range(len(new_xy_df)):
        # geo_point.append(Point(new_xy_df.loc[i,'x'],new_xy_df.loc[i,'y']))
        x,y = transform(P4326, P5181, new_xy_df.loc[i,'x'], new_xy_df.loc[i,'y'])
        geo_point_x.append(x)
        geo_point_y.append(y)

    new_xy_df['x'] = geo_point_x
    new_xy_df['y'] = geo_point_y

    geo_point = []

    for i in range(len(new_xy_df)):
        geo_point.append(Point(new_xy_df.loc[i,'x'],new_xy_df.loc[i,'y']))
        

    new_xy_df['geometry'] = geo_point

    hyd_point_gdf = gpd.GeoDataFrame(new_xy_df, geometry= 'geometry')

    #################################################################
    ### '밀양A', '천안A' 수정
    ### '밀양A'는 좌표 3개 찍히는데 위에 2개는 바다에 찍혀서 제대로 찍히는 점으로 사용
    ###  '천안A' 바다에 찍히는데 주소로 직접 찾은 좌표로 교체 127.0901    36.7751  (2020년 국립환경과학원 연보)
    if len(hyd_point_gdf.loc[hyd_point_gdf.관측소명 == '밀양A'].index.values) > 1:
        hyd_point_gdf = hyd_point_gdf.drop(hyd_point_gdf.loc[hyd_point_gdf.관측소명 == '밀양A'].index.values[:-1].tolist())
    P5181 = Proj(init='epsg:5181')
    P4326 = Proj(init='epsg:4326')
    x,y = transform(P4326, P5181, 127.0901, 36.7751)
    hyd_point_gdf.loc[hyd_point_gdf.loc[hyd_point_gdf.관측소명 == '천안A'].index.tolist(),['x','y', 'geometry']] = [x,y,Point(x,y)]
    hyd_point_gdf.index = range(len(hyd_point_gdf))
    #################################################################


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


    ### 포인트 속하는 코드 부여

    for i in range(len(hyd_point_gdf)):
        search_rs = cat_did_shp[cat_did_shp.contains(hyd_point_gdf.geometry[i])]
        try:
            # hyd_point_gdf.loc[i,'SBSNCD'] = search_rs.iloc[0,0]
            # hyd_point_gdf.loc[i,'SBSNNM'] = search_rs.iloc[0,1]
            hyd_point_gdf.loc[i,cat_did_shp.columns.tolist()[:-1]] = search_rs.iloc[0,:-1]
        except:
            pass
    # hyd_point_gdf = hyd_point_gdf[['수계명','관측소명', 'x', 'y', 'SBSNCD', 'SBSNNM', 'geometry']].astype({'SBSNCD':'string','SBSNNM':'string'})

    shp_list = [gpd.GeoDataFrame.from_file(path_shp,encoding = 'cp949') for path_shp in glob(os.path.join(data_path,"pro1shp","CaseStudy_Result (2)","shp","*.shp",))]
    base_shp = shp_list[7]

    hyd_point_gdf.crs = {'init':'epsg:5181'}
    hyd_point_gdf = hyd_point_gdf.to_crs(base_shp.crs)

    ### 유량데이터에 지점정보(해당집수구역) 병합 ###
    ### 어느순간부터 수계명하고 관측소명이 바뀌는 행들 나옴 데이터셋 보고 고치기 ###
    ### '밀양A', '천안A' 수정
    ### '밀양A'는 좌표 3개 찍히는데 위에 2개는 바다에 찍혀서 제대로 찍히는 점으로 사용
    ###  '천안A' 바다에 찍히는데 주소로 직접 찾은 좌표로 교체 127.0901    36.7751  (2020년 국립환경과학원 연보)
    na_list = []
    hyd_df_merged.index = range(len(hyd_df_merged))
    for i in range(len(hyd_df_merged)):
    # for i in range(4):
        name_p = hyd_df_merged.loc[i,['관측소명']].values[0]
        name_wb_p = hyd_df_merged.loc[i,['수계명']].values[0]
        # print(name_p)
        # print(hyd_point_gdf.loc[(hyd_point_gdf.관측소명 == name_p) & (hyd_point_gdf.수계명 == name_wb_p),:])
        # print(hyd_point_gdf.loc[(hyd_point_gdf.관측소명 == name_p) & (hyd_point_gdf.수계명 == name_wb_p),['x','y','SBSNCD','SBSNNM']].values)
        ##### 포인트 중복있는 경우 일단 첫번쨰 포인트로 사용
        try:
            # hyd_df_merged.loc[i,hyd_point_gdf.columns.tolist()[2:]] = hyd_point_gdf.loc[(hyd_point_gdf.관측소명 == name_p) & (hyd_point_gdf.수계명 == name_wb_p),hyd_point_gdf.columns.tolist()[2:]].values.tolist()[0]
            # np.array(hyd_point_gdf.loc[(hyd_point_gdf.관측소명 == name_p) & (hyd_point_gdf.수계명 == name_wb_p),hyd_point_gdf.columns.tolist()[2:]].values.tolist()[0],dtype=object)
            hyd_df_merged.loc[i,hyd_point_gdf.columns.tolist()[2:]] = np.array(hyd_point_gdf.loc[(hyd_point_gdf.관측소명 == name_p) & (hyd_point_gdf.수계명 == name_wb_p),hyd_point_gdf.columns.tolist()[2:]].values.tolist()[0],dtype=object)
        
        except:
            try:
                hyd_df_merged.loc[i,hyd_point_gdf.columns.tolist()[2:]] = np.array(hyd_point_gdf.loc[(hyd_point_gdf.관측소명 == name_wb_p) & (hyd_point_gdf.수계명 == name_p),hyd_point_gdf.columns.tolist()[2:]].values.tolist()[0],dtype=object)
                hyd_df_merged.loc[i,['수계명', '관측소명']] = hyd_df_merged.loc[i,['관측소명', '수계명']].values.tolist()
            except:
                if name_wb_p == ' 영양군(청암교)':
                    name_wb_p = '영양군(청암교)'
                elif name_p == ' 영양군(청암교)':
                    name_p = '영양군(청암교)'
                if name_wb_p == '양산천':
                    name_wb_p = '양산시(효충교)'
                elif name_p == '양산천':
                    name_p = '양산시(효충교)'

                try:
                    hyd_df_merged.loc[i,hyd_point_gdf.columns.tolist()[2:]] = np.array(hyd_point_gdf.loc[(hyd_point_gdf.관측소명 == name_p),hyd_point_gdf.columns.tolist()[2:]].values.tolist()[0],dtype=object)
                except:
                    try:
                        hyd_df_merged.loc[i,hyd_point_gdf.columns.tolist()[2:]] = np.array(hyd_point_gdf.loc[(hyd_point_gdf.관측소명 == name_wb_p),hyd_point_gdf.columns.tolist()[2:]].values.tolist()[0],dtype=object)
                        hyd_df_merged.loc[i,['수계명', '관측소명']] = hyd_df_merged.loc[i,['관측소명', '수계명']].values.tolist()
                    except:

                        print(f'지점정보 못찾음: {i}행, 수계명_ {name_wb_p},관측소명_ {name_p}')
                        na_list.append([name_wb_p,name_p])


    hyd_df_merged = gpd.GeoDataFrame(hyd_df_merged, geometry= 'geometry')
    pickle.dump(hyd_df_merged, open(os.path.join(data_path,"merged","merged_1","hyd_gdf.p",), "wb"))
        
