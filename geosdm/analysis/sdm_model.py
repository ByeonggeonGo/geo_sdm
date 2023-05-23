from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import sklearn.metrics
import shap
import os
import pandas as pd
import numpy as np

class SpiciesDistributionModel():
    def __init__(self, data_path: str):
        self._data_path = data_path
        self._col_map = col_map = {
                        '수소이온농도(pH)': 'pH',
                        '전기전도도(EC)': 'electrical_conductivity(EC)',
                        '생물화학적산소요구량(BOD)': 'biological_oxygen_demand(BOD,mg/L)',
                        '용존산소(DO)': 'dissolved_oxygen(DO,mg/L)',
                        '화학적산소요구량(COD)': 'chemical_oxygen_demand(COD,mg/L)',

                        'cate_1': 'Urbanized-dry land cover',
                        'cate_2': 'Agricultural land cover',
                        'cate_3': 'Forest land cover',
                        'cate_4': 'Grassland land cover',
                        'cate_5': 'Wetland (riparian vegetation) land cover',
                        'cate_6': 'Barren land cover',
                        'cate_7': 'Water area land cover',

                        '합계 일조시간(hr)': 'total_sunshine_duration(hr)',
                        '10분 최다 강수량(mm)': '10_min_max_precipitation(mm)',
                        '일강수량(mm)': 'daily_precipitation(mm)',
                        '1시간 최다강수량(mm)': '1hr_max_precipitation(mm)',
                        '평균기온(°C)': 'average_temperature(°C)',
                        '평균 이슬점온도(°C)': 'average_dew_point_temperature(°C)',
                        '평균 지면온도(°C)': 'average_ground_temperature(°C)',
                        '최저기온(°C)': 'minimum_temperature(°C)',
                        '최고기온(°C)': 'maximum_temperature(°C)',
                        '최저 초상온도(°C)': 'minimum_dew_point_temperature(°C)',
                        '최대 순간 풍속(m/s)': 'maximum_instantaneous_wind_speed(m/s)',
                        '평균 풍속(m/s)': 'average_wind_speed(m/s)',
                        '최대 풍속(m/s)': 'maximum_wind_speed(m/s)',
                        '최소 상대습도(%)': 'minimum_relative_humidity(%)',
                        '평균 상대습도(%)': 'average_relative_humidity(%)',
                        '평균 현지기압(hPa)': 'average_local_pressure(hPa)',
                        '평균 해면기압(hPa)': 'average_sea_level_pressure(hPa)',
                        '평균 증기압(hPa)': 'average_vapor_pressure(hPa)',
                        '최저 해면기압(hPa)': 'minimum_sea_level_pressure(hPa)',

                        '평균수심': 'mean_water_depth(m)',
                        '수면폭': 'width_water_surface(m)',
                        '유량': 'discharge(m3/s)',
                        '단면적': 'cross-sectional_area(m2)',
                        '평균유속': 'mean_flow_velocity(m/s)',
                        }
        base_df = pd.read_csv(os.path.join(self._data_path,"merged","merged_result",'name_merged_edit2.csv',), encoding= 'euc-kr')
        base_df = base_df.dropna(subset=['평균유속','용존산소(DO)'])

        inde_val_list = {'조사지점', '일시', '수면폭', '평균수심', '단면적', '유량', '평균유속', '용존산소(DO)', '화학적산소요구량(COD)', '전기전도도(EC)', '수소이온농도(pH)', '생물화학적산소요구량(BOD)', 'cate_1', 'cate_2', 'cate_3', 'cate_4', 'cate_5', 'cate_6', 'cate_7', '최대 순간 풍속(m/s)', '평균 이슬점온도(°C)', '최저기온(°C)', '최소 상대습도(%)', '10분 최다 강수량(mm)', '최저 초상온도(°C)', '합계 일조시간(hr)', '평균 지면온도(°C)', '평균 증기압(hPa)', '평균 풍속(m/s)', '최저 해면기압(hPa)', '1시간 최다강수량(mm)', '최대 풍속(m/s)', '평균 상대습도(%)', '평균 해면기압(hPa)', '평균 현지기압(hPa)', '최고기온(°C)', '일강수량(mm)', '평균기온(°C)'}
        de_val_list = set(base_df.columns.tolist()[1:]) - {'조사지점', '일시', '수면폭', '평균수심', '단면적', '유량', '평균유속', '니켈(Ni)', '투명도_x', '용존산소(DO)', '셀레늄(Se)', '염소이온(Cl-)', '화학적산소요구량(COD)', '전기전도도(EC)', '수심', '수온', '바륨(Ba)', '수소이온농도(pH)', '생물화학적산소요구량(BOD)', '함수율(%)', 'TOC(%)', '최고수심(m)', '표층-측정수심(m)', '저층DO(㎎/L)', 'Cu(㎎/㎏)', '완전연소가능량(%)', 'COD(%)', 'Ni(㎎/㎏)', '입도-실트(%)', 'T-N(㎎/㎏)', 'Li(㎎/㎏)', '표층-수온(℃)', 'Zn(㎎/㎏)', '표층전기전도도(25℃μS/㎝)', 'Cr(㎎/㎏)', '입도-모래(%)', 'Hg(㎎/㎏', 'Pb(㎎/㎏)', '표층pH', '표층-DO(㎎/L)', 'TotalPAHs(㎍/㎏)', '저층수온(℃)', 'T-P(㎎/㎏)', 'As(㎎/㎏)', '저층전기전도도(25℃μS/㎝)', '저층pH', '입도-점토(%)', '투명도_y', 'Cd(㎎/㎏)', '저층-측정수심(m)', 'Al(%)', 'cate_0', 'cate_1', 'cate_2', 'cate_3', 'cate_4', 'cate_5', 'cate_6', 'cate_7', '평균 20cm 지중온도(°C)', '9-9강수(mm)', '1시간 최다일사 시각(hhmi)', '최저기온 시각(hhmi)', '풍정합(100m)', '합계 대형증발량(mm)', '최대 풍속 시각(hhmi)', '1시간 최다 강수량 시각(hhmi)', '평균 전운량(1/10)', '최고 해면기압(hPa)', '최대 순간 풍속(m/s)', '합계 소형증발량(mm)', '평균 이슬점온도(°C)', '강수 계속시간(hr)', '최고기온 시각(hhmi)', '최소 상대습도 시각(hhmi)', '최저기온(°C)', '최저 해면기압 시각(hhmi)', '5.0m 지중온도(°C)', '합계 일사량(MJ/m2)', '최소 상대습도(%)', '최고 해면기압 시각(hhmi)', '10분 최다 강수량(mm)', '평균 중하층운량(1/10)', '가조시간(hr)', '최저 초상온도(°C)', '합계 일조시간(hr)', '평균 지면온도(°C)', '1.5m 지중온도(°C)', '평균 증기압(hPa)', '1시간 최다일사량(MJ/m2)', '평균 풍속(m/s)', '평균 5cm 지중온도(°C)', '최대 순간풍속 시각(hhmi)', '0.5m 지중온도(°C)', '평균 10cm 지중온도(°C)', '최저 해면기압(hPa)', '최대 순간 풍속 풍향(16방위)', '합계 3시간 신적설(cm)', '일 최심신적설 시각(hhmi)', '1.0m 지중온도(°C)', '일 최심신적설(cm)', '일 최심적설 시각(hhmi)', '1시간 최다강수량(mm)', '일 최심적설(cm)', '최대 풍속(m/s)', '평균 상대습도(%)', '최다풍향(16방위)', '최대 풍속 풍향(16방위)', '평균 해면기압(hPa)', '평균 30cm 지중온도(°C)', '10분 최다강수량 시각(hhmi)', '최고기온(°C)', '평균 현지기압(hPa)', '일강수량(mm)', '3.0m 지중온도(°C)', '평균기온(°C)'}

        df_list_by_name = []
        df_list_name = []
        col_list = list(inde_val_list)
        col_list.sort()



        for i in list(de_val_list):
            df_list_by_name.append(base_df.loc[:,[i]+col_list].dropna())
            df_list_name.append(i)

        self._sub_feat_list = list(set(df_list_by_name[0].iloc[:,1:].columns) - {'일시','조사지점'})
        self._df_list_by_name = df_list_by_name
        self._df_list_name = df_list_name

    def model_fitting(self, species_name: str) -> list[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:

        y = self._df_list_by_name[self._df_list_name.index('Rhithrogena na')].iloc[:,0].map(lambda x : 1 if x > 0 else 0).values
        x = self._df_list_by_name[self._df_list_name.index('Rhithrogena na')].loc[:,self._sub_feat_list].values
        x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state=93,)
        model = XGBClassifier(random_state=100)
        model.fit(x,y.ravel())

        shap_values = shap.TreeExplainer(model).shap_values(x)

        return [x, x_test, y, y_test, shap_values, [self._col_map[name] for name in self._sub_feat_list]]

        # shap.summary_plot(shap_values, x, feature_names=[self._col_map[name] for name in self._sub_feat_list],show=False,max_display=10)
    

    
