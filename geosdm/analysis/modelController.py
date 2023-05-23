from .sdm_model import SpiciesDistributionModel
import pandas as pd
import os
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class ModelController():
    def __init__(self, data_path: str, result_path: str):
        self._data_path = data_path
        self._result_path = result_path
        self._species_names = []

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

        

        
#  inputs, feature_names: list
    def get_all_shap_summary_plot_n_values(self, species_names: list, is_save: bool):

        for species_name in species_names:
            

            shap_values_train = self._models[species_name]['model'].get_shap_information(self.x_train)

            

            self._models[species_name]['shap_values_train'] = shap_values_train
            # plt.figure(figsize=[3,3])
            # plt.title(species_name + '_train')
            shap.summary_plot(shap_values_train, self.x_train, feature_names=self._models[species_name]['model']._feature_name_kor,show=False,max_display=10)

            if is_save:
                plt.savefig(os.path.join(self._result_path,'shap_summary_plots',f'{species_name}_shap_summary_plot_trainset.png'))
                plt.close()
            else:
                pass

            shap_values_test = self._models[species_name]['model'].get_shap_information(self.x_test)
            self._models[species_name]['shap_values_test'] = shap_values_test
            # plt.figure(figsize=[3,3])
            # plt.title(species_name + '_test')
            shap.summary_plot(shap_values_test, self.x_test, feature_names=self._models[species_name]['model']._feature_name_kor,show=False,max_display=10)

            if is_save:
                plt.savefig(os.path.join(self._result_path,'shap_summary_plots',f'{species_name}_shap_summary_plot_testset.png'))
                plt.close()
            else:
                pass

                
    def fit_sdms(self, species_names: list):
        sdm_dict = {}

        for species_name in species_names:
            sdm = SpiciesDistributionModel(self._data_path, species_name)

            y = self._df_list_by_name[self._df_list_name.index(species_name)].iloc[:,0].map(lambda x : 1 if x > 0 else 0).values
            x = self._df_list_by_name[self._df_list_name.index(species_name)].loc[:,self._sub_feat_list].values

            sdm.model_fitting(x,y)
            sdm_dict[species_name] = {}
            sdm_dict[species_name]['model'] = sdm
        
        # 모든 종 x는 같으므로 컨트롤러에 x는 베이스로 저장
        x_train, x_test = train_test_split(x, test_size=0.2, random_state=93,)
        self.x_train = x_train
        self.x_test = x_test

        self._models = sdm_dict
        self._species_names = species_names if len(self._species_names) == 0 else list(set(self._species_names + species_names))


    def get_model_performances_all(self):
        col_names = ['train_accuracy', 'train_recall', 'train_auc', 'test_accuracy', 'test_recall', 'test_auc', 'n_sample_train', 'n_sample_test', 'n_presence_train', 'n_absence_train', 'n_presence_test', 'n_absence_test','presence-n/all_n_ratio_train', 'presence-n/all_n_ratio_test', ]
        results = pd.DataFrame(columns=col_names)

        for i,species_name in enumerate(self._species_names):

            results.loc[species_name,col_names] = [
                self._models[species_name]['model'].train_accuracy,
                self._models[species_name]['model'].train_recall,
                self._models[species_name]['model'].train_auc,

                self._models[species_name]['model'].test_accuracy,
                self._models[species_name]['model'].test_recall,
                self._models[species_name]['model'].test_auc,

                self._models[species_name]['model'].train_len,
                self._models[species_name]['model'].test_len,

                self._models[species_name]['model'].train_presence_len,
                self._models[species_name]['model'].train_len - self._models[species_name]['model'].train_presence_len ,

                self._models[species_name]['model'].test_presence_len,
                self._models[species_name]['model'].test_len - self._models[species_name]['model'].test_presence_len ,

                self._models[species_name]['model'].train_presence_len/self._models[species_name]['model'].train_len,
                self._models[species_name]['model'].test_presence_len/self._models[species_name]['model'].test_len,
            ]
        
        results.to_csv(os.path.join(self._result_path,'model_performance','model_results.csv'))

            
            

           

