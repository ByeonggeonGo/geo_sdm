from geosdm.analysis import sdm_model
import os

data_path_list = os.getcwd().split(os.path.sep) + ['data']
data_path = os.path.sep.join(data_path_list)
sdm_controller = sdm_model.SpiciesDistributionModel(data_path)
sdm_controller.get_species_name()
