from geosdm.dataprocessing import datacontroller
import os
from glob import glob

data_path_list = os.getcwd().split(os.path.sep) + ['data']
data_path = os.path.sep.join(data_path_list)
datacontroller = datacontroller.DataContorller(data_path, use_cleaned_dataset = True)
datacontroller.get_merged_set('Name')
datacontroller.get_merged_set('Order')
datacontroller.get_merged_set('Family')

