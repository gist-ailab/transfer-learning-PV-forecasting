import pandas as pd
import os
from tqdm import tqdm

# de_data = pd.read_csv('/home/bak/Projects/PatchTST/data/household_data_60min_singleindex(selected_column).csv') # Deutschland: DE
# # print(de_data.columns)
#
# de_data_num = 0
# for datum in de_data.columns:
#     # print(datum)
#     if 'pv' in datum:
#         print(datum)
#         pv_data = de_data[datum].dropna()
#         print(pv_data.shape[0])
#         de_data_num += pv_data.shape[0]
# print(f'de_data_num: {de_data_num} [hrs]')




# DKASC_dir = '/home/bak/Dataset/DKASC_AliceSprings/'
# files_list = os.listdir(DKASC_dir)
#
# DKASC_files = []
# for i in files_list:
#     if i.endswith('.csv'):
#         file_dir = os.path.join(DKASC_dir, i)
#         # print(file_dir)
#         DKASC_files.append(file_dir)
# dkasc_data_num = 0
# for file in DKASC_files:
#     data = pd.read_csv(file)
#     # print(data.columns)
#     print(data.shape[0]//12)
#     num_rows = data.shape[0]//12
#     dkasc_data_num += num_rows
# print(f'DKASC data number: {dkasc_data_num} [hrs]')




# OEDI_dir = '/home/bak/Dataset/OEDI/2105(Maui_Ocean_Center)'
# # OEDI_dir = '/home/bak/Dataset/OEDI/2107(Arbuckle_California)'
# # OEDI_dir = '/home/bak/Dataset/OEDI/9069(Georgia)'
# files_list = os.listdir(OEDI_dir)
#
# OEDI_files = []
# pre_time = int(0)
# for i in files_list:
#     if i.endswith('.csv') and 'inv'  in i:  # change 'inv' to 'electric' for another dataset
#         file_dir = os.path.join(OEDI_dir, i)
#         print(file_dir)
#         OEDI_files.append(file_dir)
# oedi_data_num = 0
# for file in OEDI_files:
#     data = pd.read_csv(file)
#     # print(file)
#     # print(data.columns)
#
#     times = data['measured_on'].tolist()
#     count_hours = 0
#     for idx, val in tqdm(enumerate(times)):
#         cur_time = int(val.split(' ')[1][:2])
#         if cur_time != pre_time:
#             count_hours += 1
#             pre_time = cur_time
#         else:
#             pass
#     print(f'{file} hours: {count_hours} [hrs]')
#
#     oedi_data_num += count_hours
# print(f'OEDI data number: {oedi_data_num} [hrs]')
# '''
# Maui: 284396
# California: 52745
# Georgia: 64110
# '''

UK_data = '/home/bak/Dataset/UK_data/PV Data/PV Data - csv files only/2014-11-28 Cleansed and Processed/EXPORT HourlyData/EXPORT HourlyData - Customer Endpoints.csv'
data = pd.read_csv(UK_data)
UK_data_num = 0

value_to_keep = ['Forest Road', 'Maple Drive East', 'YMCA']

data = data[data['Substation'].isin(value_to_keep)]

num_rows = data.shape[0]
print(f'UK data number: {num_rows} [hrs]')
