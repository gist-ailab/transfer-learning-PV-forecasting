import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.timefeatures import time_features
import warnings
import copy
import pickle
import joblib
import random
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings('ignore')

class Dataset_DKASC(Dataset):
    def __init__(self,
                 root_path, data_path=None,
                 data_type='all', split_configs=None,
                 flag='train', size=None,
                 timeenc=0, freq='h',
                 scaler=True,
                 input_channels=None,
                ):
        """
        이 예시는 installation 단위로 데이터가 나뉘어 있고,
        split_configs로 train, val, test 설정을 받아 데이터를 분할합니다.

        Args:
            root_path (str): 데이터 파일들이 저장된 경로.
            data_path (str, optional): 추가적인 데이터 경로. 단일 PV array 데이터를 가져오는 경우 사용.
            split_configs (dict): train, val, test 설정.
            flag (str): 'train', 'val', 'test' 중 하나.
            size (tuple, optional): seq_len, label_len, pred_len의 길이를 포함.
            timeenc (int): 시간 인코딩 방법.
            freq (str): 시간 데이터 빈도.
            scaler (bool): 스케일링 여부.
        """

        if size is None:
            raise ValueError("size cannot be None. Please specify seq_len, label_len, and pred_len explicitly.")
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val', 'test'], "flag must be 'train', 'val', or 'test'."

        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.timeenc = timeenc
        self.freq = freq
        self.scaler = scaler
        self.split_configs = split_configs

        # input_channels가 None이면 기본값 사용
        if input_channels is None:
            input_channels = ['Global_Horizontal_Radiation', 'Weather_Temperature_Celsius',
                              'Weather_Relative_Humidity', 'Wind_Speed', 'Active_Power']
        self.input_channels = input_channels
        print(f"Input_channels: {self.input_channels}")

        # mapping 파일 로드
        dataset_name = self.__class__.__name__.split('_')[-1]  # 클래스 이름에서 데이터셋 이름 추출
        self.mapping_df = pd.read_csv(f'./data_provider/{dataset_name}_mapping/mapping_{data_type}.csv')
        self.current_dataset = self.mapping_df[self.mapping_df['dataset'] == dataset_name]
        self.current_dataset['index'] = self.current_dataset['mapping_name'].apply(lambda x: int(x.split('_')[0]))  # index 열 추가

        # flag에 따른 installation 리스트 설정 (train, val, test)
        self.inst_list = self.split_configs[flag]

        # 스케일러 저장 경로
        self.scaler_dir = os.path.join(root_path, 'scalers')
        os.makedirs(self.scaler_dir, exist_ok=True)

        # 데이터를 저장할 리스트
        self.data_x_list = []
        self.data_y_list = []
        self.data_stamp_list = []
        self.inst_id_list = []
        self.capacity_info = {}

        # 데이터 준비 및 indices 생성
        self._prepare_data()
        self.indices = self._create_indices()

    def _prepare_data(self):
        for inst_id in self.inst_list:
            # inst_id를 기반으로 파일 이름 가져오기
            file_row = self.current_dataset[self.current_dataset['index'] == inst_id]
            if file_row.empty:
                raise ValueError(f"No matching file found for inst_id {inst_id} in dataset {current_dataset}.")
            
            # 파일명과 capacity 정보 추출
            file_name = file_row['original_name'].values[0]
            try:
                capacity = float(file_name.split('_')[0])
                self.capacity_info[inst_id] = capacity
                self.inst_id_list.append(inst_id)
            except (IndexError, ValueError):
                raise ValueError(f"Invalid capacity format in filename: {file_name}")
            
            # 데이터 로드 및 전처리
            csv_path = os.path.join(self.root_path, file_name)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Data file not found: {csv_path}")
            
            # read dataset
            df_raw = pd.read_csv(csv_path)
            df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], errors='coerce')

            # 필요한 컬럼만 추출
            df_raw = df_raw[['timestamp'] + self.input_channels]

            # 시간 피처 생성
            df_stamp = pd.DataFrame()
            df_stamp['timestamp'] = df_raw['timestamp']
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp['timestamp'].dt.month
                df_stamp['day'] = df_stamp['timestamp'].dt.day
                df_stamp['weekday'] = df_stamp['timestamp'].dt.weekday
                df_stamp['hour'] = df_stamp['timestamp'].dt.hour
                data_stamp = df_stamp[['month', 'day', 'weekday', 'hour']].values
            else:
                data_stamp = time_features(df_stamp['timestamp'], freq=self.freq).transpose(1, 0)

            df_data = df_raw[self.input_channels]
            scaler_path = os.path.join(self.scaler_dir, f"{file_name}_scaler.pkl")

            if self.scaler:
                # 스케일러 fit & transform 로직
                if not os.path.exists(scaler_path):
                    scaler_dict = {}
                    for ch in self.input_channels:
                        scaler = StandardScaler()
                        scaler.fit(df_data[[ch]])
                        scaler_dict[ch] = scaler
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(scaler_dict, f)
                else:
                    with open(scaler_path, 'rb') as f:
                        scaler_dict = pickle.load(f)

                transformed_data = [scaler_dict[ch].transform(df_data[[ch]]) for ch in self.input_channels]
                data = np.hstack(transformed_data)
            else:
                data = df_data.values

            self.data_x_list.append(data)
            self.data_y_list.append(data)
            self.data_stamp_list.append(data_stamp)

    def _create_indices(self):
        indices = []
        for inst_idx, data_x in enumerate(self.data_x_list):
            total_len = len(data_x)
            max_start = total_len - self.seq_len - self.pred_len + 1
            for s in range(max_start):
                indices.append((inst_idx, s))
        return indices

    def __getitem__(self, index):
        inst_idx, s_begin = self.indices[index]
        inst_id = self.inst_id_list[inst_idx]

        data_x = self.data_x_list[inst_idx]
        data_y = self.data_y_list[inst_idx]
        data_stamp = self.data_stamp_list[inst_idx]

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data_x[s_begin:s_end]
        seq_y = data_y[r_begin:r_end]
        seq_x_mark = data_stamp[s_begin:s_end]
        seq_y_mark = data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, inst_id

    def __len__(self):
        return len(self.indices)

    def inverse_transform(self, data, inst_ids):
        """
        스케일링된 데이터를 원래 스케일로 변환
        
        Args:
            inst_ids: 배치 내 각 데이터의 installation ID (배치 크기만큼의 길이)
            data: 변환할 데이터 (batch_size, seq_len, feature_dim)
        Returns:
            inverse_data: 역변환된 데이터 (입력과 같은 shape)
        """
        if not self.scaler:
            return data
            
        data_org = data.copy()
        inverse_data = np.zeros_like(data_org)
        
        # unique한 installation IDs 추출
        unique_inst_ids = np.unique(inst_ids)
        
        # 각 unique installation에 대해
        for inst_id in unique_inst_ids:
            # 현재 installation의 스케일러 로드
            file_row = self.current_dataset[self.current_dataset['index'] == inst_id]
            file_name = file_row['original_name'].values[0]
            scaler_path = os.path.join(self.scaler_dir, f"{file_name}_scaler.pkl")
            
            with open(scaler_path, 'rb') as f:
                scaler_dict = pickle.load(f)
                
            # 현재 installation에 해당하는 데이터의 인덱스 찾기
            inst_mask = (inst_ids == inst_id)
            
            # 해당하는 데이터만 추출하여 역변환
            inst_data = data_org[inst_mask].reshape(-1, 1)
            inverse_inst_data = scaler_dict['Active_Power'].inverse_transform(inst_data)
            
            # 역변환된 데이터를 원래 위치에 복원
            inverse_data[inst_mask] = inverse_inst_data.reshape(data_org[inst_mask].shape)
        
        return inverse_data


########################################################################################

class Dataset_GIST(Dataset_DKASC):
    def __init__(self,
                 root_path, data_path=None,
                 data_type='all', split_configs=None,
                 flag='train', size=None,
                 timeenc=0, freq='h',
                 scaler=True,
                 ):
        super().__init__(root_path, data_path, data_type, split_configs, flag, size, timeenc, freq, scaler)

#######################################################################################

class Dataset_Miryang(Dataset_DKASC):
    def __init__(self,
                 root_path, data_path=None,
                 data_type='all', split_configs=None,
                 flag='train', size=None,
                 timeenc=0, freq='h',
                 scaler=True,
                 ):
        super().__init__(root_path, data_path, data_type, split_configs, flag, size, timeenc, freq, scaler)


#######################################################################################

class Dataset_Germany(Dataset_DKASC):
    def __init__(self,
                 root_path, data_path=None,
                 data_type='all', split_configs=None,
                 flag='train', size=None,
                 timeenc=0, freq='h',
                 scaler=True,
                 ):
        super().__init__(root_path, data_path, data_type, split_configs, flag, size, timeenc, freq, scaler)


#######################################################################################

class Dataset_TimeSplit(Dataset):
    def __init__(self, root_path, data_path=None, data_type='all', split_configs=None,
                 flag='train', size=None, timeenc=0, freq='h', scaler=True, input_channels=None):
        """
        시계열 데이터를 시간 순으로 분할하는 데이터셋 클래스
        Args:
            root_path (str): 데이터 파일들이 저장된 경로
            data_path (str): 데이터 파일명 (단일 파일 사용시)
            split_configs (dict): train, val, test 비율 설정
            flag (str): 'train', 'val', 'test' 중 하나
            size (list): [seq_len, label_len, pred_len]
            timeenc (int): 시간 인코딩 방식
            freq (str): 시계열 데이터 frequency
            scaler (bool): 스케일링 적용 여부
            input_channels (list): 입력 데이터 채널 목록
        """
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.timeenc = timeenc
        self.freq = freq
        self.scaler = scaler
        self.split_configs = split_configs

        if size is None:
            raise ValueError("size cannot be None. Please specify seq_len, label_len, and pred_len explicitly.")
        self.seq_len, self.label_len, self.pred_len = size

        if input_channels is None:
            input_channels = ['Global_Horizontal_Radiation', 'Weather_Temperature_Celsius',
                              'Weather_Relative_Humidity', 'Wind_Speed', 'Active_Power']
        self.input_channels = input_channels

        # 스케일러 저장 경로
        self.scaler_dir = os.path.join(root_path, 'scalers')
        os.makedirs(self.scaler_dir, exist_ok=True)

        # 데이터를 저장할 리스트
        self.data_x_list = []
        self.data_y_list = []
        self.data_stamp_list = []

        self.inst_info = {}  # {file_name: capacity} 형태로 저장
        self.inst_id_list = []
        self.timestamp_inst_ids = []  # 각 시점별 installation ID를 저장할 리스트 추가

        # 데이터 준비
        self._prepare_data()
        self.indices = self._create_indices()

    def _get_split_dates(self, df):
        """시간 분할을 위한 날짜 계산"""
        total_days = (df['timestamp'].max() - df['timestamp'].min()).days
        train_end = int(total_days * self.split_configs['train'])
        val_end = train_end + int(total_days * self.split_configs['val'])
        
        train_date = df['timestamp'].min() + pd.Timedelta(days=train_end)
        val_date = df['timestamp'].min() + pd.Timedelta(days=val_end)
        
        return train_date, val_date

    def _fit_scalers(self, train_data):
        """training data로 scaler를 학습하고 저장"""
        scaler_dict = {}
        for ch in self.input_channels:
            scaler = StandardScaler()
            scaler.fit(train_data[[ch]])
            scaler_dict[ch] = scaler
        
        scaler_path = os.path.join(self.scaler_dir, f"{self.__class__.__name__}_scalers.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler_dict, f)
        return scaler_dict

    def _load_scalers(self):
        """저장된 scaler 로드"""
        scaler_path = os.path.join(self.scaler_dir, f"{self.__class__.__name__}_scalers.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found. Train the model first. Path: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)

    def _prepare_data(self):

        if self.data_path is not None:
            # 단일 파일인 경우
            file_name = os.listdir(self.root_path)[0]
            file_path = os.path.join(self.root_path, file_name)
            df = pd.read_csv(file_path)
            inst_id = self._process_single_file(df, self.data_path)
            self.timestamp_inst_ids = [inst_id] * len(df)
            
        else:
            # 디렉토리의 모든 CSV 파일 처리
            dfs = []
            for file in os.listdir(self.root_path):
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(self.root_path, file))
                    inst_id = self._process_single_file(df, file)
                    self.timestamp_inst_ids.extend([inst_id] * len(df))
                    dfs.append(df)
            df = pd.concat(dfs, ignore_index=True)

        # # CSV 파일 로드
        # if self.data_path is not None:
        #     df = pd.read_csv(os.path.join(self.root_path, self.data_path))
        # else:
        #     # 디렉토리의 모든 CSV 파일 로드
        #     dfs = []
        #     for file in os.listdir(self.root_path):
        #         if file.endswith('.csv'):
        #             df_temp = pd.read_csv(os.path.join(self.root_path, file))
        #             dfs.append(df_temp)
        #     df = pd.concat(dfs, ignore_index=True)

        # timestamp 처리
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # 시간 기반 분할
        train_date, val_date = self._get_split_dates(df)

        # 전체 데이터에서 훈련 데이터 추출
        train_data = df[df['timestamp'] < train_date]
        
        # Scaler 처리
        if self.scaler:
            if self.flag == 'train':
                # 훈련 데이터로 scaler를 학습하고 저장
                scaler_dict = self._fit_scalers(train_data[self.input_channels])
            else:
                # 저장된 scaler 로드
                scaler_dict = self._load_scalers()

        # flag에 따른 데이터 선택
        if self.flag == 'train':
            df_subset = df[df['timestamp'] < train_date]
        elif self.flag == 'val':
            df_subset = df[(df['timestamp'] >= train_date) & 
                        (df['timestamp'] < val_date)]
        else:  # test
            df_subset = df[df['timestamp'] >= val_date]

        # 시간 특성 생성
        if self.timeenc == 0:
            data_stamp = pd.DataFrame({
                'month': df_subset['timestamp'].dt.month,
                'day': df_subset['timestamp'].dt.day,
                'weekday': df_subset['timestamp'].dt.weekday,
                'hour': df_subset['timestamp'].dt.hour,
            }).values
        else:
            data_stamp = time_features(df_subset['timestamp'], freq=self.freq).transpose(1, 0)

        # 데이터 변환
        df_data = df_subset[self.input_channels]
        if self.scaler:
            transformed_data = [scaler_dict[ch].transform(df_data[[ch]]) for ch in self.input_channels]
            data = np.hstack(transformed_data)
        else:
            data = df_data.values

        self.data_x_list.append(data)
        self.data_y_list.append(data)
        self.data_stamp_list.append(data_stamp)

    def _process_single_file(self, df, file_name):
        """개별 파일 처리 및 installation 정보 저장"""
        try:
            # Active_Power 컬럼의 최대값을 capacity로 사용
            capacity = df['Active_Power'].max()
            inst_id = len(self.inst_info)  # 순차적인 ID 부여
            self.inst_info[file_name] = {
                'capacity': capacity,
                'id': inst_id,
                'max_power': capacity  # MAPE 계산에 사용될 실제 최대 발전량
            }
            self.inst_id_list.append(inst_id)
            return inst_id
        except KeyError:
            print(f"Warning: 'Active_Power' column not found in file: {file_name}")
            capacity = 1.0  # 기본값 설정
            inst_id = len(self.inst_info)
            self.inst_info[file_name] = {
                'capacity': capacity,
                'id': inst_id,
                'max_power': capacity
            }
            self.inst_id_list.append(inst_id)
            return inst_id

    def _create_indices(self):
        """시퀀스 인덱스 생성"""
        indices = []
        data_x = self.data_x_list[0]  # 단일 array 사용
        total_len = len(data_x)
        max_start = total_len - self.seq_len - self.pred_len + 1
        for s in range(max_start):
            indices.append(s)
        return indices

    def __getitem__(self, index):
        s_begin = self.indices[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x_list[0][s_begin:s_end]
        seq_y = self.data_y_list[0][r_begin:r_end]
        seq_x_mark = self.data_stamp_list[0][s_begin:s_end]
        seq_y_mark = self.data_stamp_list[0][r_begin:r_end]

        # 해당 시점의 installation ID 반환
        inst_id = self.timestamp_inst_ids[s_begin]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, inst_id

    def __len__(self):
        return len(self.indices)

    def inverse_transform(self, data, inst_ids=None):
        """스케일링된 데이터를 원래 스케일로 변환"""
        if not self.scaler:
            return data
            
        data_org = data.copy()
        scaler_dict = self._load_scalers()
        
        # Active_Power에 대한 역변환만 수행
        inverse_data = scaler_dict['Active_Power'].inverse_transform(
            data_org.reshape(-1, 1)
        ).reshape(data_org.shape)
        
        return inverse_data

class Dataset_OEDI_California(Dataset_TimeSplit):
    def __init__(self, root_path, data_path=None, data_type='all', split_configs=None,
                 flag='train', size=None, timeenc=0, freq='h', scaler=True):
        input_channels = ['Global_Horizontal_Radiation', 'Weather_Temperature_Celsius',
                          'Wind_Speed', 'Active_Power']
        super().__init__(root_path, data_path, data_type, split_configs, flag, size,
                         timeenc, freq, scaler, input_channels=input_channels)

class Dataset_OEDI_Georgia(Dataset_TimeSplit):
    def __init__(self, root_path, data_path=None, data_type='all', split_configs=None,
                 flag='train', size=None, timeenc=0, freq='h', scaler=True):
        input_channels = ['Global_Horizontal_Radiation', 'Weather_Temperature_Celsius',
                          'Wind_Speed', 'Active_Power']
        super().__init__(root_path, data_path, data_type, split_configs, flag, size,
                         timeenc, freq, scaler, input_channels=input_channels)

class Dataset_UK(Dataset_TimeSplit):
    def __init__(self, root_path, data_path=None, data_type='all', split_configs=None,
                 flag='train', size=None, timeenc=0, freq='h', scaler=True):
        super().__init__(root_path, data_path, data_type, split_configs, flag, size,
                         timeenc, freq, scaler)



####################################################

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                features='MS', data_path='', target='Active_Power',
                scale=True, timeenc=0, freq='h', scaler='MinMaxScaler'):
    # size [seq_len, label_len, pred_len] 
    # info
        print("start")
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag


        self.scaler = scaler
        

        self.LOCATIONS = [
            file_name for file_name in os.listdir(self.root_path) if file_name.endswith('.csv')
        ]
        
        self.scalers = {}
        self.x_list = []
        self.y_list = []
        self.ds_list = []

        
        random.seed(42)
        if self.data_path['type'] == 'all':
            print("all")
            random.shuffle(self.LOCATIONS)  # 데이터 섞기
            total_size = len(self.LOCATIONS)
            train_size = int(0.6 * total_size)
            val_size = int(0.3 * total_size)
                    
            if self.flag == 'train':
                self.train = self.LOCATIONS[:train_size]
                print(f'[INFO] Train Locations: total {len(self.train)}')
                print(f'[INFO] Location List: {self.train}')
                
            elif self.flag == 'val':
                self.val = self.LOCATIONS[train_size:train_size + val_size]
                print(f'[INFO] Valid Locations: total {len(self.val)}')
                print(f'[INFO] Location List: {self.val}')

            elif self.flag == 'test':
                self.test = self.LOCATIONS[train_size + val_size:]
                print(f'[INFO] Test Locations: total {len(self.test)}')
                print(f'[INFO] Location List: {self.test}')



        elif self.data_path['type'] == 'debug':
            
            if self.flag == 'train':
                self.train = self.data_path['train']
            
            elif self.flag == 'val':
                self.val = self.data_path['val']
            
            elif self.flag == 'test':
                self.test = self.data_path['test']
            

        

        self.load_preprocessed_data() 
        


    def load_preprocessed_data(self):

        
        # 1. Flag에 맞게 데이터 불러오기
        if self.flag == 'train':
            df_raw = self.load_and_concat_data(self.train)
        
        elif self.flag == 'val':
            df_raw = self.load_and_concat_data(self.val)

        elif self.flag == 'test':
            df_raw = self.load_and_concat_data(self.test)

        # 칼럼 순서와 이름 확인
        print(f"[INFO] Loaded columns: {df_raw.columns}")

        # 2. Time encoding (년, 월, 일, 요일, 시간)
        if self.timeenc == 0:
            data_stamp = pd.DataFrame()
            data_stamp['date'] = pd.to_datetime(df_raw.timestamp)
            data_stamp['year'] = df_raw.date.apply(lambda row: row.year, 1)
            data_stamp['month'] = df_raw.timestamp.apply(lambda row: row.month, 1)
            data_stamp['day'] = df_raw.date.apply(lambda row: row.day, 1)
            data_stamp['weekday'] = df_raw.date.apply(lambda row: row.weekday(), 1)
            data_stamp['hour'] = df_raw.date.apply(lambda row: row.hour, 1)
            
            data_stamp = data_stamp.drop(['date'], axis=1).values()

        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_raw['timestamp'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        # 3. Encoding 후, date columns 생성
        # df_raw['date'] = data_stamp

      
        # 4. Columns 순서 정렬 (timestamp, date, ..... , target)
        cols = df_raw.columns.tolist()
        # cols.remove('date')
        cols.remove('timestamp')
        cols.remove('Active_Power')
        df_raw = df_raw[cols + [self.target]]

        if self.scale: 
            # Train일 때는, Scaler Fit 후에, 저장
            if self.flag == 'train' or self.flag == 'val' or self.flag == 'test':
                for col in df_raw.columns:
                    scaler = StandardScaler()
                    df_raw[col] = scaler.fit_transform(df_raw[[col]])
                    self.scalers[col] = scaler 
                    # Scaler를 pickle 파일로 저장
                    path = os.path.join(self.root_path, f'{col}_scaler.pkl')
                    with open(path, 'wb') as f:
                        pickle.dump(scaler, f)
        
            else:
                # Val, Test일 때는, 저장된 Scaler 불러와서 적용
                self.scalers = {}
                transformed_df = df_raw.copy()  

                for col in df_raw.columns:
                    path = os.path.join(self.root_path, f'{col}_scaler.pkl')
            
                    # Scaler가 존재하는지 확인
                    if os.path.exists(path):
                        with open(path, 'rb') as f:
                            scaler = pickle.load(f) 
                
                        # 해당 칼럼에 스케일러 적용 (transform)
                        df_raw[col] = scaler.transform(transformed_df[[col]]) 
                        self.scalers[col] = scaler
                
                    else:
                        print(f"Scaler for column {col} not found.")


        # 6. 입력할 칼럼들 지정하여 리스트 생성
        if self.features == 'M' or self.features == 'MS':
            # date 열을 제외
            cols_data = df_raw.columns[1:]
            df_x = df_raw[cols_data]
        elif self.features == 'S':
            # Active Power만 추출
            df_x = df_raw[[self.target]]
        

        self.x_list = df_x.values
        # 타겟은 마지막 열인 Active_Power
        self.y_list = df_raw[[self.target]].values
        # date columns만 선택
        self.ds_list = data_stamp      
        
    # 파일 경로를 가져와서 DataFrame으로 합치는 함수
    def load_and_concat_data(self, file_list):
        df_list = []
        for file in file_list:
            file_path = f"{self.root_path}/{file}"  
            df = pd.read_csv(file_path)  
            assert (df.isnull().sum()).sum() == 0, "허용되지 않은 열에 결측치가 존재합니다."            
            df_list.append(df) 
        return pd.concat(df_list, ignore_index=True) 

    def __getitem__(self, index):
        s_begin = index * (self.seq_len + self.label_len + self.pred_len)
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.x_list[s_begin:s_end]
        seq_y = self.y_list[r_begin:r_end]
        seq_x_mark = self.ds_list[s_begin:s_end]
        seq_y_mark = self.ds_list[r_begin:r_end]

        return seq_x, seq_y.reshape(-1, 1), seq_x_mark, seq_y_mark


    def __len__(self):
        return (len(self.x_list) - self.seq_len - self.label_len - self.pred_len + 1) // (self.seq_len + self.label_len + self.pred_len)



    # 평가 시 필요함
    def inverse_transform(self, data):
        data_org = data.copy()
        
        data = self.scalers['Active_Power'].inverse_transform(data.reshape(-1, 1))
        data = data.reshape(data_org.shape[0], data_org.shape[1], -1)
        return data

#############################################################################################3

class Dataset_SineMax(Dataset):
    def __init__(self, root_path=None, flag='train', size=None,
                 features='S', data_path='', scaler=None,
                 target='Sine', scale=False, timeenc=0, freq='h'):
        # 기본적인 정보 초기화
        if size is None:
            self.seq_len = 24
            self.label_len = 12
            self.pred_len = 12
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag
        self.scaler = scaler

        # sine 파형 생성
        total_len = 10000  # 충분히 긴 데이터 생성
        time_steps = np.linspace(0, 2 * np.pi * (total_len / 24), total_len)  # 24시간 주기
        self.y_data = np.maximum(0, np.sin(time_steps)).reshape(-1, 1)

        # 정규화 (필요한 경우)
        if self.scale and self.scaler is not None:
            self.scaler.fit(self.y_data)  # y_data의 스케일 조정
            self.y_data = self.scaler.transform(self.y_data)

        # 더미 데이터 생성
        self.data_stamp = np.tile(np.arange(total_len).reshape(-1, 1), (1, 4))  # time encoding
        self.site = np.zeros((total_len, 1))  # 단일 사이트

        # 시퀀스 인덱스 생성
        self.indices = self.create_sequences_indices()

    def create_sequences_indices(self):
        max_start = len(self.y_data) - self.seq_len - self.pred_len + 1
        return list(range(max_start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        start_idx = self.indices[index]
        s_begin = start_idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.y_data[s_begin:s_end]
        seq_y = self.y_data[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        site = self.site[s_begin:s_end]
        seq_x_ds = self.data_stamp[s_begin:s_end]
        seq_y_ds = self.data_stamp[r_begin:r_end]

        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
            torch.tensor(seq_x_mark, dtype=torch.float32),
            torch.tensor(seq_y_mark, dtype=torch.float32),
            torch.tensor(site, dtype=torch.float32),
            torch.tensor(seq_x_ds, dtype=torch.float32),
            torch.tensor(seq_y_ds, dtype=torch.float32),
        )

    def inverse_transform(self, data):
        if self.scaler is not None and self.scale:
            return self.scaler.inverse_transform(data)
        return data  # 스케일링이 적용되지 않았다면 그대로 반환
    


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)