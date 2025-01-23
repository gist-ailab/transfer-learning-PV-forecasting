# from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, \
#      Dataset_DKASC_AliceSprings, Dataset_DKASC_Yulara, Dataset_GIST, Dataset_German, Dataset_UK, Dataset_OEDI_Georgia, Dataset_OEDI_California, Dataset_Miryang, Dataset_Miryang_MinMax, Dataset_Miryang_Standard, Dataset_SineMax
from data_provider.data_loader import Dataset_DKASC, Dataset_GIST, Dataset_Miryang, Dataset_Germany, Dataset_OEDI_Georgia, Dataset_OEDI_California, Dataset_UK, Dataset_SineMax
from torch.utils.data import DataLoader, ConcatDataset
import torch

data_dict = {
    'DKASC' : Dataset_DKASC,
    'SineMax': Dataset_SineMax,
    'GIST': Dataset_GIST,
    'Miryang': Dataset_Miryang,
    'Germany': Dataset_Germany,
    'OEDI_Georgia': Dataset_OEDI_Georgia,
    'OEDI_California': Dataset_OEDI_California,
    'UK': Dataset_UK,
}

split_configs = {
    'DKASC': {
        'train': [1, 3, 5, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                  33, 34, 35, 36, 37, 39, 41, 42],
        'val': [2, 8, 14, 32, 38],   # 2
        'test': [4, 6, 31, 40, 43]  # 4, 43
    },
    # 사이트로 나누는 loc
    'DKASC_AliceSprings': {
        'train': [1, 4, 7, 9, 10, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 33, 34, 35, 36],
        'val': [2, 5, 8, 11, 14],
        'test': [3, 6, 12, 20, 28, 32]
    },
    'DKASC_Yulara': {
        'train': [1, 4, 7],
        'val': [2, 5],
        'test': [3, 6]
    },
    'GIST': {
        'train': [1, 4, 5, 6, 7, 8, 9, 10, 11, 13],
        'val': [2, 12],
        'test': [3, 14]
    },
    'Germany': {
        'train': [2, 3, 4, 5, 6, 7, 8],
        'val': [9],
        'test': [1]
    },
    'Miryang': {
        'train': [1, 2, 3, 5, 7],
        'val': [6],
        'test': [4]
    },
    #### 날짜로 나누는 loc

    'OEDI_California': { # 2017.12.05  2023.10.31
        'train' : 0.75,
        'val' : 0.1,
        'test' : 0.15
    },
    'OEDI_Georgia' : {  # 2018.03.29  2022.03.10
        'train' : 0.75,
        'val' : 0.1,
        'test' : 0.15
    },
    'UK' : {
        'train' : 0.75,
        'val' : 0.1,
        'test' : 0.15
    },
}


def data_provider(args, flag, distributed=False):
    ## flag : train, val, test
    Data = data_dict[args.data]

    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        # Data = Dataset_DKASC_AliceSprings
    else: # train, val
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        data_type=args.data_type,
        split_configs=split_configs[args.data],
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        timeenc=timeenc,
        freq=freq,
        scaler=args.scaler
        )
    print(flag, data_set.__len__())
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            data_set,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=shuffle_flag
        )
        shuffle_flag = False  # When using a sampler, DataLoader shuffle must be False
    else:
        sampler = None

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True,
        sampler=sampler)
    return data_set, data_loader
