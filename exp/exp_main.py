from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, LSTM
from models.Stat_models import Naive_repeat, Arima
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, visual_out, visual_original
from utils.metrics import MetricEvaluator

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import os

import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator

import os
import time
import datetime

import warnings
import matplotlib.pyplot as plt
import numpy as np
import wandb
from utils.wandb_uploader import upload_files_to_wandb
from tqdm import tqdm

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.project_name = "pv-forecasting-freeze-test"
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{self.args.model}_run_{current_time}"

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'Naive_repeat': Naive_repeat,
            'Arima': Arima,
            'LSTM': LSTM
        }
        
        model = model_dict[self.args.model].Model(self.args).float()
        
        # 먼저 모델을 GPU로 이동
        model = model.to(self.device)

        # 모델 파라미터 수 계산
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

        if self.args.resume:
            model = self.load_model(model, self.args.source_model_dir)

        # 2. 사전 학습된 모델 로드 및 레이어 프리징
        if (self.args.num_freeze_layers > 0) or (self.args.linear_probe):
            model = self._freeze_layers(model)
        
        # 3. 분산 학습 설정
        if self.args.distributed:
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank
            )
        
        return model
    
    def _freeze_layers(self, model):
        """Helper function to handle layer freezing"""
        model = self.load_model(model, self.args.source_model_dir)

        # Linear probing: Freeze all except head
        if self.args.linear_probe:
            for name, param in model.named_parameters():
                # Exclude positional and input embedding from freezing
                if 'W_pos' in name or 'W_P' in name:
                    continue
                if 'head' not in name:
                    param.requires_grad = False
                    print(f"Layer {name} is frozen for linear probing")
            return model
        
        # Freezing layers dynamically (start from beginning)
        if self.args.num_freeze_layers > 0:
            # head에 가까운 레이어부터 프리징
            layers_to_freeze = list(range(0, self.args.num_freeze_layers))
            
            # Build a list of layers to freeze (excluding head)
            freeze_layers = [f'backbone.encoder.layers.{i}' for i in layers_to_freeze]
            
            # Freeze specified layers
            for name, param in model.named_parameters():
                if any(layer in name for layer in freeze_layers):
                    param.requires_grad = False
                    print(f"Layer {name} is frozen")

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, self.args.distributed)
        return data_set, data_loader

    def _select_optimizer(self, part=None):
        if part is None:
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        else:
            model_optim = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        return nn.MSELoss()

    def masked_loss(self, predictions, targets, mask_value=-9999, loss_fn=torch.nn.MSELoss()):
        """
        Custom loss function to ignore specific mask_value during loss calculation.
        :param predictions: Model predictions [batch_size, seq_len, num_features]
        :param targets: Ground truth values [batch_size, seq_len, num_features]
        :param mask_value: Value to ignore in loss calculation
        :param loss_fn: Base loss function (e.g., MSELoss, MAELoss)
        """
        mask = (targets != mask_value)  # True for valid data
        valid_predictions = predictions[mask]
        valid_targets = targets[mask]
        return loss_fn(valid_predictions, valid_targets)
    
    def load_model(self, model, source_model_dir):
        source_model_dir = os.path.join('./checkpoints', source_model_dir)
        if self.args.resume:
            model_path = os.path.join(source_model_dir, 'model_latest.pth')
        elif (self.args.num_freeze_layers > 0) or (self.args.linear_probe) or self.args.is_fully_finetune:
            model_path = os.path.join(source_model_dir, 'checkpoint.pth')

        model.load_state_dict(torch.load(model_path))
        print(f'Model loaded from {model_path}')
        return model 

    def train(self, output_dir):
        self.args.output_dir = os.path.join('checkpoints', output_dir)
        # wandb 관련 작업은 rank 0에서만 실행
        if self.args.wandb and (not self.args.distributed or self.args.rank == 0):
            self._set_wandb(output_dir)
            config = {
                "model": self.args.model,
                "num_parameters": sum(p.numel() for p in self.model.parameters()),
                "batch_size": self.args.batch_size,
                "num_workers": self.args.num_workers,
                "learning_rate": self.args.learning_rate,
                "loss_function": self.args.loss,
                "dataset": self.args.data,
                "epochs": self.args.train_epochs,
                "input_seqeunce_length": self.args.seq_len,
                "prediction_sequence_length": self.args.pred_len,
                "patch_length": self.args.patch_len,
                "stride": self.args.stride,
                "num_freeze_layers": self.args.num_freeze_layers,
            }
            upload_files_to_wandb(
                project_name=self.project_name,
                run_name=self.run_name,
                config=config
            )        
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        vali_data, vali_loader = self._get_data(flag='test')

        save_path = os.path.join(self.args.output_dir) if 'checkpoint.pth' not in self.args.output_dir else self.args.output_dir
        os.makedirs(save_path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate
        )
        transfer_flag = True if (self.args.num_freeze_layers > 0) or self.args.linear_probe or self.args.is_fully_finetune else False
        print(f'Transfer learning flag: {transfer_flag}')

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_losses = []
            epoch_time = time.time()
            
            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, _) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or self.args.model == 'LSTM':
                            outputs = self.model(batch_x, transfer_flag)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or self.args.model == 'LSTM':
                        outputs = self.model(batch_x, transfer_flag)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    outputs = outputs[:, -self.args.pred_len:, -1:]
                    batch_y = batch_y[:, -self.args.pred_len:, -1:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    # loss = self.masked_loss(outputs, batch_y, mask_value=-9999, loss_fn=criterion)  ### BSH
                    
                    loss.backward()
                    model_optim.step()
                
                train_losses.append(loss.item())

                if (i + 1) % 100 == 0:
                    if self.args.wandb and (not self.args.distributed or self.args.rank == 0):
                        wandb.log({
                            "iteration": (epoch * len(train_loader)) + i + 1,
                            "train_loss_iteration": loss.item()
                        })
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                    speed = (time.time() - epoch_time) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    epoch_time = time.time()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
            
            train_loss = np.average(train_losses)
            vali_loss = self.vali(vali_loader, criterion)
            
            print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f}, Vali Loss: {vali_loss:.7f}")
            print(f"└ cost time: {time.time() - epoch_time}")
            if self.args.wandb and (not self.args.distributed or self.args.rank == 0):
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "validation_loss": vali_loss,
                })
            
            early_stopping(vali_loss, self.model, save_path)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print(f'Learning rate updated to {scheduler.get_last_lr()[0]}')
        
        best_model_path = os.path.join(save_path, 'checkpoint.pth')
        if self.args.wandb:
            upload_files_to_wandb(
                project_name=self.project_name,
                run_name=self.run_name,
                model_weights_path=best_model_path
            )

        final_model_artifact = wandb.Artifact('final_model_weights', type='model')
        final_model_artifact.add_file(best_model_path)
        if self.args.wandb:
            wandb.log_artifact(final_model_artifact)

        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_loader, criterion):
        total_loss = []
        transfer_flag = True if (self.args.num_freeze_layers > 0) or self.args.linear_probe or self.args.is_fully_finetune else False
        print(f'Transfer learning flag: {transfer_flag}')
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or self.args.model == 'LSTM':
                            outputs = self.model(batch_x, transfer_flag)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or self.args.model == 'LSTM':
                        outputs = self.model(batch_x, transfer_flag)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                outputs = outputs[:, -self.args.pred_len:, -1:]
                batch_y = batch_y[:, -self.args.pred_len:, -1:].to(self.device)
                
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
        
        self.model.train()
        return np.average(total_loss)

    def test(self, source_model_dir=None):
        test_data, test_loader = self._get_data(flag='test')
        dir_name = source_model_dir.split('/')[-1]
        result_path = os.path.join('./test_results/', dir_name)
        # if 'checkpoint.pth' in model_path:
        #     folder_path = os.path.join('./test_results/', model_path.split('/')[-1:])
        os.makedirs(result_path, exist_ok=True)

        if source_model_dir[0] == '.':
            source_model_dir = source_model_dir[2:]

        if source_model_dir.split('/')[0] != 'checkpoints':
            model_path = os.path.join('./checkpoints', source_model_dir)
        else:
            model_path = source_model_dir
            
        if 'checkpoint.pth' not in source_model_dir:
            print(f"Model path: {model_path}")
            model_path = os.path.join(model_path, 'checkpoint.pth')
        print(f"Load model from '{model_path}'")
        self.model.load_state_dict(torch.load(model_path))
        
        # MetricEvaluator 초기화
        # evaluator = MetricEvaluator(file_path=os.path.join(folder_path, "site_metrics.txt"))
        evaluator = MetricEvaluator(
            save_path=os.path.join(result_path, "site_metrics.txt"),
            dataset_name=self.args.data,
            data_type=self.args.data_type,
            ref_mse_path=self.args.ref_mse_path,
            )

        pred_list = []
        true_list = []
        input_list = []
        transfer_flag = True if (self.args.num_freeze_layers > 0) or self.args.linear_probe or self.args.is_fully_finetune else False
        print(f'Transfer learning flag: {transfer_flag}')
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, inst_id) in tqdm(enumerate(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
              
                if 'Linear' in self.args.model or 'TST' in self.args.model or self.args.model == 'LSTM':
                    outputs = self.model(batch_x, transfer_flag)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                outputs = outputs[:, -self.args.pred_len:, -1:]
                batch_y = batch_y[:, -self.args.pred_len:, -1:].to(self.device)
                
                # numpy 변환 및 inverse transform
                outputs_np = outputs.detach().cpu().numpy()
                batch_y_np = batch_y.detach().cpu().numpy()
                batch_x_np = batch_x.detach().cpu().numpy()
                inst_id_np = inst_id.cpu().numpy()
                
                # inverse transform 적용
                input_seq = test_data.inverse_transform(batch_x_np, inst_id_np)
                pred = test_data.inverse_transform(outputs_np, inst_id_np)
                true = test_data.inverse_transform(batch_y_np, inst_id_np)

                # input_seq = batch_x_np
                # pred = outputs_np
                # true = batch_y_np

                # denormalized 데이터로 평가 수행
                evaluator.update(inst_id=inst_id_np, preds=pred, targets=true)
                
                pred_list.append(pred)
                true_list.append(true)
                input_list.append(input_seq)

                # if i % 10 == 0:
                #     # self.plot_predictions(i, batch_x_np[0, -5:, -1], batch_y_np[0], outputs_np[0], folder_path)
                #     self.plot_predictions(i,
                #                           input_seq[0, -5:, -1],    # 마지막 5개 입력값
                #                           true[0],                  # 실제값
                #                           pred[0],                  # 예측값
                #                           result_path)
        # print(f"Plotting complete. Results saved in {folder_path}")

        # TODO: metric 계산하는거 개선해야 함.
        # metric 계산 및 결과 출력
        results, overall_mape = evaluator.evaluate_scale_metrics()

        for scale_name, (mae, rmse, mbe, r2, mse, skill_score), site_ids in results:
            print(f'\nScale: {scale_name}')
            print(f"Sites: {site_ids}\n")
            print(f"Number of sites: {len(site_ids)}\n")
            print(f'MAE: {mae:.4f} kW')
            print(f'RMSE: {rmse:.4f} kW')
            print(f'MBE: {mbe:.4f} kW')
            print(f'R2 Score: {r2:.4f}')
            print(f'MSE: {mse:.4f}')
            if skill_score is not None:
                print(f"Skill Score: {skill_score:.4f}\n")
        print(f'\nOverall MAPE: {overall_mape:.4f}%')

        # wandb logging (설정된 경우)
        if self.args.wandb and (not self.args.distributed or self.args.rank == 0):
            self._set_wandb(result_path)
            config = {
                "model": self.args.model,
                "num_parameters": sum(p.numel() for p in self.model.parameters()),
                "batch_size": self.args.batch_size,
                "num_workers": self.args.num_workers,
                "learning_rate": self.args.learning_rate,
                "loss_function": self.args.loss,
                "dataset": self.args.data,
                "epochs": self.args.train_epochs,
                "input_seqeunce_length": self.args.seq_len,
                "prediction_sequence_length": self.args.pred_len,
                "patch_length": self.args.patch_len,
                "stride": self.args.stride,
                "num_freeze_layers": self.args.num_freeze_layers,
            }
            upload_files_to_wandb(
                project_name=self.project_name,
                run_name=self.run_name,
                config=config
            )        

            for scale_name, (mae, rmse, mbe, r2, mse, skill_score), site_ids in results:
                wandb.log({
                    f"test/{scale_name}/Sites": site_ids,
                    f"test/{scale_name}/MAE": mae,
                    f"test/{scale_name}/RMSE": rmse,
                    f"test/{scale_name}/MBE": mbe,
                    f"test/{scale_name}/R2_Score": r2,
                    f"test/{scale_name}/MSE": mse,
                    f"test/{scale_name}/Skill_Score": skill_score if skill_score is not None else 'N/A',
                    f"test/{scale_name}/MAPE": overall_mape
                })
            wandb.log({"test/MAPE": overall_mape})


    def plot_predictions(self, i, input_sequence, ground_truth, predictions, save_path):
        """
        예측 시각화 함수 (인덱스 기반, 시각적 개선)
        Args:
            input_sequence (numpy array): 입력 시퀀스 데이터
            ground_truth (numpy array): 실제값
            predictions (numpy array): 예측값
            save_path (str): 플롯을 저장할 경로
        """
        # 인덱스 기반으로 x축을 설정
        input_index = np.arange(len(input_sequence))
        start_idx = len(input_sequence)
        ground_truth_index = np.arange(start_idx, start_idx + len(ground_truth))
        predictions_index = np.arange(start_idx, start_idx + len(predictions))

        plt.figure(figsize=(14, 8))  # 더 큰 크기로 설정하여 가독성 향상

        # 입력 시퀀스의 마지막 5개 데이터만 플롯 (점선과 작은 점 추���, 투명도 적용)
        plt.plot(input_index, input_sequence.squeeze(),
                 label='Input Sequence', color='royalblue',
                 linestyle='--', alpha=0.7)
        plt.scatter(input_index, input_sequence.squeeze(),
                    color='royalblue', s=10, alpha=0.6)

        # 수정된 ground_truth 사용하여 실제값 플롯 (굵기와 투명도 적용)
        plt.plot(ground_truth_index, ground_truth.squeeze(),
                 label='Ground Truth', color='green',
                 linewidth=2, alpha=0.8)
        
        # 예측값 플롯 (굵기와 투명도 적용)
        plt.plot(predictions_index, predictions.squeeze(),
                 label='Predictions', color='red',
                 linewidth=2, alpha=0.8)

        # 레이블, 제목 설정
        plt.xlabel('Index', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title('Prediction vs Ground Truth', fontsize=14)
        
        # 레전드를 오른쪽 상단에 고정
        plt.legend(loc='upper right', fontsize=10)
        
        # Grid 추가
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        # 플롯 저장
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'pred_{i}.png'))
        plt.close()
    
    def plot_predictions_2(self, i, input_data, actual_data, predicted_data, save_path, target_dataset=None, run_name=None):
        """
        예측 시각화 함수 (인덱스 기반, 시각적 개선)
        Args:
            input_sequence (numpy array): 입력 시퀀스 데이터
            ground_truth (numpy array): 실제값
            predictions (numpy array): 예측값
            save_path (str): 플롯을 저장할 경로
    """
        def thousands_formatter(x, pos):
            if x >= 1000:
                return f'{x/1000:.0f}k'  # 1000 이상이면 k로 변환
            return f'{x:.0f}'            # 1000 미만이면 그대로 표시

        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(25, 15))
        plt.rcParams['font.size'] = 80
        plt.rcParams['font.family'] = 'Liberation Serif'

        for spine in ax.spines.values():
                spine.set_linewidth(5)  # 원하는 굵기로 설정 (예: 2)

        dash, = ax.plot(input_data, '--', color='#4B6D41', label='Input', linewidth=8)
        dash_pattern = [5, 2]  # 선 길이 10, 빈 간격 5
        dash.set_dashes(dash_pattern)

        
        # # 입력 데이터 전체를 먼저 그립니다
        # ax.plot(input_data, '--',
        #         color= '#5E6064',#'#BEBDBD', #'#E2BEA2', #'#755139', #'#3191C7', 
        #         linewidth=3, 
        #         label='Input Data',
        #         zorder=1)
        # # forecast_length = 24

        # 0 - 9 (input) 9 - 11 (pred) 
        forecast_start_idx = len(input_data) - 1
        # 예측 시작 지점을 포함한 예측 구간의 x 좌표를 생성합니다
        forecast_x = np.arange(forecast_start_idx, forecast_start_idx + len(actual_data) + 1)
        actual_data = np.concatenate([input_data[-1].reshape(-1, 1), actual_data])
        predicted_data = np.concatenate([input_data[-1].reshape(-1, 1), predicted_data])
        # 실제 데이터를 그립니다
        # 시작점에서의 연속성을 위해 input_data의 마지막 값을 사용합니다
        # ax.plot(forecast_x, actual_data,
        #         color= '#5E6064', #'#E2BEA2', #'#755139', #'#3191C7', 
        #         linewidth=3, 
        #         label='Actual Data',
        #         zorder=2)
        
        # # 예측 데이터를 점선으로 그립니다
        # ax.plot(forecast_x, predicted_data,
        #         color= '#B31A23', #'#8E44AD', 
        #         linewidth=2, 
        #        label='Predicted Data',
        #         zorder=2)
        plt.plot(forecast_x, actual_data, color='#4B6D41', label='Ground Truth', linewidth=8)
        plt.plot(forecast_x, predicted_data, color='#77202E', label='Prediction', linewidth=8)
        
        # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        # plt.ticklabel_format(style='plain', axis='y', scilimits=(0,4))
        # 그래프 스타일링
        ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
        # 2. MaxNLocator 사용 (개수 지정)
        ax.xaxis.set_major_locator(MaxNLocator(3))  # 최대 5개의 눈금 표시
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        plt.tick_params(axis='y', direction='in', length=20, width=5, pad=12, left=True, labelleft=True, labelsize=100, labelfontfamily='Liberation Serif')

        if 'Miryang' in save_path or 'UK' in save_path:
            plt.tick_params(axis='x', length=20, width=5, pad=12, labelsize=100, labelfontfamily='Liberation Serif')
        else:
            plt.tick_params(axis='x', bottom=False, top=False, labelbottom=False, labelsize=100)
        
        # 레이블 설정
        # ax.set_xlabel('Time Steps', labelpad=10, fontdict={'fontsize': 38, 'fontfamily': 'Liberation Serif'})
        # ax.set_ylabel('Value', labelpad=10, fontdict={'fontsize': 38, 'fontfamily': 'Liberation Serif'})
        # ax.set_title('Time Series Forecasting Prediction VS GroundTruth', pad=15, fontdict={'fontfamily': 'Liberation Serif'})
        
        # # 범례 설정
        # ax.legend(loc='upper left', frameon=True, handlelength=2, edgecolor='black')
        run_name = run_name.replace('_', ' ')
        # ax.set_title(run_name, pad=30, fontdict={'fontsize': 90, 'fontfamily': 'Liberation Serif'})#, font_weight='bold')
        # 여백 조정
        plt.tight_layout()
    
        # 플롯 저장
        os.makedirs(save_path, exist_ok=True)
        plt.yticks(fontfamily='Liberation Serif')
        plt.savefig(os.path.join(save_path, f'pred_{i}.png'))
        plt.close()
        


    def predict(self, source_model_dir=None, target_dataset=None, run_name=None):
        test_data, test_loader = self._get_data(flag='test')
        dir_name = source_model_dir.split('/')[-1]
        result_path = os.path.join('./plot/', dir_name)
        # if 'checkpoint.pth' in model_path:
        #     folder_path = os.path.join('./test_results/', model_path.split('/')[-1:])
        os.makedirs(result_path, exist_ok=True)

        if source_model_dir[0] == '.':
            source_model_dir = source_model_dir[2:]

        if source_model_dir.split('/')[0] != 'checkpoints':
            model_path = os.path.join('./checkpoints', source_model_dir)
        else:
            model_path = source_model_dir
            
        if 'checkpoint.pth' not in source_model_dir:
            print(f"Model path: {model_path}")
            model_path = os.path.join(model_path, 'checkpoint.pth')
        print(f"Load model from '{model_path}'")
        self.model.load_state_dict(torch.load(model_path))
        
        # MetricEvaluator 초기화
        # evaluator = MetricEvaluator(file_path=os.path.join(folder_path, "site_metrics.txt"))
        # evaluator = MetricEvaluator(
        #     save_path=os.path.join(result_path, "site_metrics.txt"),
        #     dataset_name=self.args.data,
        #     data_type=self.args.data_type,
        #     ref_mse_path=self.args.ref_mse_path,
        #     )

        pred_list = []
        true_list = []
        input_list = []
        transfer_flag = True if (self.args.num_freeze_layers > 0) or self.args.linear_probe or self.args.is_fully_finetune else False
        print(f'Transfer learning flag: {transfer_flag}')
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, inst_id) in tqdm(enumerate(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
              
                if 'Linear' in self.args.model or 'TST' in self.args.model or self.args.model == 'LSTM':
                    outputs = self.model(batch_x, transfer_flag)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                outputs = outputs[:, -self.args.pred_len:, -1:]
                batch_y = batch_y[:, -self.args.pred_len:, -1:].to(self.device)
                
                # numpy 변환 및 inverse transform
                outputs_np = outputs.detach().cpu().numpy()
                batch_y_np = batch_y.detach().cpu().numpy()
                batch_x_np = batch_x.detach().cpu().numpy()
                inst_id_np = inst_id.cpu().numpy()
                
                # inverse transform 적용
                input_seq = test_data.inverse_transform(batch_x_np, inst_id_np)
                pred = test_data.inverse_transform(outputs_np, inst_id_np)
                true = test_data.inverse_transform(batch_y_np, inst_id_np)

                # input_seq = batch_x_np
                # pred = outputs_np
                # true = batch_y_np

                # denormalized 데이터로 평가 수행
                # evaluator.update(inst_id=inst_id_np, preds=pred, targets=true)
                
                pred_list.append(pred)
                true_list.append(true)
                input_list.append(input_seq)

                if i % 10 == 0:
                    # self.plot_predictions(i, batch_x_np[0, -5:, -1], batch_y_np[0], outputs_np[0], folder_path)
                    self.plot_predictions_2(i,
                                          input_seq[0, -3:, -1],    # 마지막 5개 입력값
                                          true[0],                  # 실제값
                                          pred[0],                  # 예측값
                                          result_path, 
                                          target_dataset,
                                          run_name)
        # print(f"Plotting complete. Results saved in {folder_path}")

        # TODO: metric 계산하는거 개선해야 함.
        # metric 계산 및 결과 출력
        # results, overall_mape = evaluator.evaluate_scale_metrics()

        # for scale_name, (mae, rmse, mbe, r2, mse, skill_score), site_ids in results:
        #     print(f'\nScale: {scale_name}')
        #     print(f"Sites: {site_ids}\n")
        #     print(f"Number of sites: {len(site_ids)}\n")
        #     print(f'MAE: {mae:.4f} kW')
        #     print(f'RMSE: {rmse:.4f} kW')
        #     print(f'MBE: {mbe:.4f} kW')
        #     print(f'R2 Score: {r2:.4f}')
        #     print(f'MSE: {mse:.4f}')
        #     if skill_score is not None:
        #         print(f"Skill Score: {skill_score:.4f}\n")
        # print(f'\nOverall MAPE: {overall_mape:.4f}%')

        # # wandb logging (설정된 경우)
        # if self.args.wandb and (not self.args.distributed or self.args.rank == 0):
        #     self._set_wandb(result_path)
        #     config = {
        #         "model": self.args.model,
        #         "num_parameters": sum(p.numel() for p in self.model.parameters()),
        #         "batch_size": self.args.batch_size,
        #         "num_workers": self.args.num_workers,
        #         "learning_rate": self.args.learning_rate,
        #         "loss_function": self.args.loss,
        #         "dataset": self.args.data,
        #         "epochs": self.args.train_epochs,
        #         "input_seqeunce_length": self.args.seq_len,
        #         "prediction_sequence_length": self.args.pred_len,
        #         "patch_length": self.args.patch_len,
        #         "stride": self.args.stride,
        #         "num_freeze_layers": self.args.num_freeze_layers,
        #     }
        #     upload_files_to_wandb(
        #         project_name=self.project_name,
        #         run_name=self.run_name,
        #         config=config
        #     )        

        #     for scale_name, (mae, rmse, mbe, r2, mse, skill_score), site_ids in results:
        #         wandb.log({
        #             f"test/{scale_name}/Sites": site_ids,
        #             f"test/{scale_name}/MAE": mae,
        #             f"test/{scale_name}/RMSE": rmse,
        #             f"test/{scale_name}/MBE": mbe,
        #             f"test/{scale_name}/R2_Score": r2,
        #             f"test/{scale_name}/MSE": mse,
        #             f"test/{scale_name}/Skill_Score": skill_score if skill_score is not None else 'N/A',
        #             f"test/{scale_name}/MAPE": overall_mape
        #         })
        #     wandb.log({"test/MAPE": overall_mape})
