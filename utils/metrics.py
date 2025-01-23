import numpy as np
import pandas as pd
import json
from sklearn.metrics import r2_score
from collections import defaultdict

class MetricEvaluator:
    def __init__(self, save_path, dataset_name='DKASC', data_type='all', ref_mse_path=None):
        """
        Args:
            save_path: 결과를 저장할 파일 경로
            dataset_name: 데이터셋 이름
            data_type: 데이터 타입 ('all', 'day' 등)
            ref_mse_path: Reference MSE 파일 경로 (JSON 또는 CSV)
        """
        self.save_path = save_path
        self.data = defaultdict(lambda: {"preds": [], "targets": []})
        self.ref_mse_dict = self._load_reference_mse(ref_mse_path)
        
        # mapping 파일 로드
        mapping_path = f'./data_provider/{dataset_name}_mapping/mapping_{data_type}.csv'
        self.mapping_df = pd.read_csv(mapping_path)
        self.current_dataset = self.mapping_df[self.mapping_df['dataset'] == dataset_name]
        self.current_dataset['index'] = self.current_dataset['mapping_name'].apply(lambda x: int(x.split('_')[0]))

    def _load_reference_mse(self, ref_mse_path):
        """Load reference MSE from file"""
        if ref_mse_path is None:
            return {}
            
        if ref_mse_path.endswith('.json'):
            with open(ref_mse_path, 'r') as f:
                return json.load(f)
        elif ref_mse_path.endswith('.csv'):
            df = pd.read_csv(ref_mse_path)
            return dict(zip(df['scale_name'], df['ref_mse']))
        else:
            raise ValueError("Reference MSE file must be either JSON or CSV")

    def update(self, inst_id, preds, targets):
        """매 배치마다 installation ID와 예측값, 실제값을 저장"""
        for i, site_id in enumerate(inst_id):
            site_id = site_id.item()  # tensor to scalar
            self.data[site_id]["preds"].append(preds[i])
            self.data[site_id]["targets"].append(targets[i])

    def _get_site_capacity(self, site_id):
        """mapping 파일에서 site의 용량 정보 가져오기"""
        file_row = self.current_dataset[self.current_dataset['index'] == site_id]
        if file_row.empty:
            raise ValueError(f"No matching file found for site_id {site_id}")
        file_name = file_row['original_name'].values[0]
        try:
            capacity = float(file_name.split('_')[0])
            return capacity
        except (IndexError, ValueError):
            raise ValueError(f"Invalid capacity format in filename: {file_name}")
    
    def _calculate_metrics(self, preds, targets, scale_name):
        """주어진 그룹의 metric 계산"""
        mae = np.mean(np.abs(preds - targets))
        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        mbe = np.mean(preds - targets)
        r2 = r2_score(targets.flatten(), preds.flatten())
        mse = np.mean((preds - targets) ** 2)
        skill_score = self.calculate_skill_score(scale_name, mse)
        return mae, rmse, mbe, r2, mse, skill_score

    def calculate_mape(self):
        """전체 데이터에 대한 MAPE 계산 - installation별 설치 용량 기준"""
        total_error = 0
        total_samples = 0
        
        for site_id, site_data in self.data.items():
            site_preds = np.concatenate(site_data["preds"])
            site_targets = np.concatenate(site_data["targets"])
            site_capacity = self._get_site_capacity(site_id)
            
            # 각 지점별 절대 오차의 합을 해당 지점의 설치 용량으로 나눔
            epsilon = 1e-10
            site_error = np.sum(np.abs(site_preds - site_targets)) / max(site_capacity, epsilon)
            total_error += site_error
            total_samples += len(site_targets)
        
        mape = (total_error / total_samples) * 100
        return mape
    
    def calculate_skill_score(self, scale_name, current_mse):
        """
        Calculate skill score for a given scale
        SS = 1 - MSE_transfer/MSE_ref
        """
        if scale_name not in self.ref_mse_dict:
            return None
            
        ref_mse = self.ref_mse_dict[scale_name]
        if ref_mse <= 0:  # Avoid division by zero
            return None
            
        skill_score = 1 - (current_mse / ref_mse)
        return skill_score
 
    def get_capacity_groups(self):
        """실제 데이터의 용량을 기반으로 그룹 생성"""
        capacity_ranges = [
            ("Small", 0, 30),
            ("Small-Medium", 30, 100)
        ]
        
        # mapping 파일의 용량 정보를 기반으로 100kW 구간 확인
        large_capacities = []
        for site_id in self.data.keys():
            capacity = self._get_site_capacity(site_id)
            if capacity >= 100:
                group_start = int((capacity // 100) * 100)
                large_capacities.append(group_start)
        
        # 중복 제거하고 정렬
        unique_groups = sorted(set(large_capacities))
        
        # 실제 존재하는 100kW 구간만 추가
        for start in unique_groups:
            capacity_ranges.append((f"{start}kW", start, start + 100))

        return capacity_ranges

    def evaluate_scale_metrics(self):
        """용량 그룹별 metric 계산"""
        results = []
        capacity_ranges = self.get_capacity_groups()
        
        # 각 용량 범위별로 metric 계산
        for group_name, min_cap, max_cap in capacity_ranges:
            group_preds = []
            group_targets = []
            group_sites = set()
            
            # 해당 용량 범위에 속하는 site의 데이터 수집
            for site_id, site_data in self.data.items():
                site_capacity = self._get_site_capacity(site_id)
                
                if min_cap <= site_capacity < max_cap:
                    site_preds = np.concatenate(site_data["preds"])
                    site_targets = np.concatenate(site_data["targets"])
                    group_preds.append(site_preds)
                    group_targets.append(site_targets)
                    group_sites.add(site_id)
            
            if group_preds:  # 데이터가 있는 경우에만 계산
                group_preds = np.concatenate(group_preds)
                group_targets = np.concatenate(group_targets)
                metrics = self._calculate_metrics(group_preds, group_targets, group_name)
                results.append((group_name, metrics, sorted(group_sites)))
        
        # 전체 MAPE 계산
        overall_mape = self.calculate_mape()
        
        # 결과 저장
        self._save_results(results, overall_mape)
        
        return results, overall_mape

    def _save_results(self, results, overall_mape):
        """결과를 파일로 저장"""
        with open(self.save_path, "w") as file:
            file.write("=" * 50 + "\n")
            file.write("Scale-Specific Evaluation Metrics\n")
            file.write("=" * 50 + "\n")
            
            for scale_name, (mae, rmse, mbe, r2, mse, skill_score), site_ids in results:
                file.write(f"Scale: {scale_name}\n")
                file.write(f"Sites: {site_ids}\n")
                file.write(f"Number of sites: {len(site_ids)}\n")
                file.write(f"MAE: {mae:.4f} kW\n")
                file.write(f"RMSE: {rmse:.4f} kW\n")
                file.write(f"MBE: {mbe:.4f} kW\n")
                file.write(f"R2 Score: {r2:.4f}\n")
                file.write(f"MSE Score: {mse:.4f}\n")
                if skill_score is not None:
                    file.write(f"Skill Score: {skill_score:.4f}\n")
                file.write("-" * 50 + "\n")
            
            # 전체 사이트 정보 추가
            all_sites = set()
            for _, _, site_ids in results:
                all_sites.update(site_ids)
            file.write(f"\nTotal number of unique sites: {len(all_sites)}\n")
            file.write(f"All site IDs: {sorted(all_sites)}\n")

            file.write(f"\nOverall MAPE: {overall_mape:.4f}%\n")



# class MetricEvaluator:
#     def __init__(self, file_path):
#         self.file_path = file_path
#         self.preds_list = []
#         self.targets_list = []
#         self.inst_ids = []

#     def update(self, inst_id, preds, targets):
#         """매 배치마다 installation ID와 함께 예측값과 실제값을 누적"""
#         self.inst_ids.append(inst_id)
#         self.preds_list.append(preds)
#         self.targets_list.append(targets)

#     def _calculate_metrics(self, preds, targets):
#         """ 주어진 그룹의 metric 계산 """
#         rmse = np.sqrt(np.mean((preds - targets) ** 2))
#         mae = np.mean(np.abs(preds - targets))
#         mbe = np.mean(preds - targets)
#         r2 = r2_score(targets, preds)
#         return rmse, mae, mbe, r2

#     def calculate_mape(self):
#         """전체 데이터에 대한 MAPE 계산 - installation별 최대값 기준"""
#         total_error = 0
#         total_samples = 0
        
#         for inst_id, inst_preds, inst_targets in zip(self.inst_ids, self.preds_list, self.targets_list):
#             inst_max = np.max(np.abs(inst_targets))
#             epsilon = 1e-10
            
#             inst_error = np.sum(np.abs(inst_preds - inst_targets)) / max(inst_max, epsilon)
#             total_error += inst_error
#             total_samples += len(inst_targets)
        
#         mape = (total_error / total_samples) * 100
#         return mape

#     def generate_scale_groups(self):
#         """용량 그룹 정의"""
#         max_target = np.max(self.targets_list)
#         scale_groups = [("Small", lambda targets: (targets >= 0) & (targets < 30)),
#                         ("Small-Medium", lambda targets: (targets >= 30) & (targets < 100))]

#         # Generate 100kW intervals
#         for i in range(1, int(max_target // 100) + 1):
#             lower_bound = i * 100
#             upper_bound = (i + 1) * 100
#             scale_groups.append((f"{lower_bound}kW",
#                                  lambda targets, lb=lower_bound, ub=upper_bound:
#                                  (targets >= lb) & (targets < ub)))

#         # Add the 1MW group
#         scale_groups.append(("MW", lambda targets: targets >= 1000))
#         return scale_groups

#     def evaluate_scale_metrics(self):
#         """용량 그룹별 metric 계산"""
#         preds = np.concatenate(self.preds_list)
#         targets = np.concatenate(self.targets_list)
#         scale_groups = self.generate_scale_groups()

#         results = []
#         for scale_name, scale_func in scale_groups:
#             mask = scale_func(targets)
#             if isinstance(mask, torch.Tensor):
#                 mask = mask.numpy()
            
#             if np.any(mask):
#                 masked_preds = preds[mask]
#                 masked_targets = targets[mask]
#                 metrics = self._calculate_metrics(masked_preds, masked_targets)
#                 results.append((scale_name, metrics))
#             else:
#                 print(f"No data for scale {scale_name}")

#         # 결과 저장
#         with open(self.file_path, "w") as file:
#             file.write("=" * 50 + "\n")
#             file.write("Scale-Specific Evaluation Metrics\n")
#             file.write("=" * 50 + "\n")
            
#             for scale_name, (rmse, mae, mbe, r2) in results:
#                 file.write(f"Scale: {scale_name}\n")
#                 file.write(f"RMSE: {rmse:.4f} kW\n")
#                 file.write(f"MAE: {mae:.4f} kW\n")
#                 file.write(f"MBE: {mbe:.4f} kW\n")
#                 file.write(f"R2 Score: {r2:.4f}\n")
#                 file.write("=" * 50 + "\n")
            
#             mape = self.calculate_mape()
#             file.write(f"\nOverall MAPE: {mape:.4f}%\n")
            
#         return results, mape