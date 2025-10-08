# ===================== 导入语句 =====================
# PyTorch 相关
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from torch.serialization import add_safe_globals

# PyTorch Geometric 相关
#import torch_scatter, torch_sparse
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv, TAGConv, GCNConv, SAGEConv
from torch_geometric.explain import Explainer, GNNExplainer
from torch.nn import Linear

# 添加安全全局变量
add_safe_globals([torch_geometric.data.data.DataEdgeAttr])

# 数据处理相关
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import KDTree
import networkx as nx

# 工具库
import random
import time
import os
import sys
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque, namedtuple
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
import threading
import psutil
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

# 自定义RMSE损失函数
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))

class SWELoss(nn.Module):
    """
    Hypsome-SWE 约束的增强损失函数。

    思想：
    - 基础项：预测与真值的点级误差（RMSE）。
    - 物理项：按流域将节点按高程分箱，使用预测的体积加权累计 SWE 曲线
      与参考曲线（SNODAS 历史中位曲线或类比年曲线）之间的差异作为正则。

    使用方法：
    - 在初始化时可传入 `reference_curves`，其为 {basin_id: Tensor[num_bins]}。
      该张量应为 [0,1] 的单调非降累计值（已归一化）。
    - 若没有参考或缺少高程/流域信息，则自动退化为纯 RMSE。
    - 支持在 `forward` 调用时传入 `batch`（PyG 的 mini-batch Data），
      以便读取 `batch.x` 中的高程以及 `batch.<basin_id_attr>` 中的流域 id。

    可配置参数：
    - base_weight: 基础 RMSE 权重
    - hypsometry_weight: Hypsome-SWE 物理项权重
    - elevation_feature_index: 高程在 `x` 中的列索引；若为 None，将尝试从
      `batch.elevation` 或 `batch.elev` 属性读取。
    - area_feature_index: 每个节点网格面积在 `x` 中的列索引；缺省时按等面积处理。
    - num_bins: 高程分箱数。
    - reference_curves: {basin_id: Tensor[num_bins]} 的参考累计曲线（0-1 归一）。
    """

    def __init__(
        self,
        base_weight: float = 1.0,
        hypsometry_weight: float = 0.1,
        elevation_feature_index: int = None,
        basin_id_attr: str = 'grid_id',
        area_feature_index: int = None,
        num_bins: int = 20,
        reference_curves: Dict[int, torch.Tensor] = None,
    ):
        super().__init__()
        self.base_weight = base_weight
        self.hypsometry_weight = hypsometry_weight
        self.elevation_feature_index = elevation_feature_index
        self.area_feature_index = area_feature_index
        self.num_bins = num_bins
        self.reference_curves = reference_curves or {}

        self._mse = nn.MSELoss()
        self._warned_missing_context = False

    def _get_elevation(self, batch: Data, seeds: int, index: int) -> torch.Tensor:
        # 优先从属性读取
        return batch.x[:seeds, index]

    def _get_area(self, batch: Data, seeds: int) -> torch.Tensor:
        # 默认等面积
        return torch.ones(seeds, device=batch.x.device, dtype=torch.float32)

    def _weighted_rmse(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        count0 = torch.sum(y_true == 0).item()
        count1 = torch.sum((y_true > 0) & (y_true <= 25)).item()
        count2 = torch.sum(y_true > 25).item()

        total_count = count0 + count1 + count2
        weight0 = total_count / (count0 + 1e-8)
        weight1 = total_count / (count1 + 1e-8)
        weight2 = total_count / (count2 + 1e-8)

        weights = torch.ones_like(y_true)
        weights[y_true == 0] = weight0
        weights[(y_true > 0) & (y_true <= 25)] = weight1
        weights[y_true > 25] =  weight2
        #print(weight0, weight1, weight2)

        # normalization to avoid heavy weight on high values
        weights = weights / torch.mean(weights)
        loss = weights * (y_pred - y_true) ** 2
        return torch.sqrt(torch.mean(loss))

    def _smooth_weighted_rmse(self, y_pred: torch.Tensor, y_true: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
        weights = 1.0 + alpha * torch.exp(beta * y_true)
        weights = torch.clamp(weights, max=30.0)
        weights = weights / torch.mean(weights)
        loss = weights * (y_pred - y_true) ** 2
        return torch.sqrt(torch.mean(loss))

    def _compute_cumulative_curve(
        self,
        elevation: torch.Tensor,
        swe_volume: torch.Tensor, 
        num_bins: int,
    ) -> torch.Tensor:
        # 以 batch 内 min-max 建立分箱，得到累计体积分布（0-1 归一）
        elev_min = torch.min(elevation)
        elev_max = torch.max(elevation)
        # 防止边界相等
        if torch.isclose(elev_min, elev_max):
            elev_max = elev_min + 1.0
        bin_edges = torch.linspace(elev_min, elev_max, steps=num_bins + 1, device=elevation.device)

        total_volume = torch.sum(swe_volume)
        if total_volume.item() == 0.0:
            total_volume = torch.tensor(1e-8, device=swe_volume.device)  # Prevent division by zero

        # Compute volume for each bin
        bin_volumes = []
        for i in range(1, len(bin_edges)):
            mask = (elevation > bin_edges[i - 1]) & (elevation <= bin_edges[i])
            bin_volumes.append(torch.sum(swe_volume[mask]))
        bin_volumes = torch.stack(bin_volumes)

        # Compute normalized cumulative curve
        cumulative_curve = torch.cumsum(bin_volumes, dim=0) / total_volume

        return cumulative_curve

    def _hypsometry_discrepancy(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        batch: Data,
        seeds: int,
    ) -> torch.Tensor:
        """
        Compute hypsometry discrepancy loss between predicted SWE-volume distribution and reference cumulative elevation curves.
        Prints detailed debug information to console.
        """
    
        #print("\n[HypsometryDiscrepancy] >>> Starting computation")
        #print(f"  -> y_pred shape: {tuple(y_pred.shape)}, seeds: {seeds}")
    
        # Get elevation
        elevation = self._get_elevation(batch, seeds, self.elevation_feature_index)
        if elevation is None:
            print("  [Skip] Elevation data unavailable.")
            return torch.tensor(0.0, device=y_pred.device)
        #print(f"  -> Elevation shape: {tuple(elevation.shape)}")
    
        # Get area
        area = self._get_area(batch, seeds)
        #print(f"  -> Area shape: {tuple(area.shape)}")
    
        # SWE volume
        swe_volume = y_pred * area
        swe_volume_true = y_true * area
        #print("  -> SWE volume computed (kept differentiable).")
    
        # Compute cumulative curve
        curve_pred = self._compute_cumulative_curve(
            elevation, swe_volume, self.num_bins
        )
        #print(f"    -> Predicted curve computed (len={curve_pred.numel()})")
        #print(curve_pred)

        # Reference curve
        ref_curve = self._compute_cumulative_curve(
            elevation, swe_volume_true, self.num_bins
        )
        #print(f"    -> Reference curve computed (len={ref_curve.numel()})")
        #print(ref_curve)
        ref_curve = ref_curve.to(curve_pred.device)

        # Compute L2 difference
        loss = torch.mean((curve_pred - ref_curve) ** 2)

        #print(f"\n[HypsometryDiscrepancy] <<< Final loss: {loss.item():.6f}\n")
        return loss

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, batch: Data = None) -> torch.Tensor:
        # 基础 RMSE
        #base = torch.sqrt(self._mse(y_pred, y_true))
        base = self._weighted_rmse(y_pred, y_true)
        #base = self._smooth_weighted_rmse(y_pred, y_true, alpha=1.0, beta=0.1)

        # Hypsometry 物理项（可选）
        phys = torch.tensor(0.0, device=y_pred.device)
        if batch is not None and self.hypsometry_weight > 0.0:
            try:
                seeds = batch.batch_size if hasattr(batch, 'batch_size') else y_pred.shape[0]
                phys = self._hypsometry_discrepancy(y_pred, y_true, batch, seeds)
                #print(base)
                #print(phys)
                #sys.exit(1)
            except Exception as e:
                if not self._warned_missing_context:
                    print(f"[SWELoss] Hypsometry term disabled this step due to: {str(e)}", flush=True)
                    self._warned_missing_context = True

        return self.base_weight * base + self.hypsometry_weight * phys

class SWEPhaseConstraintLoss(nn.Module):
    """
    SWE 物理相位约束损失：
    1) 积累期（accumulation）：ΔSWE 不得超过当步降水 δP
       -> 惩罚项: ReLU( (SWE_t_pred - SWE_{t-1}) - P_t )
    2) 消融期（melt）：在无降水条件下（P_t <= 阈值），SWE 不得上升
       -> 惩罚项: 1_{P_t<=eps} * ReLU( SWE_t_pred - SWE_{t-1} )

    说明：
    - 该损失支持与基础 RMSE/MSE 组合。
    - 需要 batch 中能提供上一时刻 SWE（prev_swe）以及本时刻降水（precip）。
      既可通过 batch.<attr> 提供，也可在 x 特征中通过列索引提供。
    - 若缺少所需信息，将自动禁用相位约束项（仅返回基础项），并仅警告一次。
    """

    def __init__(
        self,
        base_weight: float = 1.0,
        accumulation_weight: float = 1.0,
        melt_weight: float = 1.0,
        use_rmse: bool = True,
        precip_threshold: float = 1e-6,
        # 优先按属性名读取，若缺失则尝试按特征列索引从 x 读取
        precip_attr: str = 'precip',
        prev_swe_attr: str = 'prev_swe',
        precip_feature_index: int = None,
        prev_swe_feature_index: int = None,
    ):
        super().__init__()
        self.base_weight = base_weight
        self.accumulation_weight = accumulation_weight
        self.melt_weight = melt_weight
        self.use_rmse = use_rmse
        self.precip_threshold = float(precip_threshold)

        self.precip_attr = precip_attr
        self.prev_swe_attr = prev_swe_attr
        self.precip_feature_index = precip_feature_index
        self.prev_swe_feature_index = prev_swe_feature_index

        self._mse = nn.MSELoss()
        self._warned_missing_context = False

    def _get_vector_from_attr_or_x(self, batch: Data, seeds: int, attr_name: str, feature_index: int) -> torch.Tensor:
        vec = None
        if hasattr(batch, attr_name):
            vec = getattr(batch, attr_name)
        elif feature_index is not None and hasattr(batch, 'x'):
            vec = batch.x[:seeds, feature_index]
        if isinstance(vec, torch.Tensor):
            return vec[:seeds].float()
        return None

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, batch: Data = None) -> torch.Tensor:
        # 基础项
        if self.use_rmse:
            base = torch.sqrt(self._mse(y_pred, y_true))
        else:
            base = self._mse(y_pred, y_true)

        acc_penalty = torch.tensor(0.0, device=y_pred.device)
        melt_penalty = torch.tensor(0.0, device=y_pred.device)

        if batch is not None:
            try:
                seeds = batch.batch_size if hasattr(batch, 'batch_size') else y_pred.shape[0]
                prev_swe = self._get_vector_from_attr_or_x(batch, seeds, self.prev_swe_attr, self.prev_swe_feature_index)
                precip = self._get_vector_from_attr_or_x(batch, seeds, self.precip_attr, self.precip_feature_index)

                if prev_swe is None or precip is None:
                    if not self._warned_missing_context:
                        print("[SWEPhaseConstraintLoss] Missing prev_swe or precip; phase constraints disabled for this step.", flush=True)
                        self._warned_missing_context = True
                else:
                    # 仅使用种子节点预测
                    swe_t_pred = y_pred[:seeds]

                    # 保障数值合理
                    precip = precip.clamp_min(0.0)

                    delta_swe = swe_t_pred - prev_swe

                    # 1) 积累期约束：ΔSWE 不可超过降水
                    # 仅对正向增长部分施加约束
                    acc_viol = F.relu(delta_swe - precip)
                    # 不应该提.mean()
                    acc_penalty = acc_viol.mean()

                    # 2) 消融期约束：无降水时 SWE 不得上升
                    # 如果今天p_amount_1 ==0 and p_amount = 0 or temperature >= 0, delta_swe <= 0
                    no_precip_mask = (precip <= self.precip_threshold).float()
                    melt_viol = no_precip_mask * F.relu(delta_swe)
                    # 若全为零，mean() 为 0
                    melt_penalty = melt_viol.mean()
                    # 3) if current swe_value < 0, penalty F.relu(0, -swe_value)
                    # 4) if current temperature < 0, \delta swe >= 0
            except Exception as e:
                if not self._warned_missing_context:
                    print(f"[SWEPhaseConstraintLoss] Disabled this step due to: {str(e)}", flush=True)
                    self._warned_missing_context = True

        return (
            self.base_weight * base
            + self.accumulation_weight * acc_penalty
            + self.melt_weight * melt_penalty
        )

# ===================== 配置参数 =====================
CONFIG = {
    'data_path': '/groups/ESS/whung/swe_gnn/data/gnn_training_data.pt',
    'model_params': {
        'in_channels': 88, # 88
        'hidden_channels': 64,
        'out_channels': 1,
        'num_heads': 4,
        'K': 3
    },
    'training_params': {
        'batch_size': 512,
        'num_neighbors': [16, 16],
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'epochs': 100,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'seed': 42,
        'neighbors_by_model': {
            'SimplifiedGraphSAGE': [16, 16],
            'GraphSAGE': [16, 16, 16],
            'EnhancedGraphSAGE': [16, 16, 16, 16],
            'GraphSAGE_v2': [16, 16, 16, 16, 16, 16] # [25,20,15,10,5,5]
        }
    },
    'visualization': {
        'save_dir': '/groups/ESS/whung/swe_gnn/results_sweloss',
        'figsize': (12, 8),
        'dpi': 300
    }
}

# ===================== 模型定义 =====================
# class GAT_TAG_Model(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, K=3):
#         super(GAT_TAG_Model, self).__init__()

#         # 特征提取层
#         self.feature_extract = nn.Sequential(
#             nn.Linear(in_channels, hidden_channels),
#             nn.ReLU(),
#             nn.Dropout(0.2)
#         )

#         # GAT 层：处理空间关系
#         self.gat1 = GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=True)
#         self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False)

#         # TAGConv 层：处理时间特征
#         self.tag1 = TAGConv(hidden_channels, hidden_channels, K=K)
#         self.tag2 = TAGConv(hidden_channels, hidden_channels, K=K)

#         # 特征融合层
#         self.fusion = nn.Sequential(
#             nn.Linear(hidden_channels * 2, hidden_channels),
#             nn.ReLU(),
#             nn.Dropout(0.2)
#         )

#         # MLP 回归层
#         self.linear1 = Linear(hidden_channels, 32)
#         self.linear2 = Linear(32, out_channels)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
        
#         # 确保输入特征是正确的张量格式
#         if isinstance(x, list):
#             x = torch.stack(x)
#         x = x.float()  # 确保数据类型为float
        
#         # 特征提取
#         x = self.feature_extract(x)
        
#         # GAT 处理空间信息
#         x_spatial = self.gat1(x, edge_index)
#         x_spatial = F.elu(x_spatial)
#         x_spatial = self.gat2(x_spatial, edge_index)
#         x_spatial = F.elu(x_spatial)

#         # TAGConv 处理时间特征
#         x_temporal = self.tag1(x, edge_index)
#         x_temporal = F.elu(x_temporal)
#         x_temporal = self.tag2(x_temporal, edge_index)
#         x_temporal = F.elu(x_temporal)

#         # 特征融合
#         x = self.fusion(torch.cat([x_spatial, x_temporal], dim=1))

#         # MLP回归
#         x = self.linear1(x)
#         x = F.relu(x)
#         x = self.linear2(x)
#         return x.squeeze()

# Start with GCN and GraphSAGE

class GCN_Model(torch.nn.Module):
    """
    图卷积网络模型
    用于处理一般的图结构数据，不区分时间和空间关系
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN_Model, self).__init__()
        
        # 增加隐藏层大小
        hidden_channels = hidden_channels * 2
        
        # 特征提取层
        self.feature_extract = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 图卷积层
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # MLP回归层
        self.linear1 = Linear(hidden_channels, 32)
        self.linear2 = Linear(32, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # 确保输入特征是正确的张量格式
        if isinstance(x, list):
            x = torch.stack(x)
        x = x.float()  # 确保数据类型为float
        
        # 特征提取
        x = self.feature_extract(x)
        
        # 图卷积层处理
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        
        # 添加残差连接
        x = x + x3
        
        # 特征融合
        x = self.fusion(torch.cat([x1, x2, x3], dim=1))
        
        # MLP回归
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x.squeeze(-1)

class GraphSAGE_Model(torch.nn.Module):
    """
    GraphSAGE模型
    用于处理一般的图结构数据，不区分时间和空间关系
    使用采样聚合的方式处理邻居信息
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE_Model, self).__init__()
        
        # 特征提取层
        self.feature_extract = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # GraphSAGE层
        self.sage1 = SAGEConv(hidden_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        self.sage3 = SAGEConv(hidden_channels, hidden_channels)
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # MLP回归层
        self.linear1 = Linear(hidden_channels, 32)
        self.linear2 = Linear(32, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # 确保输入特征是正确的张量格式
        if isinstance(x, list):
            x = torch.stack(x)
        x = x.float()  # 确保数据类型为float
        
        # 特征提取
        x = self.feature_extract(x)
        
        # GraphSAGE层处理
        x1 = self.sage1(x, edge_index)
        x1 = F.relu(x1)
        
        x2 = self.sage2(x1, edge_index)
        x2 = F.relu(x2)
        
        x3 = self.sage3(x2, edge_index)
        x3 = F.relu(x3)
        
        # 特征融合
        x = self.fusion(torch.cat([x1, x2, x3], dim=1))
        
        # MLP回归
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x.squeeze(-1)

class EnhancedGraphSAGE_Model(torch.nn.Module):
    """
    增强版 GraphSAGE 模型
    用于处理一般的图结构数据，不区分时间和空间关系
    使用采样聚合的方式处理邻居信息
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(EnhancedGraphSAGE_Model, self).__init__()
        
        # 增加隐藏层大小
        hidden_channels = hidden_channels * 2
        
        # 特征提取层
        self.feature_extract = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LeakyReLU(),  # 使用 LeakyReLU 激活函数
            nn.Dropout(0.3)  # 增加 Dropout 概率
        )
        
        # 增加 GraphSAGE 层数
        self.sage1 = SAGEConv(hidden_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.sage3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.sage4 = SAGEConv(hidden_channels, hidden_channels)
        self.bn4 = nn.BatchNorm1d(hidden_channels)
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_channels * 4, hidden_channels),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        
        # MLP回归层
        self.linear1 = Linear(hidden_channels, 32)
        self.linear2 = Linear(32, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # 确保输入特征是正确的张量格式
        if isinstance(x, list):
            x = torch.stack(x)
        x = x.float()  # 确保数据类型为float
        
        # 特征提取
        x = self.feature_extract(x)
        
        # GraphSAGE层处理
        x1 = self.sage1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1)
        
        x2 = self.sage2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.leaky_relu(x2)
        
        x3 = self.sage3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.leaky_relu(x3)
        
        x4 = self.sage4(x3, edge_index)
        x4 = self.bn4(x4)
        x4 = F.leaky_relu(x4)
        
        # 添加残差连接
        x = x + x4
        
        # 特征融合
        x = self.fusion(torch.cat([x1, x2, x3, x4], dim=1))
        
        # MLP回归
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
        return x.squeeze(-1)

class GraphSAGE_v2(torch.nn.Module):
    """
    更复杂的 GraphSAGE 版本：
    - 更深的 SAGE 层堆叠（6 层）
    - 每层后加入 BatchNorm 与 Dropout
    - 引入跨层残差（每两层一个短残差）
    - 融合各层输出（concat）后过更宽的 MLP
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE_v2, self).__init__()

        widened = hidden_channels * 2
        self.input = nn.Sequential(
            nn.Linear(in_channels, widened),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )

        self.sage1 = SAGEConv(widened, widened)
        self.bn1 = nn.BatchNorm1d(widened)
        self.sage2 = SAGEConv(widened, widened)
        self.bn2 = nn.BatchNorm1d(widened)
        self.sage3 = SAGEConv(widened, widened)
        self.bn3 = nn.BatchNorm1d(widened)
        self.sage4 = SAGEConv(widened, widened)
        self.bn4 = nn.BatchNorm1d(widened)
        self.sage5 = SAGEConv(widened, widened)
        self.bn5 = nn.BatchNorm1d(widened)
        self.sage6 = SAGEConv(widened, widened)
        self.bn6 = nn.BatchNorm1d(widened)

        self.fusion = nn.Sequential(
            nn.Linear(widened * 6, widened * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.4)
        )

        self.head = nn.Sequential(
            nn.Linear(widened * 2, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, out_channels),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if isinstance(x, list):
            x = torch.stack(x)
        x = x.float()

        x0 = self.input(x)

        x1 = self.sage1(x0, edge_index); x1 = self.bn1(x1); x1 = F.leaky_relu(x1)
        x2 = self.sage2(x1, edge_index); x2 = self.bn2(x2); x2 = F.leaky_relu(x2)
        x2 = x2 + x0  # 短残差

        x3 = self.sage3(x2, edge_index); x3 = self.bn3(x3); x3 = F.leaky_relu(x3)
        x4 = self.sage4(x3, edge_index); x4 = self.bn4(x4); x4 = F.leaky_relu(x4)
        x4 = x4 + x1  # 短残差

        x5 = self.sage5(x4, edge_index); x5 = self.bn5(x5); x5 = F.leaky_relu(x5)
        x6 = self.sage6(x5, edge_index); x6 = self.bn6(x6); x6 = F.leaky_relu(x6)
        x6 = x6 + x2  # 短残差

        x_cat = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        x_fused = self.fusion(x_cat)
        out = self.head(x_fused)
        return out.squeeze(-1)

class SimplifiedGraphSAGE_Model(torch.nn.Module):
    """
    简化版 GraphSAGE 模型
    用于处理一般的图结构数据，不区分时间和空间关系
    使用采样聚合的方式处理邻居信息
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimplifiedGraphSAGE_Model, self).__init__()
        
        # 减少隐藏单元数量
        hidden_channels = hidden_channels // 2
        
        # 特征提取层
        self.feature_extract = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1)  # 减少 Dropout 概率
        )
        
        # GraphSAGE层，减少层数
        self.sage1 = SAGEConv(hidden_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        
        # 特征融合层，减少神经元数量
        self.fusion = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # MLP回归层
        self.linear1 = Linear(hidden_channels, 16)  # 减少神经元数量
        self.linear2 = Linear(16, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # 确保输入特征是正确的张量格式
        if isinstance(x, list):
            x = torch.stack(x)
        x = x.float()  # 确保数据类型为float
        
        # 特征提取
        x = self.feature_extract(x)
        
        # GraphSAGE层处理
        x1 = self.sage1(x, edge_index)
        x1 = F.relu(x1)
        
        x2 = self.sage2(x1, edge_index)
        x2 = F.relu(x2)
        
        # 特征融合
        x = self.fusion(torch.cat([x1, x2], dim=1))
        
        # MLP回归
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x.squeeze(-1)

# ===================== 数据处理函数 =====================
def split_masks(data: Data, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42) -> None:
    """在数据上划分训练、验证和测试集"""
    num_nodes = data.num_nodes
    torch.manual_seed(seed)
    perm = torch.randperm(num_nodes, device=data.x.device)

    n_train = int(train_ratio * num_nodes)
    n_val = int(val_ratio * num_nodes)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

def build_loaders(data: Data, num_neighbors: List[int], batch_size: int) -> Tuple[NeighborLoader, NeighborLoader, NeighborLoader]:
    """构建训练、验证和测试数据加载器"""
    
    # 确保所有特征数据都是张量格式
    # if not isinstance(data.x, torch.Tensor):
    #     data.x = torch.tensor(data.x, dtype=torch.float)
    # if not isinstance(data.edge_index, torch.Tensor):
    #     data.edge_index = torch.tensor(data.edge_index, dtype=torch.long)
    # if not isinstance(data.y, torch.Tensor):
    #     data.y = torch.tensor(data.y, dtype=torch.float)
    
    # # 确保掩码是张量格式
    # if not isinstance(data.train_mask, torch.Tensor):
    #     data.train_mask = torch.tensor(data.train_mask, dtype=torch.bool)
    # if not isinstance(data.val_mask, torch.Tensor):
    #     data.val_mask = torch.tensor(data.val_mask, dtype=torch.bool)
    # if not isinstance(data.test_mask, torch.Tensor):
    #     data.test_mask = torch.tensor(data.test_mask, dtype=torch.bool)
    
    # 创建数据加载器
    train_loader = NeighborLoader(
        data,
        input_nodes=data.train_mask,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True,
        #transform=lambda data: data.to(data.x.device)  # 确保数据在正确的设备上
    )
    
    val_loader = NeighborLoader(
        data,
        input_nodes=data.val_mask,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=False,
        #transform=lambda data: data.to(data.x.device)
    )
    
    test_loader = NeighborLoader(
        data,
        input_nodes=data.test_mask,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=False,
        #transform=lambda data: data.to(data.x.device)
    )
    
    return train_loader, val_loader, test_loader

# ===================== 评估指标 =====================
def calculate_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
    """
    计算RMSE, R²指标
    
    参数:
        y_pred: 预测值
        y_true: 真实值
    
    返回:
        包含RMSE, R²的字典
    """
    # 使用PyTorch的var计算R²
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    var_y = torch.var(y_true, unbiased=False)
    var_res = torch.var(y_true - y_pred, unbiased=False)
    
    # 处理零方差的情况
    if var_y == 0:
        if torch.allclose(y_pred, y_true):
            r2 = torch.tensor(1.0, device=device)  # 如果预测值完全等于真实值，R²为1
        else:
            r2 = torch.tensor(0.0, device=device)  # 如果真实值都是相同的，但预测值不同，R²为0
    else:
        r2 = torch.tensor(1 - (var_res / var_y), device=device)

    # 计算RMSE
    var_rmse = torch.sqrt(F.mse_loss(y_pred, y_true))
    rmse = torch.tensor(var_rmse, device=device)

    # print(f'var_y = {var_y}, var_res = {var_res}')
    # print(f'y_pred shape = {y_pred.shape}, y_true shape = {y_true.shape}')
    # print(f'y_pred mean = {y_pred.mean()}, y_true mean = {y_true.mean()}')
    
    return {
        'rmse': rmse.item(),
        'r2': r2.item()
    }

def plot_model_results(y_pred: torch.Tensor, y_true: torch.Tensor, model_name: str = "Model", save_path: str = None):
    ## move tensor to cpu before plotting
    X = y_true.cpu().numpy()
    Y = y_pred.cpu().numpy()

    tick_limit = np.ceil(np.max([X.max(), Y.max()]))

    plt.figure(figsize=(10, 10))
    plt.plot(np.arange(tick_limit),
        np.arange(tick_limit),
        linewidth=1,
        linestyle=':',
        color='k'
    )
    plt.scatter(X, Y, s=30)
    plt.title(f'Result Comparison ({model_name})')
    plt.xlabel('Observation')
    plt.ylabel(f'Model: {model_name}')
    plt.xlim([0, tick_limit])
    plt.ylim([0, tick_limit])
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def train_step(model: torch.nn.Module,
               dataloader: NeighborLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Dict[str, float]:
    """单步训练"""
    model.train()
    total_loss = 0.0
    total_seeds = 0
    all_preds = []
    all_targets = []

    #i = 1

    for batch in dataloader:
        # 确保数据在GPU上
        batch = batch.to(device)

        # 确保所有数据都是张量格式
        if not isinstance(batch.x, torch.Tensor):
            batch.x = torch.tensor(batch.x, dtype=torch.float, device=device)
        if not isinstance(batch.edge_index, torch.Tensor):
            batch.edge_index = torch.tensor(batch.edge_index, dtype=torch.long, device=device)
        if not isinstance(batch.y, torch.Tensor):
            batch.y = torch.tensor(batch.y, dtype=torch.float, device=device)
        
        optimizer.zero_grad()

        out = model(batch)
        seeds = batch.batch_size
        y_pred = out[:seeds]
        y_true = batch.y[:seeds]

        # 支持带 batch 的 SWELoss；若不支持则退化为普通调用
        try:
            loss = loss_fn(y_pred, y_true, batch)
        except TypeError:
            loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * seeds
        total_seeds += seeds
        
        all_preds.append(y_pred.detach())
        all_targets.append(y_true.detach())
        # print(f'batch y_pred = {y_pred}, y_true = {y_true}')
        # print(f'all_targets_{i}st = {all_targets}')
        # print(f'all_preds_{i}st = {all_preds}')
        # print("--------------------------------")
        # print(f'<<<   {i}     >>>>')
        # print('--------------------------------')
        # i += 1

    # 将所有批次的预测值和真实值连接起来
    #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    #print(f'all_preds = {all_preds}')
    #print(f'all_targets = {all_targets}')

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 计算评估指标
    #print("--------------------------------")
    #print(f'all_preds shape = {all_preds.shape}, all_targets shape = {all_targets.shape}')
    metrics = calculate_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / total_seeds

    return metrics

def eval_step(model: torch.nn.Module,
              dataloader: NeighborLoader,
              loss_fn: torch.nn.Module,
              device: torch.device,
              model_name: str = "Model",
              save_path: str = None,
              plot_option: bool = False) -> Dict[str, float]:
    """单步评估"""
    model.eval()
    total_loss = 0.0
    total_seeds = 0
    all_preds = []
    all_targets = []

    with torch.inference_mode():
        for batch in dataloader:
            # 确保数据在GPU上
            batch = batch.to(device)
            out = model(batch)
            seeds = batch.batch_size
            y_pred = out[:seeds]
            y_true = batch.y[:seeds]

            try:
                loss = loss_fn(y_pred, y_true, batch)
            except TypeError:
                loss = loss_fn(y_pred, y_true)
            total_loss += loss.item() * seeds
            total_seeds += seeds
            
            all_preds.append(y_pred)
            all_targets.append(y_true)

        # 将所有批次的预测值和真实值连接起来
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        # 计算评估指标
        metrics = calculate_metrics(all_preds, all_targets)
        metrics['loss'] = total_loss / total_seeds

        if plot_option == True:
            plot_model_results(
                all_preds,
                all_targets,
                model_name,
                save_path
            )

    return metrics

def train(model: torch.nn.Module,
          train_loader: NeighborLoader,
          val_loader: NeighborLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List[float]]:
    """完整训练流程"""
    history = {
        "train_loss": [], "val_loss": [],
        "train_r2": [], "val_r2": []
    }
    model.to(device)

    for epoch in range(1, epochs + 1):
        # 训练阶段
        train_metrics = train_step(model, train_loader, loss_fn, optimizer, device)
        history["train_loss"].append(train_metrics['loss'])
        history["train_r2"].append(train_metrics['r2'])

        # 验证阶段
        #val_metrics = eval_step(model, val_loader, loss_fn, device)
        val_metrics = eval_step(
            model,
            val_loader,
            loss_fn,
            device,
            '',
            '',
            plot_option=False
        )
        history["val_loss"].append(val_metrics['loss'])
        history["val_r2"].append(val_metrics['r2'])

        print(f"Epoch {epoch:03d} | "
              f"Train Loss (RMSE): {train_metrics['loss']:.4f} | "
              f"Val Loss (RMSE): {val_metrics['loss']:.4f} | "
              f"Train R²: {train_metrics['r2']:.4f} | "
              f"Val R²: {val_metrics['r2']:.4f}")
        #break
    return history

# ===================== 可视化函数 =====================
def plot_training_history(histories: Dict[str, Dict[str, List[float]]], save_path: str = None):
    """绘制训练历史"""
    plt.figure(figsize=(15, 5))  # 调整图形大小以适应两个子图
    
    # 绘制损失
    plt.subplot(1, 2, 1)  # 修改为1行2列
    for model_name, history in histories.items():
        plt.plot(history['train_loss'], label=f'{model_name} (Train)')
        plt.plot(history['val_loss'], label=f'{model_name} (Val)')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制R²
    plt.subplot(1, 2, 2)  # 修改为1行2列
    for model_name, history in histories.items():
        plt.plot(history['train_r2'], label=f'{model_name} (Train)')
        plt.plot(history['val_r2'], label=f'{model_name} (Val)')
    plt.title('Training and Validation R²')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_model_comparison(test_metrics: Dict[str, Dict[str, float]], save_path: str = None):
    """绘制模型对比柱状图"""
    plt.figure(figsize=(15, 5))
    
    metrics = ['r2', 'loss']
    models = list(test_metrics.keys())
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 2, i)
        values = [test_metrics[model][metric] for model in models]
        plt.bar(models, values)
        plt.title(f'Model Comparison ({metric.upper()})')
        plt.xlabel('Model')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        
        # 在柱状图上添加具体数值
        for j, v in enumerate(values):
            plt.text(j, v, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

# ===================== 模型解释函数 =====================
def explain_model(model: torch.nn.Module,
                 data: Data,
                 target_node: int,
                 device: torch.device,
                 epochs: int = 100) -> Dict[str, torch.Tensor]:
    """
    使用 GNNExplainer 解释模型对特定节点的预测
    
    参数:
        model: 训练好的模型
        data: 图数据
        target_node: 要解释的目标节点索引
        device: 计算设备
        epochs: 解释器训练的轮数
    
    返回:
        包含节点重要性和边重要性的字典
    """
    model.eval()
    
    # 创建一个包装类来适配explainer的调用方式
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x, edge_index, **kwargs):
            data = Data(x=x, edge_index=edge_index)
            return self.model(data)
    
    wrapped_model = ModelWrapper(model)
    
    explainer = Explainer(
        model=wrapped_model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='node',
            return_type='raw',
        ),
    )
    
    # 获取目标节点的解释
    explanation = explainer(
        data.x,
        data.edge_index,
        index=target_node
    )
    
    return {
        'node_mask': explanation.node_mask,
        'edge_mask': explanation.edge_mask
    }

def visualize_explanation(data: Data,
                         explanation: Dict[str, torch.Tensor],
                         target_node: int,
                         save_path: str = None):
    """
    可视化模型解释结果
    
    参数:
        data: 图数据
        explanation: 解释结果字典
        target_node: 目标节点索引
        save_path: 保存图像的路径
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # 创建图
    G = nx.Graph()
    
    # 添加节点
    node_mask = explanation['node_mask']
    if len(node_mask.shape) > 1:
        # 如果node_mask是二维的，取平均值
        node_mask = node_mask.mean(dim=1)
    
    # 确保target_node是整数
    target_node = int(target_node)
    
    # 只添加与目标节点相关的节点和边
    edge_index = data.edge_index.cpu().numpy()
    edge_mask = explanation['edge_mask']
    if len(edge_mask.shape) > 1:
        edge_mask = edge_mask.mean(dim=1)
    
    # 找到与目标节点相关的边
    relevant_edges = []
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        if src == target_node or dst == target_node:
            relevant_edges.append((int(src), int(dst), edge_mask[i].item()))
    
    # 添加相关节点
    relevant_nodes = set()
    for src, dst, _ in relevant_edges:
        relevant_nodes.add(src)
        relevant_nodes.add(dst)
    
    # 添加节点到图中
    for node in relevant_nodes:
        G.add_node(node, importance=node_mask[node].item())
    
    # 添加边到图中
    for src, dst, weight in relevant_edges:
        G.add_edge(src, dst, weight=weight)
    
    # 绘制图
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    
    # 绘制节点
    node_importance = [G.nodes[n]['importance'] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_importance,
                          cmap=plt.cm.Reds,
                          node_size=100)
    
    # 绘制边
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos,
                          width=edge_weights,
                          alpha=0.5)
    
    # 高亮目标节点
    nx.draw_networkx_nodes(G, pos,
                          nodelist=[target_node],
                          node_color='blue',
                          node_size=200)
    
    plt.title(f'Model Explanation for Node {target_node}')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def analyze_feature_importance(model: torch.nn.Module,
                             data: Data,
                             device: torch.device) -> torch.Tensor:
    """
    分析特征重要性
    
    参数:
        model: 训练好的模型
        data: 图数据
        device: 计算设备
    
    返回:
        特征重要性得分
    """
    model.eval()
    feature_importance = torch.zeros(data.x.size(1), device=device)
    
    # 确保数据在GPU上
    data = data.to(device)
    
    # 对每个特征进行扰动
    for i in range(data.x.size(1)):
        # 保存原始特征
        original_feature = data.x[:, i].clone()
        
        # 扰动特征
        data.x[:, i] = torch.randn_like(data.x[:, i])
        
        # 计算扰动后的预测
        with torch.no_grad():
            perturbed_output = model(data)
        
        # 恢复原始特征
        data.x[:, i] = original_feature
        
        # 计算特征重要性
        with torch.no_grad():
            original_output = model(data)
        
        # 使用预测差异作为特征重要性
        feature_importance[i] = torch.mean(torch.abs(perturbed_output - original_output))
    
    return feature_importance

# 资源监控线程
class ResourceMonitor(threading.Thread):
    def __init__(self, interval=1.0):
        super().__init__()
        self.interval = interval
        self.cpu = []
        self.mem = []
        self.gpu = []
        self.gpu_mem = []
        self.gpu_power = []
        self.timestamps = []
        self.running = False

    def run(self):
        self.running = True
        start_time = time.time()
        while self.running:
            self.timestamps.append(time.time() - start_time)
            self.cpu.append(psutil.cpu_percent())
            self.mem.append(psutil.virtual_memory().percent)
            if NVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
                    self.gpu.append(util.gpu)
                    self.gpu_mem.append(meminfo.used / meminfo.total * 100)
                    self.gpu_power.append(power)
                except Exception:
                    self.gpu.append(0)
                    self.gpu_mem.append(0)
                    self.gpu_power.append(0)
            else:
                self.gpu.append(0)
                self.gpu_mem.append(0)
                self.gpu_power.append(0)
            time.sleep(self.interval)

    def stop(self):
        self.running = False

# ===================== 模型分析函数 =====================
def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    统计模型参数数量
    参数:
        model: 需要统计的模型
    返回:
        包含总参数、可训练参数、不可训练参数数量的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params
    }

def print_model_summary(model: torch.nn.Module, model_name: str = "Model") -> None:
    """
    打印模型的详细参数信息
    参数:
        model: 需要分析的模型
        model_name: 模型名称
    """
    param_stats = count_parameters(model)
    
    print(f"\n=== {model_name} 参数统计 ===")
    print(f"总参数数: {param_stats['total_params']:,}")
    print(f"可训练参数数: {param_stats['trainable_params']:,}")
    print(f"不可训练参数数: {param_stats['non_trainable_params']:,}")
    print(f"模型大小估计: {param_stats['total_params'] * 4 / 1024 / 1024:.2f} MB (假设32位浮点)")
    
    print(f"\n=== {model_name} 层级结构 ===")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape} -> {param.numel():,} 参数")

# ===================== 模型保存函数 =====================
def save_model(model: torch.nn.Module, path: str) -> None:
    """
    保存模型到指定路径
    参数:
        model: 需要保存的模型
        path: 保存路径
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# ===================== 主函数 =====================
def main():
    # 命令行参数：选择运行的模型
    parser = argparse.ArgumentParser(description="Train selected GNN models")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Using comma to add different models (GraphSAGE,EnhancedGraphSAGE,SimplifiedGraphSAGE,GraphSAGE_v2) or 'all'"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to gnn_training_data.pt (override CONFIG['data_path'])"
    )
    args = parser.parse_args()

    # 创建保存目录
    save_dir = Path(CONFIG['visualization']['save_dir'])
    save_dir.mkdir(exist_ok=True)
    
    monitor = None

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    print("Loading data...")
    data_path = args.data_path if args.data_path else CONFIG['data_path']
    try:
        # 尝试直接加载
        graph_data = torch.load(data_path, map_location=device, weights_only=False)
        print("Data loaded successfully with weights_only=False")
    except Exception as e1:
        print(f"First loading attempt failed: {str(e1)}")
        try:
            # 尝试使用map_location
            graph_data = torch.load(data_path, map_location=device)
            print("Data loaded successfully with map_location")
        except Exception as e2:
            print(f"Second loading attempt failed: {str(e2)}")
            try:
                # 尝试使用pickle
                import pickle
                with open(data_path, 'rb') as f:
                    graph_data = pickle.load(f)
                print("Data loaded successfully with pickle")
            except Exception as e3:
                print(f"All loading attempts failed. Last error: {str(e3)}")
                raise
    
    # 打印详细的数据信息
    print("\nDetailed data structure:")
    print(f"Loaded from: {data_path}")
    print(f"Type: {type(graph_data)}")
    print("\nAttributes:")
    
    # 打印基本属性
    print("x:")
    print(f"  Type: {type(graph_data.x)}")
    print(f"  Shape: {graph_data.x.shape}")
    print(f"  Dtype: {graph_data.x.dtype}")
    print(f"  Device: {graph_data.x.device}")
    
    print("\nedge_index:")
    print(f"  Type: {type(graph_data.edge_index)}")
    print(f"  Shape: {graph_data.edge_index.shape}")
    print(f"  Dtype: {graph_data.edge_index.dtype}")
    print(f"  Device: {graph_data.edge_index.device}")
    
    print("\ny:")
    print(f"  Type: {type(graph_data.y)}")
    print(f"  Shape: {graph_data.y.shape}")
    print(f"  Dtype: {graph_data.y.dtype}")
    print(f"  Device: {graph_data.y.device}")
    
    # 打印其他属性
    print("\nOther attributes:")
    for key in ['num_nodes', 'num_edges', 'num_node_features']:
        if hasattr(graph_data, key):
            print(f"{key}: {getattr(graph_data, key)}")
    
    # 确保所有特征数据都是张量格式
    if not isinstance(graph_data.x, torch.Tensor):
        graph_data.x = torch.tensor(graph_data.x, dtype=torch.float)
    if not isinstance(graph_data.edge_index, torch.Tensor):
        graph_data.edge_index = torch.tensor(graph_data.edge_index, dtype=torch.long)
    if not isinstance(graph_data.y, torch.Tensor):
        graph_data.y = torch.tensor(graph_data.y, dtype=torch.float)

    
    # 将数据移动到指定设备
    graph_data = graph_data.to(device)

    grid_id = graph_data.grid_id
    label_encoder = LabelEncoder()
    grid_id_encoded = label_encoder.fit_transform(grid_id)
    grid_id_tensor = torch.tensor(grid_id_encoded, dtype=torch.long)
    graph_data.grid_id = grid_id_tensor
    if 'dates' in graph_data._store:
        del graph_data._store['dates']
    print(f"\nData loaded: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges, {graph_data.num_node_features} features per node")

    for key in graph_data._store.keys():
        value = graph_data._store[key]
        print(f"{key}: {type(value)} | shape: {getattr(value, 'shape', None)}")

    # 数据预处理
    split_masks(graph_data, 
                train_ratio=CONFIG['training_params']['train_ratio'],
                val_ratio=CONFIG['training_params']['val_ratio'],
                seed=CONFIG['training_params']['seed'])

    # 成功加载数据后再启动资源监控，避免出错时挂起
    monitor = ResourceMonitor(interval=1.0)
    monitor.daemon = True
    monitor.start()

    # 确保掩码是张量格式
    if not isinstance(graph_data.train_mask, torch.Tensor):
        graph_data.train_mask = torch.tensor(graph_data.train_mask, dtype=torch.bool)
    if not isinstance(graph_data.val_mask, torch.Tensor):
        graph_data.val_mask = torch.tensor(graph_data.val_mask, dtype=torch.bool)
    if not isinstance(graph_data.test_mask, torch.Tensor):
        graph_data.test_mask = torch.tensor(graph_data.test_mask, dtype=torch.bool)

    # 更新模型参数以匹配数据
    CONFIG['model_params']['in_channels'] = graph_data.num_node_features
    print(f"\nUpdated model parameters:")
    print(f"Input channels: {CONFIG['model_params']['in_channels']}")

    # 数据加载器按模型在训练环中分别构建（基于每个模型的 hops）
    print("\nData loaders will be built per model based on model-specific hops (num_neighbors)...")

    # 定义要训练的模型
    models = {
        #'GAT-TAG': GAT_TAG_Model(**CONFIG['model_params']),

        'GCN': GCN_Model(
            in_channels=CONFIG['model_params']['in_channels'],
            hidden_channels=CONFIG['model_params']['hidden_channels'],
            out_channels=CONFIG['model_params']['out_channels']
        ),
        #'GraphSAGE': GraphSAGE_Model(
        #    in_channels=CONFIG['model_params']['in_channels'],
        #    hidden_channels=CONFIG['model_params']['hidden_channels'],
        #    out_channels=CONFIG['model_params']['out_channels']
        #),
        #'EnhancedGraphSAGE': EnhancedGraphSAGE_Model(
        #    in_channels=CONFIG['model_params']['in_channels'],
        #    hidden_channels=CONFIG['model_params']['hidden_channels'],
        #    out_channels=CONFIG['model_params']['out_channels']
        #),
        #'GraphSAGE_v2': GraphSAGE_v2(
        #    in_channels=CONFIG['model_params']['in_channels'],
        #    hidden_channels=CONFIG['model_params']['hidden_channels'],
        #    out_channels=CONFIG['model_params']['out_channels']
        #),
        #'SimplifiedGraphSAGE': SimplifiedGraphSAGE_Model(
        #    in_channels=CONFIG['model_params']['in_channels'],
        #    hidden_channels=CONFIG['model_params']['hidden_channels'],
        #    out_channels=CONFIG['model_params']['out_channels']
        #)
    }

    # 训练所有模型并记录结果
    histories = {}
    test_metrics = {}

    # 根据命令行选择模型
    selected_models = None
    if args.models and args.models.lower() != 'all':
        selected_models = [m.strip() for m in args.models.split(',') if m.strip()]
        models = {name: mdl for name, mdl in models.items() if name in selected_models}
        if len(models) == 0:
            print(f"No valid models selected from --models={args.models}. Exiting.")
            return

    # 准备每模型 neighbors（优先 neighbors_by_model；否则按层数自动生成）
    neighbors_by_model = CONFIG['training_params'].get('neighbors_by_model', {})
    default_neighbors = CONFIG['training_params'].get('num_neighbors', [16, 16])
    base_neighbor = default_neighbors[0] if isinstance(default_neighbors, list) and len(default_neighbors) > 0 else 16

    def count_message_passing_layers(model: torch.nn.Module) -> int:
        count = 0
        for m in model.modules():
            if isinstance(m, (SAGEConv, GCNConv)):
                count += 1
        return count

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        print(f"Model structure:")
        print(model)
        
        # 打印模型参数统计
        print_model_summary(model, model_name)
        
        # 为该模型构建专属 data loaders（按 hops）
        if model_name in neighbors_by_model:
            num_neighbors_for_model = neighbors_by_model[model_name]
        else:
            L = count_message_passing_layers(model)
            num_neighbors_for_model = default_neighbors if L <= 0 else [base_neighbor] * L

        print(f"{model_name} num_neighbors (hops): {num_neighbors_for_model}")
        train_loader, val_loader, test_loader = build_loaders(
            data=graph_data,
            num_neighbors=num_neighbors_for_model,
            batch_size=CONFIG['training_params']['batch_size']
        )

        # 将模型移到GPU
        model = model.to(device)
        
        # 初始化优化器
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=CONFIG['training_params']['learning_rate'],
            weight_decay=CONFIG['training_params']['weight_decay']
        )
        # 使用增强的 SWELoss（如无参考曲线则自动退化为 RMSE 行为）
        loss_fn = SWELoss(
            base_weight=1.0,
            hypsometry_weight=10.0,  # 可根据需要在 CONFIG 中暴露
            elevation_feature_index=3,  # 若高程在 x 的列索引已知，可填写索引
            area_feature_index=None,
            num_bins=20,
            reference_curves=None,  # 之后可加载 SNODAS/类比年参考曲线后设置
        )
        #loss_fn = nn.MSELoss()
        #loss_fn = SWEPhaseConstraintLoss()

        # 训练模型
        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=CONFIG['training_params']['epochs'],
            device=device
        )
        histories[model_name] = history

        # 测试集评估
        #test_metrics[model_name] = eval_step(model, test_loader, loss_fn, device)
        test_metrics[model_name] = eval_step(
            model,
            test_loader,
            loss_fn,
            device,
            model_name,
            save_dir / f'{model_name}_result_comparison.png',
            plot_option=True
        )
        print(f"{model_name} Test Metrics:")
        print(f"  R²: {test_metrics[model_name]['r2']:.4f}")
        print(f"  RMSE: {test_metrics[model_name]['rmse']:.4f}")

        # 保存模型
        #model_save_path = '/projects/kzhou6/bcui2/git_submit/SWE_25/model/' + f"{model_name}_model.pth"
        #model_save_path = '/groups/ESS/whung/swe_gnn/model/' + f"{model_name}_model_v2.pth"
        model_save_path = '/groups/ESS/whung/swe_gnn/model/' + f"{model_name}_model_sweloss.pth"
        save_model(model, model_save_path)

    # 停止资源监控
    monitor.stop()
    monitor.join()

    # 绘制资源监控图表
    def plot_resource(data, ylabel, title, save_name):
        plt.figure(figsize=(10, 6))
        plt.plot(monitor.timestamps, data)
        plt.xlabel('Time (s)')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / save_name)
        plt.close()

    plot_resource(monitor.cpu, 'CPU Usage (%)', 'CPU Usage Over Time', 'cpu_usage.png')
    plot_resource(monitor.mem, 'Memory Usage (%)', 'Memory Usage Over Time', 'memory_usage.png')
    plot_resource(monitor.gpu, 'GPU Utilization (%)', 'GPU Utilization Over Time', 'gpu_utilization.png')
    plot_resource(monitor.gpu_mem, 'GPU Memory Usage (%)', 'GPU Memory Usage Over Time', 'gpu_memory_usage.png')
    plot_resource(monitor.gpu_power, 'GPU Power (W)', 'GPU Power Over Time', 'gpu_power.png')

    # 绘制训练历史
    plot_training_history(
        histories,
        save_path=save_dir / 'training_history.png'
    )

    # 绘制模型对比
    plot_model_comparison(
        test_metrics,
        save_path=save_dir / 'model_comparison.png'
    )

if __name__ == "__main__":
    main()