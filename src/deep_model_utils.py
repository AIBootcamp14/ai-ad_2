from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from config import RANDOM_SEED, LSTMAEConfig  # 재현성용 (set_global_seed는 main 쪽에서 이미 호출한다고 가정)


# Scaler / Artifact 정의
@dataclass
class StandardScaler:
    """
    numpy 기반 간단 StandardScaler (mean/std 저장만 함).
    torch에서 그대로 사용 가능.
    """
    mean_: np.ndarray
    std_: np.ndarray

    @classmethod
    def fit(cls, arr: np.ndarray) -> "StandardScaler":
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)

        # 분산 0인 feature 방어
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean_=mean.astype(np.float32), std_=std.astype(np.float32))

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.mean_) / self.std_


@dataclass
class LSTMAEArtifacts:
    """
    학습이 끝난 후 추론에 필요한 모든 정보를 묶어둔 구조체.
    - model: 학습된 LSTM AutoEncoder
    - scaler: train 데이터 기준 StandardScaler
    - window_size / window_stride: 윈도우 생성시 사용한 파라미터
    - threshold: point-wise reconstruction error 기준 이상치 임계값
    """
    model: "LSTMAutoEncoder"
    scaler: StandardScaler
    window_size: int
    window_stride: int
    threshold: float


# Dataset / 시퀀스 유틸 함수들
class SequenceDataset(Dataset):
    """
    (N_windows, T, F) shape의 numpy 배열을 받아
    torch Dataset으로 감싸는 간단한 래퍼.
    """
    def __init__(self, sequences: np.ndarray) -> None:
        super().__init__()
        if sequences.ndim != 3:
            raise ValueError(f"sequences.ndim must be 3, got {sequences.ndim}")
        self.sequences = torch.from_numpy(sequences.astype(np.float32))

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]


def _create_sequences(
    arr: np.ndarray,
    window_size: int,
    window_stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D array (N, F)를 받아 슬라이딩 윈도우로
    3D array (N_windows, T, F)를 생성.

    반환:
        sequences: (N_windows, window_size, F)
        start_indices: 각 window가 원본에서 시작하는 인덱스 (N_windows,)
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    num_points, num_features = arr.shape
    if num_points < window_size:
        raise ValueError(
            f"num_points({num_points}) < window_size({window_size}), "
            "너무 긴 윈도우입니다."
        )

    sequences = []
    start_indices = []
    for start in range(0, num_points - window_size + 1, window_stride):
        end = start + window_size
        sequences.append(arr[start:end])
        start_indices.append(start)

    sequences_np = np.stack(sequences, axis=0)  # (N_windows, T, F)
    start_indices_np = np.asarray(start_indices, dtype=np.int64)
    return sequences_np, start_indices_np


def _aggregate_window_scores_to_point_scores(
    num_points: int,
    window_size: int,
    window_scores: np.ndarray,
    window_starts: np.ndarray,
) -> np.ndarray:
    """
    윈도우 단위 reconstruction error를
    원래 시계열의 각 시점(point) 단위 score로 평균 집계.

    - window_scores: (N_windows,)  각 윈도우의 scalar score
    - window_starts: (N_windows,)  각 윈도우의 시작 인덱스 (0-based)

    반환:
        point_scores: (num_points,) 각 row별 평균 score
    """
    point_scores = np.zeros(num_points, dtype=np.float64)
    counts = np.zeros(num_points, dtype=np.int64)

    for score, start in zip(window_scores, window_starts):
        end = start + window_size
        if end > num_points:
            # 방어적 코드
            end = num_points
        point_scores[start:end] += float(score)
        counts[start:end] += 1

    # 윈도우에 포함되지 않은 위치 방어
    counts[counts == 0] = 1
    point_scores /= counts
    return point_scores.astype(np.float32)


def _create_sequences_by_group(
    arr: np.ndarray,
    group_ids: np.ndarray,
    window_size: int,
    window_stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    group_ids(예: simulationRun) 별로만 슬라이딩 윈도우를 만든다.

    반환:
      sequences: (N_windows, T, F)
      window_point_indices: (N_windows, T)  # 각 윈도우가 참조하는 '원본 row 인덱스'
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    if group_ids.ndim != 1 or len(group_ids) != arr.shape[0]:
        raise ValueError("group_ids must be 1D and same length as arr")

    sequences: list[np.ndarray] = []
    window_indices: list[np.ndarray] = []

    # run id 등장 순서를 보존 (np.unique는 정렬되므로 주의)
    seen = set()
    ordered_runs = []
    for g in group_ids.tolist():
        if g not in seen:
            seen.add(g)
            ordered_runs.append(g)

    for run in ordered_runs:
        idx = np.where(group_ids == run)[0]
        if idx.size < window_size:
            continue

        # 안전하게 오름차순 정렬(대부분 이미 정렬되어 있겠지만 방어)
        idx = np.sort(idx)

        for start_pos in range(0, idx.size - window_size + 1, window_stride):
            inds = idx[start_pos : start_pos + window_size]  # (T,)
            sequences.append(arr[inds])      # (T, F)
            window_indices.append(inds)      # (T,)

    if len(sequences) == 0:
        raise ValueError("No sequences created: check window_size/window_stride and group sizes.")

    return np.stack(sequences, axis=0), np.stack(window_indices, axis=0)


def _aggregate_window_scores_to_point_scores_by_indices(
    num_points: int,
    window_scores: np.ndarray,          # (N_windows,)
    window_point_indices: np.ndarray,   # (N_windows, T)
) -> np.ndarray:
    """
    각 윈도우 점수를, 윈도우가 덮는 '원본 row 인덱스'들에 누적/평균해서 point score로 만든다.
    """
    point_scores = np.zeros(num_points, dtype=np.float64)
    counts = np.zeros(num_points, dtype=np.int64)

    for score, inds in zip(window_scores, window_point_indices):
        point_scores[inds] += float(score)
        counts[inds] += 1

    counts[counts == 0] = 1
    point_scores /= counts
    return point_scores.astype(np.float32)


# LSTM AutoEncoder 모델 정의
class LSTMAutoEncoder(nn.Module):
    """
    간단한 sequence-wise LSTM AutoEncoder.

    인코더:
        x (B, T, F) -> LSTM -> 마지막 hidden state h_T (B, H) -> Linear -> z (B, latent_dim)

    디코더:
        z (B, latent_dim) -> Linear -> h_dec (B, H)
        h_dec를 T 타임스텝 길이로 반복해서 LSTM 입력으로 사용
        -> (B, T, F) 재구성
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # num_layers가 1이면 PyTorch LSTM은 dropout을 무시하므로 0으로 처리
        enc_dropout = dropout if num_layers > 1 else 0.0
        dec_dropout = dropout if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=enc_dropout,
        )
        self.enc_fc = nn.Linear(hidden_dim, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=input_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dec_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        return: (B, T, F)  (재구성된 시퀀스)
        """
        batch_size, seq_len, _ = x.shape

        # Encoder: 마지막 layer의 hidden state 사용
        _, (h_n, _) = self.encoder(x)  # h_n: (num_layers, B, H)
        last_hidden = h_n[-1]          # (B, H)

        z = self.enc_fc(last_hidden)   # (B, latent_dim)

        # Decoder 입력 준비
        dec_hidden = self.dec_fc(z)    # (B, H)
        # (B, 1, H) -> (B, T, H)로 반복
        dec_input = dec_hidden.unsqueeze(1).repeat(1, seq_len, 1)

        out, _ = self.decoder(dec_input)  # (B, T, F)
        return out


# Device 헬퍼
def _get_device(cfg: LSTMAEConfig) -> torch.device:
    if cfg.device is not None:
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 학습 함수

def train_lstm_autoencoder(
    train_X: pl.DataFrame,
    cfg: Optional[LSTMAEConfig] = None,
    group_ids: Optional[np.ndarray] = None,
) -> LSTMAEArtifacts:
    """
    Polars DataFrame(train_X)을 받아 LSTM AutoEncoder를 학습하고,
    point-wise reconstruction error 기반 threshold까지 계산해서 반환합니다.

    train_X는 이미 NUMERIC_COLS만 선택된 상태라고 가정합니다.
    (현재 프로젝트의 process_data() 결과를 그대로 넣으면 됨)
    """
    if cfg is None:
        cfg = LSTMAEConfig()

    # 데이터 준비 (numpy)
    # (N, F)
    arr = train_X.to_numpy().astype(np.float32)
    num_points, num_features = arr.shape

    if cfg.input_dim is None:
        cfg.input_dim = num_features

    # 스케일링
    scaler = StandardScaler.fit(arr)
    arr_norm = scaler.transform(arr)

    # 시퀀스 생성 (run별)
    if group_ids is not None:
        sequences, window_point_indices = _create_sequences_by_group(
            arr=arr_norm,
            group_ids=group_ids,
            window_size=cfg.window_size,
            window_stride=cfg.window_stride,
        )
    else:
        sequences, start_indices = _create_sequences(
            arr=arr_norm,
            window_size=cfg.window_size,
            window_stride=cfg.window_stride,
        )
        # start_indices -> (N_windows, T)로 변환
        T = cfg.window_size
        window_point_indices = start_indices[:, None] + np.arange(T)[None, :]

    dataset = SequenceDataset(sequences)
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # 모델/옵티마 설정
    device = _get_device(cfg)
    torch.manual_seed(RANDOM_SEED)

    model = LSTMAutoEncoder(
        input_dim=cfg.input_dim, # type: ignore
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.MSELoss(reduction="none")  # (B, T, F) shape 유지

    # 학습 루프
    model.train()
    for epoch in range(cfg.num_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)  # (B, T, F)

            optimizer.zero_grad()
            recon = model(batch)      # (B, T, F)

            # element-wise MSE -> sample-wise scalar loss
            loss_element = criterion(recon, batch)          # (B, T, F)
            loss = loss_element.mean()                      # 전체 평균

            loss.backward()
            if cfg.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if cfg.print_progress:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"[LSTM-AE][Train] epoch {epoch+1}/{cfg.num_epochs} "
                  f"loss={avg_loss:.6f}")

    # 학습 데이터에 대한 reconstruction error 측정
    model.eval()
    all_window_scores: list[np.ndarray] = []

    eval_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            recon = model(batch)

            # (B, T, F)
            loss_elem = criterion(recon, batch)
            # 윈도우 단위 scalar score: T, F 평균
            scores = loss_elem.mean(dim=(1, 2))  # (B,)
            all_window_scores.append(scores.cpu().numpy())

    window_scores = np.concatenate(all_window_scores, axis=0)  # (N_windows,)

    # 윈도우 score -> point-wise score 집계
    point_scores = _aggregate_window_scores_to_point_scores_by_indices(
        num_points=num_points,
        window_scores=window_scores,
        window_point_indices=window_point_indices,
    ) # (N,)

    # threshold 계산 (train 기준)
    threshold = float(np.quantile(point_scores, cfg.threshold_quantile))
    if cfg.print_progress:
        print(
            f"[LSTM-AE] threshold (quantile={cfg.threshold_quantile}) "
            f"= {threshold:.6f}"
        )

    artifacts = LSTMAEArtifacts(
        model=model,
        scaler=scaler,
        window_size=cfg.window_size,
        window_stride=cfg.window_stride,
        threshold=threshold,
    )
    return artifacts


# 추론 함수

def inference_lstm_autoencoder(
    test_X: pl.DataFrame,
    artifacts: LSTMAEArtifacts,
    cfg: Optional[LSTMAEConfig] = None,
    group_ids: Optional[np.ndarray] = None,
) -> pl.DataFrame:
    """
    학습된 LSTM-AE artifacts + test_X(Polars DataFrame)를 받아
    `faultNumber` 컬럼 하나를 가진 Polars DataFrame을 반환.

    - train 때와 동일하게 test_X는 (N, F) numeric DataFrame이라고 가정.
    - point-wise reconstruction error가 threshold보다 크면 1(이상치),
      아니면 0(정상)으로 라벨링.
    """
    if cfg is None:
        # cfg가 없어도 batch_size만 있으면 되므로 기본값으로 생성
        cfg = LSTMAEConfig()

    model = artifacts.model
    scaler = artifacts.scaler
    window_size = artifacts.window_size
    window_stride = artifacts.window_stride
    threshold = artifacts.threshold

    device = next(model.parameters()).device
    criterion = nn.MSELoss(reduction="none")

    # test 데이터 준비
    arr = test_X.to_numpy().astype(np.float32)
    num_points, _ = arr.shape

    arr_norm = scaler.transform(arr)

    if group_ids is not None:
        sequences, window_point_indices = _create_sequences_by_group(
            arr=arr_norm,
            group_ids=group_ids,
            window_size=window_size,
            window_stride=window_stride,
        )
    else:
        sequences, start_indices = _create_sequences(
            arr=arr_norm,
            window_size=window_size,
            window_stride=window_stride,
        )
        T = window_size
        window_point_indices = start_indices[:, None] + np.arange(T)[None, :]
    dataset = SequenceDataset(sequences)

    # 윈도우별 reconstruction error 계산
    model.eval()
    all_window_scores: list[np.ndarray] = []

    infer_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    with torch.no_grad():
        for batch in infer_loader:
            batch = batch.to(device)
            recon = model(batch)

            loss_elem = criterion(recon, batch)        # (B, T, F)
            scores = loss_elem.mean(dim=(1, 2))        # (B,)
            all_window_scores.append(scores.cpu().numpy())

    window_scores = np.concatenate(all_window_scores, axis=0)

    # point-wise score로 집계
    point_scores = _aggregate_window_scores_to_point_scores_by_indices(
        num_points=num_points,
        window_scores=window_scores,
        window_point_indices=window_point_indices,
    )

    # threshold 기반 이상치 판별
    labels = (point_scores > threshold).astype(int)

    pred_df = pl.DataFrame({"faultNumber": labels})
    return pred_df
