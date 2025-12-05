from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import polars as pl

from config import DATA_DIR, NUMERIC_COLS, RANDOM_SEED


def set_global_seed(seed: int = RANDOM_SEED, deterministic_torch: bool = True) -> None:
    """
    전체 파이썬/ML 스택의 랜덤 시드를 최대한 통일되게 설정합니다.

    - PYTHONHASHSEED
    - random (표준 라이브러리)
    - numpy
    - torch + CUDA + cuDNN


    Parameters
    ----------
    seed : int
        사용할 랜덤 시드 값.
    deterministic_torch : bool
        True 이면 torch/cudnn 관련 설정을 deterministic 모드로 맞춥니다.
        (속도는 조금 느려질 수 있지만, 재현성은 좋아집니다.)
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 일부 연산에서 필요할 수 있는 옵션 (선택사항)
    # os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")


def load_train_data(data_dir: Optional[Path] = None) -> pl.DataFrame:
    """
    train.csv 를 로드해서 Polars DataFrame 으로 반환합니다.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    train_path = data_dir / "train.csv"
    df = pl.read_csv(train_path)
    return df


def load_test_data(data_dir: Optional[Path] = None) -> pl.DataFrame:
    """
    test.csv 를 로드해서 Polars DataFrame 으로 반환합니다.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    test_path = data_dir / "test.csv"
    df = pl.read_csv(test_path)
    return df


def process_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    학습/추론에 사용할 수치 컬럼만 추출합니다.
    """
    # select 는 새로운 DataFrame 을 반환하므로 별도 copy 는 필요 없음
    return df.select(NUMERIC_COLS)
