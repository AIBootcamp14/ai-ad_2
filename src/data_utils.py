from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from config import DATA_DIR, NUMERIC_COLS, RANDOM_SEED


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """numpy 등 전역 랜덤 시드를 설정합니다."""
    import random

    np.random.seed(seed)
    random.seed(seed)


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
