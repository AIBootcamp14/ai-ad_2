from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
from enum import Enum
import csv

import numpy as np
import polars as pl
import wandb
from omegaconf import OmegaConf
import hydra
from hydra.utils import to_absolute_path
from hydra.core.config_store import ConfigStore

from config import (
    IsolationForestConfig,
    SGDOneClassSVMConfig,
    LOFConfig,
    EllipticEnvelopeConfig,
    LSTMAEConfig,
)
from data_utils import (
    set_global_seed,
    load_train_data,
    load_test_data,
    process_data,
)
from model_utils import train_model, inference
from deep_model_utils import (
    train_lstm_autoencoder,
    inference_lstm_autoencoder,
)
from eda import run_eda  # run_eda를 eda.py에 옮겨 두었으면 거기서 import

log = logging.getLogger(__name__)

class ModelType(str, Enum):
    IFOREST = "iforest"
    SGD = "sgd"
    LOF = "lof"
    EE = "ee"
    LSTM_AE = "lstm_ae"
    ENSEMBLE = "ensemble"


@dataclass
class ModelConfig:
    # Hydra에서 각 모델에 해당하는 서브 config 를 넣기 위한 래퍼
    iforest: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    sgd: SGDOneClassSVMConfig = field(default_factory=SGDOneClassSVMConfig)
    lof: LOFConfig = field(default_factory=LOFConfig)
    ee: EllipticEnvelopeConfig = field(default_factory=EllipticEnvelopeConfig)
    lstm_ae: LSTMAEConfig = field(default_factory=LSTMAEConfig)


@dataclass
class AppConfig:
    seed: int = 42
    output_path: str = "output_hydra.csv"
    model_type: ModelType = ModelType.IFOREST
    model: ModelConfig = field(default_factory=ModelConfig)
    eda_only: bool = False

    ensemble_models: list[ModelType] = field(
        default_factory=lambda: [
            ModelType.IFOREST,
            ModelType.SGD,
            ModelType.LOF,
            ModelType.EE,
        ]
    )
    ensemble_vote_threshold: float = 0.5

    use_wandb: bool = True
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: list[str] = field(default_factory=list)

cs = ConfigStore.instance()
cs.store(name="app_schema", node=AppConfig)


def save_predictions_as_submission(pred_df: Any, output_path: Path | str) -> None:
    """
    submission 요구사항에 맞게 `,faultNumber` 헤더 형식으로 CSV를 저장합니다.

    - 첫 번째 컬럼: 0..N-1 인덱스 (헤더는 빈 문자열 "")
    - 두 번째 컬럼: faultNumber

    pandas / polars DataFrame 모두 지원합니다.
    """
    output_path = Path(output_path)

    try:
        col = pred_df["faultNumber"]
    except Exception as e:  # 방어적 코드
        raise KeyError("pred_df에서 'faultNumber' 컬럼을 찾을 수 없습니다.") from e

    # pandas / polars 모두 대응
    if hasattr(col, "to_list"):      # polars Series
        values = col.to_list()
    elif hasattr(col, "tolist"):     # pandas Series
        values = col.tolist()
    else:
        values = list(col)

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        # 헤더: 빈 문자열, 'faultNumber'
        writer.writerow(["", "faultNumber"])
        for idx, v in enumerate(values):
            writer.writerow([idx, int(v)])

    print(f"[Hydra] Saved predictions to {output_path.resolve()}")


def _get_internal_model_and_kwargs(
    mt: ModelType,
    cfg: AppConfig,
) -> tuple[str, dict[str, Any]]:
    """
    단일 모델 학습을 위해
    - 내부 model_type 문자열 (IsolationForest 등)
    - train_model 에 넘길 config kwargs
    를 반환하는 헬퍼 함수.
    """
    if mt == ModelType.IFOREST:
        return "IsolationForest", {"if_config": cfg.model.iforest}
    elif mt == ModelType.SGD:
        return "SGDOneClassSVM", {"sgd_config": cfg.model.sgd}
    elif mt == ModelType.LOF:
        return "LocalOutlierFactor", {"lof_config": cfg.model.lof}
    elif mt == ModelType.EE:
        return "EllipticEnvelope", {"ee_config": cfg.model.ee}
    else:
        raise ValueError(f"Unsupported single ModelType for internal mapping: {mt}")


@hydra.main(version_base=None, config_name="config", config_path="conf")
def hydra_app(cfg: AppConfig) -> None:
    """
    Hydra 기반 엔트리포인트.
    conf/ 밑의 YAML + dataclass 조합으로 하이퍼파라미터를 관리.
    """
    total_start = time.perf_counter()
    # 작업 디렉토리는 Hydra가 자동으로 runs/2025-... 이런 식으로 바꿔줌
    print("[Hydra] Working directory:", Path.cwd())

    set_global_seed(cfg.seed)
    log.info(f"Global seed set to {cfg.seed}")

    if cfg.eda_only:
        print("[Hydra] EDA only mode")
        run_eda(train_output=True, test_output=True)
        return

    # 1. 데이터 로드 & 전처리
    train_df = load_train_data()
    test_df = load_test_data()
    train_X = process_data(train_df)
    test_X = process_data(test_df)

    # 2. W&B 설정
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    run = None
    if cfg.use_wandb and cfg.wandb_project is not None:
        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            config=cfg_dict,  # type: ignore
            tags=cfg.wandb_tags,
        )
        log.info(f"[W&B] Initialized run: {wandb.run.name}")  # type: ignore
    else:
        log.info("[W&B] Disabled (use_wandb=False or wandb_project is None)")

    # 3. 모델 학습 & 추론
    # 3-1. 앙상블 모드
    if cfg.model_type == ModelType.ENSEMBLE:
        log.info(f"[ENSEMBLE] Using models: {cfg.ensemble_models}")
        train_start = time.perf_counter()

        # 각 서브 모델별 예측값 리스트 (numpy 배열)
        preds_list: list[np.ndarray] = []

        for sub_mt in cfg.ensemble_models:
            # ENSEMBLE 같은 잘못된 값이 들어오지 않도록 방어
            if sub_mt == ModelType.ENSEMBLE:
                raise ValueError("ensemble_models 안에 ENSEMBLE 을 넣을 수 없습니다.")

            internal_model_type, kwargs = _get_internal_model_and_kwargs(sub_mt, cfg)
            log.info(f"[ENSEMBLE] Training sub model: {sub_mt} ({internal_model_type})")

            # 단일 모델 학습
            sub_model = train_model(
                train_X=train_X,
                model_type=internal_model_type,  # "IsolationForest" 등 # type: ignore
                **kwargs,
            )

            # 단일 모델 추론
            sub_pred_df = inference(test_X=test_X, model=sub_model)
            log.info(
                f"[ENSEMBLE] Sub model {sub_mt} prediction head:\n"
                f"{sub_pred_df.head()}"
            )

            # Polars → numpy (0/1 레이블)
            sub_pred = sub_pred_df["faultNumber"].to_numpy()
            preds_list.append(sub_pred)

        train_elapsed = time.perf_counter() - train_start
        log.info(f"[ENSEMBLE][TRAIN] elapsed: {train_elapsed:.3f} seconds")
        if wandb.run is not None:
            wandb.log({"time/train_elapsed_sec": train_elapsed})
            wandb.log({"ensemble/num_models": len(preds_list)})

        # 각 모델의 예측을 (n_models, n_samples) 로 쌓아서 평균 → threshold
        infer_start = time.perf_counter()
        preds_array = np.stack(preds_list, axis=0)  # (M, N)
        mean_scores = preds_array.mean(axis=0)      # (N,)

        threshold = cfg.ensemble_vote_threshold
        final_labels = (mean_scores >= threshold).astype(int)

        pred_df = pl.DataFrame({"faultNumber": final_labels})
        print("[Hydra][ENSEMBLE] Prediction head:")
        print(pred_df.head())

        infer_elapsed = time.perf_counter() - infer_start
        log.info(f"[ENSEMBLE][INFER] elapsed: {infer_elapsed:.3f} seconds")
        if wandb.run is not None:
            wandb.log({"time/infer_elapsed_sec": infer_elapsed})
            wandb.log({"ensemble/vote_threshold": threshold})

    # 3-2. 단일 모델 모드 (기존 로직)
    else:
        mt = cfg.model_type

        # LSTM-AE
        if mt == ModelType.LSTM_AE:
            log.info("[SINGLE][LSTM_AE] Training LSTM AutoEncoder model")

            lstm_cfg = cfg.model.lstm_ae

            # 학습
            train_start = time.perf_counter()
            artifacts = train_lstm_autoencoder(
                train_X=train_X,
                cfg=lstm_cfg,
            )
            train_elapsed = time.perf_counter() - train_start
            log.info(f"[LSTM_AE][TRAIN] elapsed: {train_elapsed:.3f} seconds")
            if wandb.run is not None:
                wandb.log({"time/train_elapsed_sec": train_elapsed})

            # 추론
            infer_start = time.perf_counter()
            pred_df = inference_lstm_autoencoder(
                test_X=test_X,
                artifacts=artifacts,
                cfg=lstm_cfg,
            )
            infer_elapsed = time.perf_counter() - infer_start
            log.info(f"[LSTM_AE][INFER] elapsed: {infer_elapsed:.3f} seconds")
            if wandb.run is not None:
                wandb.log({"time/infer_elapsed_sec": infer_elapsed})

            print("[Hydra][LSTM_AE] Prediction head:")
            print(pred_df.head())

        # sklearn 단일 모델 로직
        else:
            internal_model_type, kwargs = _get_internal_model_and_kwargs(mt, cfg)
            log.info(
                f"[SINGLE] Training model_type={mt} "
                f"(internal={internal_model_type})"
            )

            # 학습
            train_start = time.perf_counter()
            model = train_model(
                train_X=train_X,
                model_type=internal_model_type,  # type: ignore
                **kwargs,
            )
            train_elapsed = time.perf_counter() - train_start
            log.info(f"[TRAIN] elapsed: {train_elapsed:.3f} seconds")
            if wandb.run is not None:
                wandb.log({"time/train_elapsed_sec": train_elapsed})

            # 추론
            infer_start = time.perf_counter()
            pred_df = inference(test_X=test_X, model=model)
            print("[Hydra] Prediction head:")
            print(pred_df.head())
            infer_elapsed = time.perf_counter() - infer_start
            log.info(f"[INFER] elapsed: {infer_elapsed:.3f} seconds")
            if wandb.run is not None:
                wandb.log({"time/infer_elapsed_sec": infer_elapsed})

    # 공통 후처리 (제출 파일 저장 & W&B artifact 업로드)
    total_elapsed = time.perf_counter() - total_start
    log.info(f"[TOTAL] Full run elapsed: {total_elapsed:.3f} seconds")
    if wandb.run is not None:
        wandb.log({"time/total_elapsed_sec": total_elapsed})

    # 4. 저장 (`,faultNumber` 형식)
    out_path = Path(to_absolute_path(cfg.output_path))
    save_predictions_as_submission(pred_df, out_path)

    # 5. 제출 파일을 W&B Artifact로 업로드
    if wandb.run is not None:
        artifact = wandb.Artifact(
            name="submission_csv",
            type="prediction",
            metadata={
                "model_type": str(cfg.model_type),
                "output_path": str(out_path),
                "ensemble_models": [str(m) for m in cfg.ensemble_models],
                "ensemble_vote_threshold": cfg.ensemble_vote_threshold,
            },
        )
        artifact.add_file(str(out_path))
        wandb.log_artifact(artifact)

        # run 종료
        wandb.finish()


if __name__ == "__main__":
    hydra_app()
