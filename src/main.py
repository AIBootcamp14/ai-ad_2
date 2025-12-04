from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Literal
from enum import Enum
import csv

import wandb
from omegaconf import OmegaConf
import hydra
from hydra.utils import to_absolute_path
from hydra.core.config_store import ConfigStore

from config import (
    IsolationForestConfig,
    SGDOneClassSVMConfig,
    LOFConfig,
    EllipticEnvelopeCfg,
)
from data_utils import (
    set_global_seed,
    load_train_data,
    load_test_data,
    process_data,
)
from model_utils import train_model, inference
from eda import run_eda  # run_eda를 eda.py에 옮겨 두었으면 거기서 import

log = logging.getLogger(__name__)

class ModelType(str, Enum):
    IFOREST = "iforest"
    SGD = "sgd"
    LOF = "lof"
    EE = "ee"


@dataclass
class ModelConfig:
    # Hydra에서 각 모델에 해당하는 서브 config 를 넣기 위한 래퍼
    iforest: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    sgd: SGDOneClassSVMConfig = field(default_factory=SGDOneClassSVMConfig)
    lof: LOFConfig = field(default_factory=LOFConfig)
    ee: EllipticEnvelopeCfg = field(default_factory=EllipticEnvelopeCfg)


@dataclass
class AppConfig:
    seed: int = 42
    output_path: str = "output_hydra.csv"
    model_type: ModelType = ModelType.IFOREST
    model: ModelConfig = field(default_factory=ModelConfig)
    eda_only: bool = False

    wandb_project: Literal[str] = None
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

    if cfg.eda_only:
        print("[Hydra] EDA only mode")
        run_eda(train_output=True, test_output=True)
        return

    # 1. 데이터 로드 & 전처리
    train_df = load_train_data()
    test_df = load_test_data()
    train_X = process_data(train_df)
    test_X = process_data(test_df)

    # 2. model_type 해석 (Enum 또는 문자열 모두 대응)
    mt = cfg.model_type

    # Enum이면 .name 사용, 문자열이면 바로 사용
    if isinstance(mt, Enum):
        mt_name = mt.name  # IFOREST / SGD / LOF / EE
    else:
        mt_name = str(mt).upper()

    model_type_map = {
        "IFOREST": "IsolationForest",
        "SGD": "SGDOneClassSVM",
        "LOF": "LocalOutlierFactor",
        "EE": "EllipticEnvelope",
    }

    if mt_name not in model_type_map:
        raise ValueError(f"Unknown model_type: {mt} (normalized: {mt_name})")

    internal_model_type = model_type_map[mt_name]

    # 3. Weights & Biases 초기화
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_run_name,
        config=cfg_dict, # type: ignore
        tags=cfg.wandb_tags,
    )

    # 4. 모델 학습
    train_start = time.perf_counter()
    model = train_model(
        train_X=train_X,
        model_type=internal_model_type,  # "SGDOneClassSVM" 등 # type: ignore
        if_config=cfg.model.iforest if cfg.model_type == ModelType.IFOREST else None,
        sgd_config=cfg.model.sgd if cfg.model_type == ModelType.SGD else None,
        lof_config=cfg.model.lof if cfg.model_type == ModelType.LOF else None,
        ee_config=cfg.model.ee if cfg.model_type == ModelType.EE else None,
    )
    train_elapsed = time.perf_counter() - train_start
    log.info(f"[TRAIN] elapsed: {train_elapsed:.3f} seconds")
    wandb.log({"time/train_elapsed_sec": train_elapsed})

    # 5. 추론
    infer_start = time.perf_counter()
    pred_df = inference(test_X=test_X, model=model)
    print("[Hydra] Prediction head:")
    print(pred_df.head())

    infer_elapsed = time.perf_counter() - infer_start
    log.info(f"[INFER] elapsed: {infer_elapsed:.3f} seconds")
    wandb.log({"time/infer_elapsed_sec": infer_elapsed})

    total_elapsed = time.perf_counter() - total_start
    log.info(f"[TOTAL] Full run elapsed: {total_elapsed:.3f} seconds")
    wandb.log({"time/total_elapsed_sec": total_elapsed})

    # 6. 저장 (`,faultNumber` 형식)
    out_path = Path(to_absolute_path(cfg.output_path))
    save_predictions_as_submission(pred_df, out_path)

    # 7. 제출 파일을 W&B Artifact로 업로드
    if wandb.run is not None:
        artifact = wandb.Artifact(
            name="submission_csv",
            type="prediction",
            metadata={
                "model_type": mt_name,
                "internal_model_type": internal_model_type,
                "output_path": str(out_path),
            },
        )
        artifact.add_file(str(out_path))
        wandb.log_artifact(artifact)

        # run 종료
        wandb.finish()


if __name__ == "__main__":
    hydra_app()
