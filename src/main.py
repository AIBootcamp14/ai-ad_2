from __future__ import annotations

import time
import logging
import sys
import platform
from contextlib import contextmanager
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
from sklearn.preprocessing import StandardScaler, RobustScaler

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


@contextmanager
def log_time(logger: logging.Logger, name: str, level: int = logging.INFO):
    """
    표준화된 타이밍 로깅용 컨텍스트 매니저.
    예) with log_time(log, "load_data"): ...
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        logger.log(level, "%s took %.3fs", name, elapsed)


class CtxAdapter(logging.LoggerAdapter):
    """
    로그 메시지에 컨텍스트를 prefix로 붙여주는 어댑터.
    formatter에 extra 필드가 없어도 컨텍스트가 보이도록 설계.
    """
    def process(self, msg, kwargs):
        if self.extra:
            prefix = " ".join(f"{k}={v}" for k, v in self.extra.items())
            return f"[{prefix}] {msg}", kwargs
        return msg, kwargs


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
class PreprocessConfig:
    enabled: bool = True
    # "none" | "standard" | "robust"
    scaler: str = "standard"

    # StandardScaler: with_mean/with_std
    # RobustScaler: with_centering/with_scaling
    with_centering: bool = True
    with_scaling: bool = True

    # RobustScaler 전용(하지만 통일해서 둬도 무방)
    quantile_range: tuple[float, float] = (25.0, 75.0)

    # 선택: 이상치로 인한 수치 폭발 완화
    clip: float | None = None


def _fit_preprocess(train_X: pl.DataFrame, pp: PreprocessConfig):
    cols = train_X.columns
    X = train_X.to_numpy()

    if (not pp.enabled) or pp.scaler == "none":
        return train_X, None

    if pp.scaler == "standard":
        scaler = StandardScaler(
            with_mean=pp.with_centering,
            with_std=pp.with_scaling,
        )
    elif pp.scaler == "robust":
        scaler = RobustScaler(
            with_centering=pp.with_centering,
            with_scaling=pp.with_scaling,
            quantile_range=tuple(pp.quantile_range), # type: ignore
        )
    else:
        raise ValueError(f"Unsupported preprocess.scaler: {pp.scaler}")

    Xs = scaler.fit_transform(X)
    if pp.clip is not None:
        Xs = np.clip(Xs, -pp.clip, pp.clip)

    return pl.DataFrame(Xs, schema=cols), scaler


def _apply_preprocess(X_df: pl.DataFrame, pp: PreprocessConfig, scaler):
    if scaler is None:
        return X_df

    cols = X_df.columns
    X = X_df.to_numpy()
    Xs = scaler.transform(X)
    if pp.clip is not None:
        Xs = np.clip(Xs, -pp.clip, pp.clip)

    return pl.DataFrame(Xs, schema=cols)



@dataclass
class AppConfig:
    seed: int = 42
    output_path: str = "output_hydra.csv"
    model_type: ModelType = ModelType.IFOREST
    model: ModelConfig = field(default_factory=ModelConfig)
    eda_only: bool = False

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)

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

    log.info("[Hydra] Saved predictions to %s", output_path.resolve())


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


def wandb_log_pred_stats(
    labels: np.ndarray,
    group_ids: Optional[np.ndarray] = None,
    prefix: str = "pred",
    topk: int = 20,
) -> None:
    """W&B에 예측 통계(전체 이상치 비율 + run별 이상치 비율 분포)를 로깅."""
    if wandb.run is None:
        return

    labels = np.asarray(labels).astype(np.int64)
    n_total = int(labels.size)
    n_anom = int(labels.sum())
    anom_ratio = float(n_anom / max(n_total, 1))

    wandb.log({
        f"{prefix}/num_total": n_total,
        f"{prefix}/num_anomaly": n_anom,
        f"{prefix}/anomaly_ratio": anom_ratio,
    })

    if group_ids is None:
        return

    group_ids = np.asarray(group_ids)
    if group_ids.shape[0] != labels.shape[0]:
        # 길이 불일치 방어
        wandb.log({f"{prefix}/warn_group_mismatch": 1})
        return

    # run별 ratio 계산 (np.unique + bincount로 빠르게)
    uniq, inv = np.unique(group_ids, return_inverse=True)
    counts = np.bincount(inv)
    sums = np.bincount(inv, weights=labels)
    ratios = sums / np.maximum(counts, 1)

    # 분포 요약 통계
    p = np.percentile(ratios, [0, 25, 50, 75, 90, 95, 99, 100])
    wandb.log({
        f"{prefix}/runs": int(uniq.size),
        f"{prefix}/run_ratio_mean": float(ratios.mean()),
        f"{prefix}/run_ratio_std": float(ratios.std()),
        f"{prefix}/run_ratio_p0": float(p[0]),
        f"{prefix}/run_ratio_p25": float(p[1]),
        f"{prefix}/run_ratio_p50": float(p[2]),
        f"{prefix}/run_ratio_p75": float(p[3]),
        f"{prefix}/run_ratio_p90": float(p[4]),
        f"{prefix}/run_ratio_p95": float(p[5]),
        f"{prefix}/run_ratio_p99": float(p[6]),
        f"{prefix}/run_ratio_p100": float(p[7]),
        f"{prefix}/runs_with_any_anomaly": int((sums > 0).sum()),
        f"{prefix}/runs_with_any_anomaly_ratio": float((sums > 0).mean()),
    })

    # 히스토그램(= run별 이상치 비율 분포)
    wandb.log({f"{prefix}/run_ratio_hist": wandb.Histogram(ratios)}) # type: ignore

    # 상위 topk run 테이블(원하면 UI에서 bar chart로 바로 뽑기 좋음)
    top_idx = np.argsort(-ratios)[:topk]
    table = wandb.Table(columns=["simulationRun", "anomaly_ratio", "n_points", "n_anomaly"])
    for i in top_idx:
        table.add_data(int(uniq[i]), float(ratios[i]), int(counts[i]), int(sums[i]))
    wandb.log({f"{prefix}/run_ratio_top": table})


@hydra.main(version_base=None, config_name="config", config_path="conf")
def hydra_app(cfg: AppConfig) -> None:
    """
    Hydra 기반 엔트리포인트.
    conf/ 밑의 YAML + dataclass 조합으로 하이퍼파라미터를 관리.
    """
    total_start = time.perf_counter()

    # W&B run 핸들 (예외가 나도 finish 되도록 finally에서 정리)
    run = None

    try:
        # 작업 디렉토리는 Hydra가 자동으로 outputs/.. 로 바뀔 수 있음
        log.info("[Hydra] Working directory: %s", Path.cwd())

        # 0) Seed
        set_global_seed(cfg.seed)
        log.info("Global seed set to %s", cfg.seed)

        # 0-1) Config / Environment dump (재현성 핵심)
        try:
            log.info("===== CONFIG (resolved) =====\n%s", OmegaConf.to_yaml(cfg, resolve=True))
        except Exception:
            # to_yaml이 실패하는 케이스 대비(드물지만 안전)
            cfg_dict_fallback = OmegaConf.to_container(cfg, resolve=True)
            log.info("===== CONFIG (resolved, fallback dict) =====\n%s", cfg_dict_fallback)

        log.info("Python: %s", sys.version.replace("\n", " "))
        log.info("Platform: %s", platform.platform())
        log.info("Versions: numpy=%s polars=%s wandb=%s", np.__version__, pl.__version__, wandb.__version__)
        log.info("Output path(cfg.output_path): %s", cfg.output_path)

        # EDA only
        if cfg.eda_only:
            log.info("[Hydra] EDA only mode enabled (no training/inference)")
            with log_time(log, "eda/run_eda"):
                run_eda(train_output=True, test_output=True)
            return

        # 1. 데이터 로드 & 전처리
        with log_time(log, "data/load_train_data"):
            train_df = load_train_data()
            train_group = train_df["simulationRun"].to_numpy()
        with log_time(log, "data/load_test_data"):
            test_df = load_test_data()
            test_group = test_df["simulationRun"].to_numpy()
        log.info("train_df shape=%s, test_df shape=%s", train_df.shape, test_df.shape)

        with log_time(log, "data/process_data(train)"):
            train_X = process_data(train_df)
        with log_time(log, "data/process_data(test)"):
            test_X = process_data(test_df)
        log.info("train_X shape=%s, test_X shape=%s", train_X.shape, test_X.shape)

        with log_time(log, "preprocess/fit_scaler"):
            train_X_pp, fitted_scaler = _fit_preprocess(train_X, cfg.preprocess)
        with log_time(log, "preprocess/apply_scaler"):
            test_X_pp = _apply_preprocess(test_X, cfg.preprocess, fitted_scaler)

        # 모든 모델(LSTM_AE 포함)에서 동일한 전처리 결과를 사용
        sk_train_X, sk_test_X = train_X_pp, test_X_pp

        # 2. W&B 설정
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        if cfg.use_wandb and cfg.wandb_project is not None:
            run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_run_name,
                config=cfg_dict,  # type: ignore
                tags=cfg.wandb_tags,
            )
            log.info("[W&B] Initialized run: %s", wandb.run.name)  # type: ignore
        else:
            log.info("[W&B] Disabled (use_wandb=False or wandb_project is None)")

        # 3. 모델 학습 & 추론
        # 3-1. 앙상블 모드
        if cfg.model_type == ModelType.ENSEMBLE:
            log.info("[ENSEMBLE] Using models: %s", cfg.ensemble_models)
            train_start = time.perf_counter()

            # 각 서브 모델별 예측값 리스트 (numpy 배열)
            preds_list: list[np.ndarray] = []

            for sub_mt in cfg.ensemble_models:
                # ENSEMBLE 같은 잘못된 값이 들어오지 않도록 방어
                if sub_mt == ModelType.ENSEMBLE:
                    raise ValueError("ENSEMBLE cannot include itself in ensemble_models.")

                sublog = CtxAdapter(log, {"mode": "ENSEMBLE", "sub_model": str(sub_mt), "seed": cfg.seed})

                internal_model_type, kwargs = _get_internal_model_and_kwargs(sub_mt, cfg)
                sublog.info("Training sub model => %s", internal_model_type)

                with log_time(sublog.logger, f"ensemble/train_sub/{sub_mt}"):
                    # 단일 모델 학습 (sklearn 계열)
                    sub_model = train_model(train_X=sk_train_X, model_type=internal_model_type, **kwargs) # type: ignore

                with log_time(sublog.logger, f"ensemble/infer_sub/{sub_mt}"):
                    # 단일 모델 추론
                    sub_pred_df = inference(test_X=sk_test_X, model=sub_model)

                sublog.info("Prediction head:\n%s", sub_pred_df.head())

                # Polars → numpy (0/1 레이블)
                sub_pred = sub_pred_df["faultNumber"].to_numpy()
                preds_list.append(sub_pred)

            train_elapsed = time.perf_counter() - train_start
            log.info("[ENSEMBLE][TRAIN] elapsed: %.3f seconds", train_elapsed)
            if wandb.run is not None:
                wandb.log({"time/train_elapsed_sec": train_elapsed})
                wandb.log({"ensemble/num_models": len(preds_list)})

            # 각 모델의 예측을 (n_models, n_samples) 로 쌓아서 평균 → threshold
            infer_start = time.perf_counter()
            with log_time(log, "ensemble/aggregate_vote"):
                preds_array = np.stack(preds_list, axis=0)  # (M, N)
                mean_scores = preds_array.mean(axis=0)      # (N,)

                threshold = cfg.ensemble_vote_threshold
                final_labels = (mean_scores >= threshold).astype(int)

                pred_df = pl.DataFrame({"faultNumber": final_labels})
                log.info("[ENSEMBLE] Prediction head:\n%s", pred_df.head())
                if wandb.run is not None:
                    wandb_log_pred_stats(final_labels, test_group, prefix="pred/ENSEMBLE")

            infer_elapsed = time.perf_counter() - infer_start
            log.info("[ENSEMBLE][INFER] elapsed: %.3f seconds", infer_elapsed)
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

                train_group = train_df["simulationRun"].to_numpy()
                test_group  = test_df["simulationRun"].to_numpy()
                # 학습
                train_start = time.perf_counter()
                with log_time(log, "lstm_ae/train_lstm_autoencoder"):
                    artifacts = train_lstm_autoencoder(
                        train_X=sk_train_X,
                        inputs_are_preprocessed=True,
                        preprocess_scaler=fitted_scaler,
                        preprocess_clip=cfg.preprocess.clip,
                        cfg=lstm_cfg,
                        group_ids=train_group,
                    )
                train_elapsed = time.perf_counter() - train_start
                log.info("[LSTM_AE][TRAIN] elapsed: %.3f seconds", train_elapsed)
                pt = getattr(artifacts, "threshold", None)
                rt = getattr(artifacts, "run_threshold", None)
                rsq = getattr(artifacts, "run_score_quantile", None)
                run_q = getattr(lstm_cfg, "run_threshold_quantile", None) or lstm_cfg.threshold_quantile

                log.info("[LSTM_AE][THRESH] point_threshold(point_q=%.4f)=%.6f", lstm_cfg.threshold_quantile, pt)
                if rt is not None:
                    log.info("[LSTM_AE][THRESH] run_threshold(run_score_q=%s, run_q=%.4f)=%.6f", str(rsq), run_q, rt)
                else:
                    log.info("[LSTM_AE][THRESH] run_threshold=None (point-level decision)")
                if wandb.run is not None:
                    wandb.log({"time/train_elapsed_sec": train_elapsed})

                # 추론
                infer_start = time.perf_counter()
                with log_time(log, "lstm_ae/inference_lstm_autoencoder"):
                    pred_df = inference_lstm_autoencoder(
                        test_X=sk_test_X,
                        inputs_are_preprocessed=True,
                        artifacts=artifacts,
                        cfg=lstm_cfg,
                        group_ids=test_group,
                    )
                log.info("[SINGLE][LSTM_AE] Inference done")
                log.info("[LSTM_AE] Prediction head:\n%s", pred_df.head())
                if wandb.run is not None:
                    labels = pred_df["faultNumber"].to_numpy()
                    wandb_log_pred_stats(labels, test_group, prefix="pred/LSTM_AE")

                infer_elapsed = time.perf_counter() - infer_start
                log.info("[LSTM_AE][INFER] elapsed: %.3f seconds", infer_elapsed)
                if wandb.run is not None:
                    wandb.log({"time/infer_elapsed_sec": infer_elapsed})

            # sklearn 단일 모델 로직
            else:
                internal_model_type, kwargs = _get_internal_model_and_kwargs(mt, cfg)
                log.info("[SINGLE] Training %s => %s", mt, internal_model_type)

                # 학습
                train_start = time.perf_counter()
                with log_time(log, f"sklearn/train/{mt}"):
                    model = train_model(train_X=sk_train_X, model_type=internal_model_type, **kwargs) # type: ignore
                train_elapsed = time.perf_counter() - train_start
                log.info("[TRAIN] elapsed: %.3f seconds", train_elapsed)
                if wandb.run is not None:
                    wandb.log({"time/train_elapsed_sec": train_elapsed})

                # 추론
                infer_start = time.perf_counter()
                with log_time(log, f"sklearn/infer/{mt}"):
                    pred_df = inference(test_X=sk_test_X, model=model)
                log.info("[SINGLE] Prediction head:\n%s", pred_df.head())
                if wandb.run is not None:
                    labels = pred_df["faultNumber"].to_numpy()
                    wandb_log_pred_stats(labels, test_group, prefix=f"pred/{mt}")

                infer_elapsed = time.perf_counter() - infer_start
                log.info("[INFER] elapsed: %.3f seconds", infer_elapsed)
                if wandb.run is not None:
                    wandb.log({"time/infer_elapsed_sec": infer_elapsed})

        # 공통 후처리 (제출 파일 저장 & W&B artifact 업로드)
        total_elapsed = time.perf_counter() - total_start
        log.info("[TOTAL] Full run elapsed: %.3f seconds", total_elapsed)
        if wandb.run is not None:
            wandb.log({"time/total_elapsed_sec": total_elapsed})

        # 4. 저장 (`,faultNumber` 형식)
        out_path = Path(to_absolute_path(cfg.output_path))
        with log_time(log, "save/submission_csv"):
            save_predictions_as_submission(pred_df, out_path)

        # 5. 제출 파일을 W&B Artifact로 업로드
        if wandb.run is not None:
            with log_time(log, "wandb/log_artifact"):
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

    except Exception:
        # 어떤 단계에서든 예외가 나면 stacktrace를 로그로 남기고 재-raise
        log.exception("Run failed with exception")
        raise
    finally:
        # W&B는 예외 여부와 무관하게 종료(중복 호출은 wandb가 대체로 안전하게 처리)
        if run is not None:
            try:
                wandb.finish()
            except Exception:
                log.exception("wandb.finish() failed")


if __name__ == "__main__":
    hydra_app()
