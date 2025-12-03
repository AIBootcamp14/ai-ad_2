from __future__ import annotations

from typing import Literal, Union

import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

from config import (
    RANDOM_SEED,
    IsolationForestConfig,
    SGDOneClassSVMConfig,
    LOFConfig,
    EllipticEnvelopeCfg,
    DEFAULT_IFOREST_CONFIG,
    DEFAULT_SGDONECLASS_CONFIG,
    DEFAULT_LOF_CONFIG,
    DEFAULT_EE_CONFIG,
)

AnomalyModelType = Literal[
    "IsolationForest",
    "SGDOneClassSVM",
    "LocalOutlierFactor",
    "EllipticEnvelope",
]

AnomalyModel = Union[
    IsolationForest,
    SGDOneClassSVM,
    LocalOutlierFactor,
    EllipticEnvelope,
]


def build_isolation_forest(
    config: IsolationForestConfig | None = None,
) -> IsolationForest:
    """
    IsolationForestConfig 를 받아 IsolationForest 인스턴스를 생성하는 함수.
    """
    if config is None:
        config = DEFAULT_IFOREST_CONFIG

    model = IsolationForest(
        n_estimators=config.n_estimators,
        max_samples=config.max_samples,  # type: ignore[arg-type]
        contamination=config.contamination,
        max_features=config.max_features,
        bootstrap=config.bootstrap,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
        verbose=config.verbose,
        warm_start=config.warm_start,
    )
    return model


def build_sgd_one_class_svm(
    config: SGDOneClassSVMConfig | None = None,
) -> SGDOneClassSVM:
    """
    SGDOneClassSVMConfig 를 받아 SGDOneClassSVM 인스턴스를 생성하는 함수.
    (sklearn 버전에 따라 세부 파라미터는 조금 다를 수 있음)
    """
    if config is None:
        config = DEFAULT_SGDONECLASS_CONFIG

    model = SGDOneClassSVM(
        nu=config.nu,
        max_iter=config.max_iter,
        tol=config.tol,
        shuffle=config.shuffle,
        random_state=config.random_state,
        verbose=config.verbose,
    )
    return model


def build_lof(config: LOFConfig | None = None) -> LocalOutlierFactor:
    """
    LocalOutlierFactor(LOF) 모델 생성.
    novelty=True로 설정해야 새로운 test 데이터에 대해 predict 가능.
    """
    if config is None:
        config = DEFAULT_LOF_CONFIG

    model = LocalOutlierFactor(
        n_neighbors=config.n_neighbors,
        contamination=config.contamination,
        n_jobs=config.n_jobs,
        novelty=config.novelty,
        metric=config.metric,
    )
    return model


def build_elliptic_envelope(
    config: EllipticEnvelopeCfg | None = None,
) -> EllipticEnvelope:
    """
    EllipticEnvelope 모델 생성.
    """
    if config is None:
        config = DEFAULT_EE_CONFIG

    model = EllipticEnvelope(
        contamination=config.contamination,
        support_fraction=config.support_fraction,
        random_state=config.random_state,
    )
    return model


def build_model(
    model_type: AnomalyModelType,
    if_config: IsolationForestConfig | None = None,
    sgd_config: SGDOneClassSVMConfig | None = None,
    lof_config: LOFConfig | None = None,
    ee_config: EllipticEnvelopeCfg | None = None,
) -> AnomalyModel:
    if model_type == "IsolationForest":
        return build_isolation_forest(config=if_config)
    elif model_type == "SGDOneClassSVM":
        return build_sgd_one_class_svm(config=sgd_config)
    elif model_type == "LocalOutlierFactor":
        return build_lof(config=lof_config)
    elif model_type == "EllipticEnvelope":
        return build_elliptic_envelope(config=ee_config)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def train_model(
    train_X: pl.DataFrame,
    model_type: AnomalyModelType = "IsolationForest",
    if_config: IsolationForestConfig | None = None,
    sgd_config: SGDOneClassSVMConfig | None = None,
    lof_config: LOFConfig | None = None,
    ee_config: EllipticEnvelopeCfg | None = None,
) -> AnomalyModel:
    """
    Polars DataFrame 을 입력으로 받아 sklearn 이상탐지 모델을 학습합니다.
    내부에서는 numpy 배열로 변환하여 사용합니다.
    """
    model = build_model(
        model_type=model_type,
        if_config=if_config,
        sgd_config=sgd_config,
        lof_config=lof_config,
        ee_config=ee_config,
    )

    # sklearn 은 Polars DataFrame 을 직접 지원하지 않으므로
    # numpy 배열로 변환해서 사용합니다.
    X = train_X.to_numpy()
    model.fit(X)
    return model


def inference(test_X: pl.DataFrame, model: AnomalyModel) -> pl.DataFrame:
    """
    학습된 모델로 test_X 에 대한 이상 여부를 예측합니다.
    IsolationForest/OneClassSVM 의 결과(-1, 1)를 (0, 1) 레이블로 변환합니다.

    반환값은 단일 컬럼 'faultNumber' 를 가진 Polars DataFrame 입니다.
    """
    label_col = "faultNumber"
    X = test_X.to_numpy()
    pred_y = model.predict(X)

    # inliers: 1 -> 0 , outliers: -1 -> 1
    if -1 in pred_y:
        pred_y = (pred_y == -1).astype(int)

    return pl.DataFrame({label_col: pred_y})
