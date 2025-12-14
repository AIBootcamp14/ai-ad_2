from pathlib import Path
from dataclasses import dataclass

# 공통 설정 값들
RANDOM_SEED: int = 42

# 데이터가 있는 디렉토리
DATA_DIR: Path = Path("./data")

# 학습/추론에 사용할 수치 컬럼들
NUMERIC_COLS = [
    'xmeas_1', 'xmeas_10', 'xmeas_11', 'xmeas_12', 'xmeas_13', 'xmeas_14',
    'xmeas_15', 'xmeas_16', 'xmeas_17', 'xmeas_18', 'xmeas_19', 'xmeas_2',
    'xmeas_20', 'xmeas_21', 'xmeas_22', 'xmeas_23', 'xmeas_24', 'xmeas_25',
    'xmeas_26', 'xmeas_27', 'xmeas_28', 'xmeas_29', 'xmeas_3', 'xmeas_30',
    'xmeas_31', 'xmeas_32', 'xmeas_33', 'xmeas_34', 'xmeas_35', 'xmeas_36',
    'xmeas_37', 'xmeas_38', 'xmeas_39', 'xmeas_4', 'xmeas_40', 'xmeas_41',
    'xmeas_5', 'xmeas_6', 'xmeas_7', 'xmeas_8', 'xmeas_9',
    'xmv_1', 'xmv_10', 'xmv_11', 'xmv_2', 'xmv_3', 'xmv_4',
    'xmv_5', 'xmv_6', 'xmv_7', 'xmv_8', 'xmv_9'
]


@dataclass
class IsolationForestConfig:
    """
    IsolationForest 하이퍼파라미터 설정용 dataclass.
    """
    n_estimators: int = 200                   # 트리 개수
    max_samples: str | int | float = "auto"   # 각 트리에서 사용할 샘플 수
    contamination: float | str = "auto"       # 이상치 비율 (비지도면 "auto" 추천)
    max_features: float = 1.0                 # feature 비율
    bootstrap: bool = False
    n_jobs: int | None = -1
    random_state: int | None = None
    verbose: int = 0
    warm_start: bool = False


@dataclass
class SGDOneClassSVMConfig:
    """
    SGDOneClassSVM 하이퍼파라미터 설정용 dataclass.
    (여기서는 자주 건드릴만한 몇 개만 노출해둔다)
    """
    nu: float = 0.01                # 이상치 비율 비슷한 역할(클래스 분리 마진)
    max_iter: int = 1000
    tol: float = 1e-3
    shuffle: bool = True
    random_state: int | None = None
    verbose: int = 0
    # 필요하면 loss, penalty, alpha 등도 추가 가능

# LocalOutlierFactor(LOF)용 Config
@dataclass
class LOFConfig:
    n_neighbors: int = 20
    contamination: float | str = "auto"  # float or "auto"
    n_jobs: int | None = -1
    novelty: bool = True                 # True여야 predict로 새로운 데이터 예측 가능
    metric: str = "minkowski"

# EllipticEnvelope용 Config
@dataclass
class EllipticEnvelopeConfig:
    contamination: float = 0.1
    support_fraction: float | None = None
    random_state: int | None = None

# LSTM AutoEncoder용 Config (딥러닝용)
@dataclass
class LSTMAEConfig:
    seed: int | None = None
    # 입력 차원은 train_X.shape[1]에서 자동으로 설정하게 할 것이므로 보통 None으로 둠
    input_dim: int | None = None

    # LSTM AutoEncoder 구조 관련
    hidden_dim: int = 64
    latent_dim: int = 16
    num_layers: int = 2
    dropout: float = 0.1

    # 시계열 윈도우 관련
    window_size: int = 30
    window_stride: int = 1

    # 학습 관련
    batch_size: int = 128
    num_epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    clip_grad_norm: float | None = 1.0

    # device 지정 (None이면 자동으로 "cuda" 또는 "cpu")
    device: str | None = None

    # 이상치 판별 임계값 (train point-wise reconstruction error 상위 q%)
    threshold_quantile: float = 0.995

    # epoch별 로그 출력 여부
    print_progress: bool = True


# 기본값
DEFAULT_IFOREST_CONFIG = IsolationForestConfig()
DEFAULT_SGDONECLASS_CONFIG = SGDOneClassSVMConfig()
DEFAULT_LOF_CONFIG = LOFConfig()
DEFAULT_EE_CONFIG = EllipticEnvelopeConfig()
DEFAULT_LSTMAE_CONFIG = LSTMAEConfig()
