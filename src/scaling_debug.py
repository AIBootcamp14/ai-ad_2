# debug_ee_scaling.py
import numpy as np
import polars as pl

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

# 선택(있으면 더 보기 좋음)
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

from data_utils import process_data


def summarize_matrix(X: np.ndarray, cols: list[str], name: str, topk: int = 10):
    # per-feature 요약
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    q25 = np.quantile(X, 0.25, axis=0)
    med = np.quantile(X, 0.50, axis=0)
    q75 = np.quantile(X, 0.75, axis=0)
    iqr = q75 - q25

    # std 큰 피처 top-k만 출력(전체 출력하면 너무 김)
    idx = np.argsort(-std)[:topk]
    print(f"\n[{name}] per-feature summary (top {topk} by std)")
    print("col | mean | std | min | q25 | median | q75 | max | IQR")
    for i in idx:
        print(
            f"{cols[i]} | {mean[i]:.4g} | {std[i]:.4g} | {mn[i]:.4g} | "
            f"{q25[i]:.4g} | {med[i]:.4g} | {q75[i]:.4g} | {mx[i]:.4g} | {iqr[i]:.4g}"
        )


def fit_predict_score(X_train: np.ndarray, X_test: np.ndarray, *, n_neighbors=20, contamination=0.02, n_jobs=-1, novelty=True, metric="minkowski", random_state=42):
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        n_jobs=n_jobs,
        novelty=novelty,
        metric=metric,
    )
    lof.fit(X_train)

    # sklearn: +1 정상, -1 이상치
    pred_pm = lof.predict(X_test)
    pred01 = (pred_pm == -1).astype(np.int32)

    # decision_function: 값이 클수록 정상(대체로 0 근처가 경계)
    score = lof.decision_function(X_test)

    return pred01, score, lof


def basic_compare(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str):
    agree = (a == b).mean()
    print(f"Agreement({name_a} vs {name_b}) = {agree:.6f}  (diff={1.0-agree:.6f})")


def corr(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x); y = np.asarray(y)
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def rank_corr_spearman_approx(x: np.ndarray, y: np.ndarray):
    # scipy 없이 스피어만 근사(랭크 후 피어슨)
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    return corr(rx, ry)


def print_score_stats(score: np.ndarray, name: str):
    qs = np.quantile(score, [0.0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0])
    print(f"\n[{name}] score stats (decision_function)")
    print(f"mean={score.mean():.6g}, std={score.std():.6g}")
    print("q   : 0    1%    5%    10%   50%   90%   95%   99%   100%")
    print("val :", " ".join(f"{v:.6g}" for v in qs))


def main():
    # ✅ 경로는 필요에 맞게 수정
    train_path = "data/train.csv"
    test_path = "data/test.csv"

    train_df = pl.read_csv(train_path)
    test_df = pl.read_csv(test_path)

    # 시간축 보장(권장)
    if set(["simulationRun", "sample"]).issubset(train_df.columns):
        train_df = train_df.sort(["simulationRun", "sample"])
        test_df = test_df.sort(["simulationRun", "sample"])

    train_X_df = process_data(train_df)  # xmeas/xmv만 사용
    test_X_df = process_data(test_df)

    cols = train_X_df.columns
    Xtr_raw = train_X_df.to_numpy()
    Xte_raw = test_X_df.to_numpy()

    # --- 스케일링 ---
    std_scaler = StandardScaler(with_mean=True, with_std=True).fit(Xtr_raw)
    Xtr_std = std_scaler.transform(Xtr_raw)
    Xte_std = std_scaler.transform(Xte_raw)

    rob_scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)).fit(Xtr_raw)
    Xtr_rob = rob_scaler.transform(Xtr_raw)
    Xte_rob = rob_scaler.transform(Xte_raw)

    # --- “스케일링이 실제로 달라졌는지” 체크 ---
    print("\n=== Scaling sanity checks ===")
    print("allclose(train_raw, train_std) =", np.allclose(Xtr_raw, Xtr_std))
    print("allclose(train_raw, train_rob) =", np.allclose(Xtr_raw, Xtr_rob))
    print("mean(abs(train_raw-train_std)) =", float(np.mean(np.abs(Xtr_raw - Xtr_std))))
    print("mean(abs(train_raw-train_rob)) =", float(np.mean(np.abs(Xtr_raw - Xtr_rob))))

    summarize_matrix(Xtr_raw, cols, "TRAIN_RAW", topk=10)
    summarize_matrix(Xtr_std, cols, "TRAIN_STANDARD", topk=10)
    summarize_matrix(Xtr_rob, cols, "TRAIN_ROBUST", topk=10)

    # --- EE 학습/추론: train은 동일(각 스케일 공간에서), test에서 비교 ---
    # contamination/support_fraction을 본인 실험과 동일하게 맞추면 더 좋습니다.
    contamination = 0.02
    seed = 42

    pred_raw, score_raw, _ = fit_predict_score(Xtr_raw, Xte_raw, contamination=contamination, random_state=seed)
    pred_std, score_std, _ = fit_predict_score(Xtr_std, Xte_std, contamination=contamination, random_state=seed)
    pred_rob, score_rob, _ = fit_predict_score(Xtr_rob, Xte_rob, contamination=contamination, random_state=seed)

    print("\n=== Prediction agreement (0/1; 1=anomaly) ===")
    basic_compare(pred_raw, pred_std, "RAW", "STANDARD")
    basic_compare(pred_raw, pred_rob, "RAW", "ROBUST")
    basic_compare(pred_std, pred_rob, "STANDARD", "ROBUST")

    # 예측이 다르다면 “어느 정도로 다르냐”를 한 번 더
    diff_raw_std = np.where(pred_raw != pred_std)[0]
    diff_raw_rob = np.where(pred_raw != pred_rob)[0]
    print("\n#diff(RAW vs STANDARD) =", diff_raw_std.size)
    print("#diff(RAW vs ROBUST)   =", diff_raw_rob.size)

    print("\n=== Score comparisons (decision_function) ===")
    print_score_stats(score_raw, "TEST_RAW")
    print_score_stats(score_std, "TEST_STANDARD")
    print_score_stats(score_rob, "TEST_ROBUST")

    print("\nCorr(score_raw, score_std) =", corr(score_raw, score_std))
    print("Corr(score_raw, score_rob) =", corr(score_raw, score_rob))
    print("SpearmanApprox(score_raw, score_std) =", rank_corr_spearman_approx(score_raw, score_std))
    print("SpearmanApprox(score_raw, score_rob) =", rank_corr_spearman_approx(score_raw, score_rob))

    # 분포 히스토그램(옵션)
    if HAS_PLT:
        plt.figure()
        plt.hist(score_raw, bins=100, alpha=0.5, label="raw")
        plt.hist(score_std, bins=100, alpha=0.5, label="standard")
        plt.hist(score_rob, bins=100, alpha=0.5, label="robust")
        plt.legend()
        plt.title("LocalOutlierFactor decision_function score distribution")
        plt.show()


if __name__ == "__main__":
    main()
