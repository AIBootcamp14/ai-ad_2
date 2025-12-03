from __future__ import annotations

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_utils import load_test_data, load_train_data


def plot_basic_hist(train_data: pl.DataFrame) -> None:
    """전체 numeric 컬럼에 대한 히스토그램을 그립니다."""
    n_cols = len(train_data.columns)
    n_rows = int(np.ceil(n_cols / 4))
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(train_data.columns):
        axes[i].hist(train_data[col].to_numpy())
        axes[i].set_title(col)

    # 남는 subplot 은 숨기기
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_simulation_and_sample(
    df: pl.DataFrame, n: int = 5000, title_prefix: str = ""
) -> None:
    """
    simulationRun 과 sample 을 같이 그리는 Plotly 시각화.
    train / test 데이터 모두에 사용할 수 있습니다.
    """
    subfig = make_subplots(specs=[[{"secondary_y": True}]])

    subset = df.head(n)
    idx = np.arange(subset.height)
    sim = subset["simulationRun"].to_numpy()
    sample = subset["sample"].to_numpy()

    fig1 = px.line(x=idx, y=sim)
    fig2 = px.line(x=idx, y=sample)
    fig2.update_traces(yaxis="y2")

    subfig.layout.yaxis.title = "simulationRun"  # type: ignore
    subfig.layout.yaxis2.title = "sample"  # type: ignore
    subfig.layout.title = f"{title_prefix} simulationRun and sample plot"

    subfig.add_traces(fig1.data + fig2.data)
    subfig.for_each_trace(lambda t: t.update(line=dict(color=t.marker.color)))
    subfig.show()


def plot_meas_mv_for_run(df: pl.DataFrame, sim_run: int = 1) -> None:
    """
    특정 simulationRun 에 대해 xmeas_1, xmv_1 을 함께 시각화합니다.
    """
    subfig = make_subplots(specs=[[{"secondary_y": True}]])

    subset = df.filter(pl.col("simulationRun") == sim_run)
    idx = np.arange(subset.height)
    xmeas = subset["xmeas_1"].to_numpy()
    xmv = subset["xmv_1"].to_numpy()

    fig1 = px.line(x=idx, y=xmeas)
    fig2 = px.line(x=idx, y=xmv)
    fig2.update_traces(yaxis="y2")

    subfig.layout.yaxis.title = "xmeas_1"  # type: ignore
    subfig.layout.yaxis2.title = "xmv_1"  # type: ignore
    subfig.layout.title = f"xmeas_1 and xmv_1 plot for simulationRun == {sim_run}"

    subfig.add_traces(fig1.data + fig2.data)
    subfig.for_each_trace(lambda t: t.update(line=dict(color=t.marker.color)))
    subfig.show()


def plot_correlation_heatmap(
    train_data: pl.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """
    상관계수 행렬을 계산하고, Heatmap 을 그립니다.
    상관계수(corr matrix)와 컬럼 이름 리스트를 반환합니다.
    """
    # 여기서는 모든 컬럼을 대상으로 상관계수를 계산합니다.
    cols = train_data.columns
    data = train_data.to_numpy()
    correlation_matrix = np.corrcoef(data, rowvar=False)

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix,
            x=cols,
            y=cols,
            colorscale="Viridis",
            text=np.round(correlation_matrix, 2),
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title="Correlation Matrix",
        xaxis=dict(title="Variables"),
        yaxis=dict(title="Variables"),
        width=800,
        height=800,
    )
    fig.show()

    return correlation_matrix, cols


def _compute_correlation_pairs(
    correlation_matrix: np.ndarray, columns: list[str]
) -> list[tuple[str, str, float]]:
    """
    correlation matrix 의 upper triangle 을
    (feature1, feature2, corr) 튜플 리스트로 변환합니다.
    """
    pairs: list[tuple[str, str, float]] = []
    n = len(columns)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((columns[i], columns[j], float(correlation_matrix[i, j])))
    return pairs


def plot_pairs(
    data1: np.ndarray,
    name1: str,
    data2: np.ndarray,
    name2: str,
    correlation_value: float,
) -> None:
    """
    두 시계열 컬럼을 같은 그래프에 서로 다른 y축으로 플롯합니다.
    """
    subfig = make_subplots(specs=[[{"secondary_y": True}]])

    idx = np.arange(len(data1))

    fig1 = px.line(x=idx, y=data1)
    fig2 = px.line(x=idx, y=data2)
    fig2.update_traces(yaxis="y2")

    subfig.layout.yaxis.title = name1  # type: ignore
    subfig.layout.yaxis2.title = name2  # type: ignore
    subfig.layout.title = (
        f"{name1} and {name2} plot (Correlation : {correlation_value:.4f})"
    )
    subfig.add_traces(fig1.data + fig2.data)
    subfig.for_each_trace(lambda t: t.update(line=dict(color=t.marker.color)))
    subfig.show()


def plot_top_bottom_correlations(
    df: pl.DataFrame,
    correlation_matrix: np.ndarray,
    column_names: list[str],
    n_top: int = 5,
    n_bottom: int = 5,
    sim_run: int = 1,
) -> None:
    """
    상관계수가 가장 높은 상위 n_top 쌍과 가장 낮은 n_bottom 쌍에 대해
    simulationRun == sim_run 인 구간만 추출하여 plot_pairs 로 시각화합니다.
    """
    pairs = _compute_correlation_pairs(correlation_matrix, column_names)

    # 상위/하위 상관계수 선택
    sorted_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
    top_n = sorted_pairs[:n_top]
    bottom_n = sorted(pairs, key=lambda x: x[2])[:n_bottom]

    plot_data = df.filter(pl.col("simulationRun") == sim_run)

    print("[Top correlations]")
    for feature_1, feature_2, corr_val in top_n:
        print(f"{feature_1:20s} - {feature_2:20s} : {corr_val:.4f}")
        data1 = plot_data[feature_1].to_numpy()
        data2 = plot_data[feature_2].to_numpy()
        plot_pairs(
            data1=data1,
            name1=feature_1,
            data2=data2,
            name2=feature_2,
            correlation_value=corr_val,
        )

    print("\n[Bottom correlations]")
    for feature_1, feature_2, corr_val in bottom_n:
        print(f"{feature_1:20s} - {feature_2:20s} : {corr_val:.4f}")
        data1 = plot_data[feature_1].to_numpy()
        data2 = plot_data[feature_2].to_numpy()
        plot_pairs(
            data1=data1,
            name1=feature_1,
            data2=data2,
            name2=feature_2,
            correlation_value=corr_val,
        )


def run_eda(train_output: bool = True, test_output: bool = True) -> None:
    """
    EDA만 따로 실행하고 싶을 때 사용하는 함수입니다.
    """
    train_df = load_train_data()
    test_df = load_test_data()

    if train_output:
        print("[Train] shape:", train_df.shape)
        print(train_df.describe())
        plot_basic_hist(train_df)
        plot_simulation_and_sample(train_df, n=5000, title_prefix="[Train]")
        plot_meas_mv_for_run(train_df, sim_run=1)

        corr, cols = plot_correlation_heatmap(train_df)
        plot_top_bottom_correlations(
            df=train_df,
            correlation_matrix=corr,
            column_names=cols,
            n_top=5,
            n_bottom=5,
            sim_run=1,
        )

    if test_output:
        print("[Test] shape:", test_df.shape)
        print(test_df.describe())
        plot_simulation_and_sample(test_df, n=5000, title_prefix="[Test]")
