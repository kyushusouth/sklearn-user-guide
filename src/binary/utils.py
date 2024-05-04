from collections import Counter
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import plotly.graph_objects as go
import polars as pl
import seaborn as sns
from sklearn.calibration import (
    CalibratedClassifierCV,
    CalibrationDisplay,
)
from sklearn.inspection import (
    PartialDependenceDisplay,
    permutation_importance,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    DetCurveDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

import wandb


def save_roc_curve(
    y_train: np.ndarray,
    y_pred_proba_oof: np.ndarray,
    y_test: np.ndarray,
    y_pred_proba_test: np.ndarray,
    save_dir_pipeline: Path,
) -> None:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    RocCurveDisplay.from_predictions(
        y_true=y_train,
        y_pred=y_pred_proba_oof[:, 1],
        plot_chance_level=False,
        name="train",
        ax=ax,
    )
    RocCurveDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_pred_proba_test[:, 1],
        plot_chance_level=True,
        name="test",
        ax=ax,
    )
    fig.tight_layout()
    save_path = save_dir_pipeline / "roc_curve.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))


def save_pr_curve(
    y_train: np.ndarray,
    y_pred_proba_oof: np.ndarray,
    y_test: np.ndarray,
    y_pred_proba_test: np.ndarray,
    save_dir_pipeline: Path,
):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    PrecisionRecallDisplay.from_predictions(
        y_true=y_train,
        y_pred=y_pred_proba_oof[:, 1],
        plot_chance_level=True,
        name="train",
        ax=ax,
    )
    PrecisionRecallDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_pred_proba_test[:, 1],
        plot_chance_level=True,
        name="test",
        ax=ax,
    )
    fig.tight_layout()
    save_path = save_dir_pipeline / "pr_curve.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))


def save_det_curve(
    y_train: np.ndarray,
    y_pred_proba_oof: np.ndarray,
    y_test: np.ndarray,
    y_pred_proba_test: np.ndarray,
    save_dir_pipeline: Path,
):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    DetCurveDisplay.from_predictions(
        y_true=y_train,
        y_pred=y_pred_proba_oof[:, 1],
        name="train",
        ax=ax,
    )
    DetCurveDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_pred_proba_test[:, 1],
        name="test",
        ax=ax,
    )
    fig.tight_layout()
    save_path = save_dir_pipeline / "det_curve.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))


def save_confusion_matrix(
    y_train: np.ndarray,
    y_pred_oof: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    save_dir_pipeline: Path,
):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true=y_train,
        y_pred=y_pred_oof,
        normalize="true",
        cmap="viridis",
        ax=ax[0],
    )
    ax[0].set_title("oof")
    ax[0].grid(False)
    ConfusionMatrixDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_pred_test,
        normalize="true",
        cmap="viridis",
        ax=ax[1],
    )
    ax[1].set_title("test")
    ax[1].grid(False)
    fig.tight_layout()
    save_path = save_dir_pipeline / "confusion_matrix.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))


def log_metrics(
    y_train: np.ndarray,
    y_pred_proba_oof: np.ndarray,
    y_pred_oof: np.ndarray,
    y_test: np.ndarray,
    y_pred_proba_test: np.ndarray,
    y_pred_test: np.ndarray,
) -> None:
    accuracy_oof = accuracy_score(y_true=y_train, y_pred=y_pred_oof)
    average_precision_oof = average_precision_score(
        y_true=y_train, y_score=y_pred_proba_oof[:, 1], average="macro"
    )
    f1_oof = f1_score(y_true=y_train, y_pred=y_pred_oof, average="macro")
    precision_oof = precision_score(y_true=y_train, y_pred=y_pred_oof, average="macro")
    recall_oof = recall_score(y_true=y_train, y_pred=y_pred_oof, average="macro")
    roc_auc_oof = roc_auc_score(
        y_true=y_train, y_score=y_pred_proba_oof[:, 1], average="macro"
    )
    accuracy_test = accuracy_score(y_true=y_test, y_pred=y_pred_test)
    average_precision_test = average_precision_score(
        y_true=y_test, y_score=y_pred_proba_test[:, 1], average="macro"
    )
    f1_test = f1_score(y_true=y_test, y_pred=y_pred_test, average="macro")
    precision_test = precision_score(y_true=y_test, y_pred=y_pred_test, average="macro")
    recall_test = recall_score(y_true=y_test, y_pred=y_pred_test, average="macro")
    roc_auc_test = roc_auc_score(
        y_true=y_test, y_score=y_pred_proba_test[:, 1], average="macro"
    )
    wandb.log(
        {
            "accuracy_oof": accuracy_oof,
            "average_precision_oof": average_precision_oof,
            "f1_oof": f1_oof,
            "precision_oof": precision_oof,
            "recall_oof": recall_oof,
            "roc_auc_oof": roc_auc_oof,
            "accuracy_test": accuracy_test,
            "average_precision_test": average_precision_test,
            "f1_test": f1_test,
            "precision_test": precision_test,
            "recall_test": recall_test,
            "roc_auc_test": roc_auc_test,
        }
    )


def log_multiple_figs(
    y_train: np.ndarray,
    y_pred_proba_oof: np.ndarray,
    y_test: np.ndarray,
    y_pred_proba_test: np.ndarray,
) -> None:
    wandb.log(
        {
            "roc_curve_oof": wandb.plot.roc_curve(
                y_true=y_train,
                y_probas=y_pred_proba_oof,
                labels=None,
                classes_to_plot=None,
                title="roc_curve_oof",
            ),
            "roc_curve_test": wandb.plot.roc_curve(
                y_true=y_test,
                y_probas=y_pred_proba_test,
                labels=None,
                classes_to_plot=None,
                title="roc_curve_test",
            ),
            "pr_curve_oof": wandb.plot.pr_curve(
                y_true=y_train,
                y_probas=y_pred_proba_oof,
                labels=None,
                classes_to_plot=None,
                title="pr_curve_oof",
            ),
            "pr_curve_test": wandb.plot.pr_curve(
                y_true=y_test,
                y_probas=y_pred_proba_test,
                labels=None,
                classes_to_plot=None,
                title="pr_curve_test",
            ),
            "confusion_matrix_oof": wandb.plot.confusion_matrix(
                y_true=y_train,  # type: ignore
                probs=y_pred_proba_oof,  # type: ignore
                title="confusion_matrix_oof",
            ),
            "confusion_matrix_test": wandb.plot.confusion_matrix(
                y_true=y_test,  # type: ignore
                probs=y_pred_proba_test,  # type: ignore
                title="confusion_matrix_test",
            ),
        }
    )


def log_roc_curve_oof_vs_test(
    y_train: np.ndarray,
    y_pred_proba_oof: np.ndarray,
    y_test: np.ndarray,
    y_pred_proba_test: np.ndarray,
) -> None:
    fpr_oof, tpr_oof, _ = roc_curve(
        y_true=y_train,
        y_score=y_pred_proba_oof[:, 1],
        pos_label=1,
    )
    fpr_test, tpr_test, _ = roc_curve(
        y_true=y_test,
        y_score=y_pred_proba_test[:, 1],
        pos_label=1,
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr_oof,
            y=tpr_oof,
            mode="lines",
            name="oof",
            line={
                "color": "#3e59b8",
            },
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fpr_test,
            y=tpr_test,
            mode="lines",
            name="test",
            line={
                "color": "#f78528",
            },
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="ChanceLevel",
            line={
                "color": "#3c3d3c",
                "dash": "dot",
            },
        )
    )
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
    )
    wandb.log({"roc_curve_oof_vs_test": fig})


def log_pr_curve_oof_vs_test(
    y_train: np.ndarray,
    y_pred_proba_oof: np.ndarray,
    y_test: np.ndarray,
    y_pred_proba_test: np.ndarray,
) -> None:
    class_count_train = Counter(y_train)
    prevalence_pos_label_train = class_count_train[1] / sum(class_count_train.values())
    class_count_test = Counter(y_test)
    prevalence_pos_label_test = class_count_test[1] / sum(class_count_test.values())
    precision_oof, recall_oof, _ = precision_recall_curve(
        y_true=y_train,
        probas_pred=y_pred_proba_oof[:, 1],
        pos_label=1,
    )
    precision_test, recall_test, _ = precision_recall_curve(
        y_true=y_test,
        probas_pred=y_pred_proba_test[:, 1],
        pos_label=1,
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall_oof,
            y=precision_oof,
            mode="lines",
            name="oof",
            line={
                "color": "#3e59b8",
            },
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[prevalence_pos_label_train, prevalence_pos_label_train],
            mode="lines",
            name="oof_ChanceLevel",
            line={
                "color": "#3e59b8",
                "dash": "dot",
            },
        )
    )
    fig.add_trace(
        go.Scatter(
            x=recall_test,
            y=precision_test,
            mode="lines",
            name="test",
            line={
                "color": "#f78528",
            },
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[prevalence_pos_label_test, prevalence_pos_label_test],
            mode="lines",
            name="test_ChanceLevel",
            line={
                "color": "#f78528",
                "dash": "dot",
            },
        )
    )
    fig.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
    )
    wandb.log({"pr_curve_oof_vs_test": fig})


def plot_feature_importance_and_partial_dependence(
    pipeline: Pipeline | CalibratedClassifierCV,
    X_train: pl.DataFrame,
    y_train: np.ndarray,
    X_test: pl.DataFrame,
    y_test: np.ndarray,
    scoring: dict[str, Callable],
    cfg: omegaconf.DictConfig,
    feature_names: list[str],
    n_repeats: int,
    save_dir: Path,
) -> None:
    feat_imp_train = permutation_importance(
        estimator=pipeline,
        X=X_train.to_pandas(),
        y=y_train,
        scoring=scoring,
        n_repeats=n_repeats,
        n_jobs=-1,
        random_state=cfg.seed,
        max_samples=1.0,
    )
    feat_imp_test = permutation_importance(
        estimator=pipeline,
        X=X_test.to_pandas(),
        y=y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        n_jobs=-1,
        random_state=cfg.seed,
        max_samples=1.0,
    )
    for metric_name in scoring.keys():
        df_feat_imp = pl.concat(
            items=[
                pl.DataFrame(
                    data={
                        "data": "train",
                        "feature_name": feature_names * n_repeats,
                        "metric_name": metric_name,
                        "n": [
                            i
                            for i in range(n_repeats)
                            for _ in range(len(feature_names))
                        ],
                        "importance": feat_imp_train[metric_name][
                            "importances"
                        ].T.reshape(-1),
                    }
                ),
                pl.DataFrame(
                    data={
                        "data": "test",
                        "feature_name": feature_names * n_repeats,
                        "metric_name": metric_name,
                        "n": [
                            i
                            for i in range(n_repeats)
                            for _ in range(len(feature_names))
                        ],
                        "importance": feat_imp_test[metric_name][
                            "importances"
                        ].T.reshape(-1),
                    }
                ),
            ],
            how="vertical",
        )
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df_feat_imp, x="importance", y="feature_name", hue="data")
        plt.tight_layout()
        save_path = save_dir / f"feature_importance_{metric_name}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path))
        plt.close()

    feature_name_plot_list_train = list(
        df_feat_imp.filter(
            (pl.col("data") == "train") & (pl.col("metric_name") == cfg.eval_metric)
        )
        .group_by(pl.col("feature_name"))
        .agg(pl.col("importance").mean().alias("importance_mean"))
        .top_k(k=6, by="importance_mean", descending=False)
        .select(pl.col("feature_name"))
        .to_numpy()
        .reshape(-1)
    )
    feature_name_plot_list_test = list(
        df_feat_imp.filter(
            (pl.col("data") == "test") & (pl.col("metric_name") == cfg.eval_metric)
        )
        .group_by(pl.col("feature_name"))
        .agg(pl.col("importance").mean().alias("importance_mean"))
        .top_k(k=6, by="importance_mean", descending=False)
        .select(pl.col("feature_name"))
        .to_numpy()
        .reshape(-1)
    )
    feature_index_plot_list_train = [
        feature_names.index(feature_name_plot)
        for feature_name_plot in feature_name_plot_list_train
    ]
    feature_index_plot_list_test = [
        feature_names.index(feature_name_plot)
        for feature_name_plot in feature_name_plot_list_test
    ]
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 16))
    PartialDependenceDisplay.from_estimator(
        estimator=pipeline,
        X=X_train.to_pandas(),
        features=feature_index_plot_list_train,
        feature_names=feature_names,
        categorical_features=cfg.data.cat_col_list,
        response_method="auto",
        method="auto",
        n_jobs=-1,
        kind="average",
        centered=False,
        subsample=X_train.shape[0],
        ax=ax[0][0],
        random_state=cfg.seed,
    )
    ax[0][0].set_title(
        "train_pdp computed by features with high importance from the training data."
    )
    PartialDependenceDisplay.from_estimator(
        estimator=pipeline,
        X=X_test.to_pandas(),
        features=feature_index_plot_list_train,
        feature_names=feature_names,
        categorical_features=cfg.data.cat_col_list,
        response_method="auto",
        method="auto",
        n_jobs=-1,
        kind="average",
        centered=False,
        subsample=X_test.shape[0],
        ax=ax[0][1],
        random_state=cfg.seed,
    )
    ax[0][1].set_title(
        "test_pdp computed by features with high importance from the training data."
    )
    PartialDependenceDisplay.from_estimator(
        estimator=pipeline,
        X=X_train.to_pandas(),
        features=feature_index_plot_list_test,
        feature_names=feature_names,
        categorical_features=cfg.data.cat_col_list,
        response_method="auto",
        method="auto",
        n_jobs=-1,
        kind="average",
        centered=False,
        subsample=X_train.shape[0],
        ax=ax[1][0],
        random_state=cfg.seed,
    )
    ax[1][0].set_title(
        "train_pdp computed by features with high importance from the test data."
    )
    PartialDependenceDisplay.from_estimator(
        estimator=pipeline,
        X=X_test.to_pandas(),
        features=feature_index_plot_list_test,
        feature_names=feature_names,
        categorical_features=cfg.data.cat_col_list,
        response_method="auto",
        method="auto",
        n_jobs=-1,
        kind="average",
        centered=False,
        subsample=X_test.shape[0],
        ax=ax[1][1],
        random_state=cfg.seed,
    )
    ax[1][1].set_title(
        "test_pdp computed by features with high importance from the test data."
    )
    plt.tight_layout()
    save_path = save_dir / "pdp.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path))
    plt.close()


def plot_calibration_curve(
    pipeline: Pipeline | CalibratedClassifierCV,
    X_train: pl.DataFrame,
    y_train: np.ndarray,
    X_test: pl.DataFrame,
    y_test: np.ndarray,
    save_dir: Path,
) -> None:
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    CalibrationDisplay.from_estimator(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        n_bins=10,
        strategy="uniform",
        pos_label=pipeline.classes_[1],
        ref_line=True,
        ax=ax[0],
    )
    ax[0].set_title("train")
    CalibrationDisplay.from_estimator(
        estimator=pipeline,
        X=X_test,
        y=y_test,
        n_bins=10,
        strategy="uniform",
        pos_label=pipeline.classes_[1],
        ref_line=True,
        ax=ax[1],
    )
    ax[1].set_title("test")
    plt.tight_layout()
    save_path = save_dir / "calibration.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path))
    plt.close()
