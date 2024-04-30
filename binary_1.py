from pathlib import Path
from typing import Callable

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import polars as pl
import seaborn as sns
from sklearn.calibration import (
    CalibratedClassifierCV,
    CalibrationDisplay,
)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import (
    enable_halving_search_cv,  # noqa
    enable_iterative_imputer,  # noqa
)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.inspection import (
    PartialDependenceDisplay,
    permutation_importance,
)
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    DetCurveDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

sns.set_style("whitegrid")


def log_metrics(
    X_test: pl.DataFrame,
    y_test: np.ndarray,
    pipeline: Pipeline,
    cv_result: dict[str, np.ndarray],
) -> None:
    accuracy_oof = cv_result["test_accuracy_score"]
    average_precision_oof = cv_result["test_average_precision_score"]
    f1_oof = cv_result["test_f1_score"]
    precision_oof = cv_result["test_precision_score"]
    recall_oof = cv_result["test_recall_score"]
    roc_auc_oof = cv_result["test_roc_auc_score"]

    y_pred_proba_test = pipeline.predict_proba(X_test)[:, 1]
    y_pred_test = np.where(y_pred_proba_test > 0.5, 1, 0)
    accuracy_test = accuracy_score(y_true=y_test, y_pred=y_pred_test)
    average_precision_test = average_precision_score(
        y_true=y_test, y_score=y_pred_proba_test, average="macro"
    )
    f1_test = f1_score(y_true=y_test, y_pred=y_pred_test, average="macro")
    precision_test = precision_score(y_true=y_test, y_pred=y_pred_test, average="macro")
    recall_test = recall_score(y_true=y_test, y_pred=y_pred_test, average="macro")
    roc_auc_test = roc_auc_score(
        y_true=y_test, y_score=y_pred_proba_test, average="macro"
    )

    print("--- oof score ---")
    print(f"accuracy = {accuracy_oof.mean()}")
    print(f"average_precision = {average_precision_oof.mean()}")
    print(f"f1 = {f1_oof.mean()}")
    print(f"precision = {precision_oof.mean()}")
    print(f"recall = {recall_oof.mean()}")
    print(f"roc_auc = {roc_auc_oof.mean()}")
    print()
    print("--- test score ---")
    print(f"accuracy = {accuracy_test}")
    print(f"average_precision = {average_precision_test}")
    print(f"f1 = {f1_test}")
    print(f"precision = {precision_test}")
    print(f"recall = {recall_test}")
    print(f"roc_auc = {roc_auc_test}")
    print()


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


@hydra.main(version_base=None, config_path="conf", config_name="test")
def main(cfg: omegaconf.DictConfig) -> None:
    df = pl.read_csv(cfg.data.csv_path)
    df = df.rename(lambda x: x.replace(" ", "_"))
    df = df.cast({col: pl.String for col in cfg.data.cat_col_list})
    df = df.cast({col: pl.Categorical for col in cfg.data.cat_col_list})
    df_train, df_test = train_test_split(
        df,
        test_size=cfg.train_test_split.test_size,
        random_state=cfg.seed,
        shuffle=cfg.train_test_split.shuffle,
        stratify=df.select(pl.col("target")),
    )
    X_train: pl.DataFrame = df_train.select(pl.col("*").exclude("target"))
    y_train = df_train.select(pl.col("target")).to_numpy().reshape(-1)
    X_test: pl.DataFrame = df_test.select(pl.col("*").exclude("target"))
    y_test = df_test.select(pl.col("target")).to_numpy().reshape(-1)

    onehot_encoder = OneHotEncoder(
        categories="auto",
        drop="first",
        sparse_output=False,
        handle_unknown="infrequent_if_exist",
    )

    if cfg.data.imputer == "simple":
        imputer = SimpleImputer(
            missing_values=np.nan,
            strategy=cfg.data.imputer_params.simple.strategy,
            add_indicator=cfg.data.imputer_params.simple.add_indicator,
        )
    elif cfg.data.imputer == "iterative":
        if cfg.data.imputer_params.iterative.estmator == "bayes":
            estimator = BayesianRidge()
        elif cfg.data.imputer_params.iterative.estmator == "rf":
            estimator = RandomForestRegressor()
        imputer = IterativeImputer(
            estimator=estimator,
            missing_values=np.nan,
            sample_posterior=cfg.data.imputer_params.iterative.sample_posterior,
            max_iter=cfg.data.imputer_params.iterative.max_iter,
            tol=cfg.data.imputer_params.iterative.tol,
            initial_strategy=cfg.data.imputer_params.iterative.initial_strategy,
            imputation_order=cfg.data.imputer_params.iterative.imputation_order,
            add_indicator=cfg.data.imputer_params.iterative.add_indicator,
            random_state=cfg.seed,
        )
    elif cfg.data.imputer == "knn":
        imputer = KNNImputer(
            missing_values=np.nan,
            n_neighbors=cfg.data.imputer_params.knn.n_neighbors,
            weights=cfg.data.imputer_params.knn.weights,
            add_indicator=cfg.data.imputer_params.knn.add_indicator,
        )

    if cfg.data.scaler == "standard":
        scaler = StandardScaler(with_mean=True, with_std=True)
    elif cfg.data.scaler == "robust":
        scaler = RobustScaler(
            with_centering=True,
            with_scaling=True,
            quantile_range=(25.0, 75.0),
            unit_variance=False,
        )
    elif cfg.data.scaler == "power":
        scaler = PowerTransformer(method="yeo-johnson", standardize=True)
    elif cfg.data.scaler == "quantile":
        scaler = QuantileTransformer(
            output_distribution="normal", random_state=cfg.seed
        )

    step_preprocess = ColumnTransformer(
        transformers=[
            (
                "cat_process",
                onehot_encoder,
                omegaconf.OmegaConf.to_object(cfg.data.cat_col_list),
            ),
            (
                "num_process",
                Pipeline(steps=[("imputation", imputer), ("scaling", scaler)]),
                omegaconf.OmegaConf.to_object(cfg.data.num_col_list),
            ),
        ]
    )
    step_model = LogisticRegression(random_state=cfg.seed)

    splitter = RepeatedStratifiedKFold(
        n_splits=cfg.splitter.n_splits,
        n_repeats=cfg.splitter.n_repeats,
        random_state=cfg.seed,
    )

    scoring = {
        "accuracy_score": make_scorer(accuracy_score),
        "average_precision_score": make_scorer(
            average_precision_score, average="macro"
        ),
        "f1_score": make_scorer(f1_score, average="macro"),
        "precision_score": make_scorer(precision_score, average="macro"),
        "recall_score": make_scorer(recall_score, average="macro"),
        "roc_auc_score": make_scorer(roc_auc_score, average="macro"),
    }

    pipeline = Pipeline(
        steps=[
            ("preprocess", step_preprocess),
            ("model", step_model),
        ]
    )
    search_result = GridSearchCV(
        estimator=pipeline,
        param_grid={"model__C": [1.0 * i for i in np.linspace(1.0e-4, 1.0e4, num=10)]},
        scoring=scoring,
        n_jobs=-1,
        refit=cfg.eval_metric,
        cv=splitter,
    )
    search_result.fit(X=X_train, y=y_train)
    pipeline.set_params(**search_result.best_params_)

    y_pred_proba_oof = cross_val_predict(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=splitter,
        n_jobs=-1,
        method="predict_proba",
    )[:, 1]
    y_pred_oof = np.where(y_pred_proba_oof > 0.5, 1, 0)

    pipeline.fit(X=X_train, y=y_train)
    y_pred_proba_test = pipeline.predict_proba(X_test)[:, 1]
    y_pred_test = np.where(y_pred_proba_test > 0.5, 1, 0)

    # pipeline_calibrated_sigmoid_prefit = Pipeline(
    #     steps=[
    #         ("preprocess", step_preprocess),
    #         ("model", step_model),
    #     ]
    # )
    # pipeline_calibrated_sigmoid_prefit.set_params(**search_result.best_params_)
    # pipeline_calibrated_sigmoid_prefit = CalibratedClassifierCV(
    #     estimator=pipeline_calibrated_sigmoid_prefit,
    #     method="sigmoid",
    #     cv=splitter,
    #     n_jobs=-1,
    #     ensemble=False,
    # )
    # pipeline_calibrated_sigmoid_prefit.fit(X=X_train, y=y_train)

    # pipeline_calibrated_isotonic_prefit = Pipeline(
    #     steps=[
    #         ("preprocess", step_preprocess),
    #         ("model", step_model),
    #     ]
    # )
    # pipeline_calibrated_isotonic_prefit.set_params(**search_result.best_params_)
    # pipeline_calibrated_isotonic_prefit = CalibratedClassifierCV(
    #     estimator=pipeline,
    #     method="isotonic",
    #     cv=splitter,
    #     n_jobs=-1,
    #     ensemble=False,
    # )
    # pipeline_calibrated_isotonic_prefit.fit(X=X_train, y=y_train)

    feature_names = list(pipeline.feature_names_in_)
    save_dir = Path(cfg.save_dir)
    save_dir_pipeline = save_dir / "pipeline"
    save_dir_pipeline_calibrated_sigmoid_prefit = (
        save_dir / "pipeline_calibrated_sigmoid_prefit"
    )
    save_dir_pipeline_calibrated_isotonic_prefit = (
        save_dir / "pipeline_calibrated_isotonic_prefit"
    )

    accuracy_oof = accuracy_score(y_true=y_train, y_pred=y_pred_oof)
    average_precision_oof = average_precision_score(
        y_true=y_train, y_score=y_pred_proba_oof, average="macro"
    )
    f1_oof = f1_score(y_true=y_train, y_pred=y_pred_oof, average="macro")
    precision_oof = precision_score(y_true=y_train, y_pred=y_pred_oof, average="macro")
    recall_oof = recall_score(y_true=y_train, y_pred=y_pred_oof, average="macro")
    roc_auc_oof = roc_auc_score(
        y_true=y_train, y_score=y_pred_proba_oof, average="macro"
    )

    accuracy_test = accuracy_score(y_true=y_test, y_pred=y_pred_test)
    average_precision_test = average_precision_score(
        y_true=y_test, y_score=y_pred_proba_test, average="macro"
    )
    f1_test = f1_score(y_true=y_test, y_pred=y_pred_test, average="macro")
    precision_test = precision_score(y_true=y_test, y_pred=y_pred_test, average="macro")
    recall_test = recall_score(y_true=y_test, y_pred=y_pred_test, average="macro")
    roc_auc_test = roc_auc_score(
        y_true=y_test, y_score=y_pred_proba_test, average="macro"
    )

    print("--- oof score ---")
    print(f"accuracy = {accuracy_oof}")
    print(f"average_precision = {average_precision_oof}")
    print(f"f1 = {f1_oof}")
    print(f"precision = {precision_oof}")
    print(f"recall = {recall_oof}")
    print(f"roc_auc = {roc_auc_oof}")
    print()
    print("--- test score ---")
    print(f"accuracy = {accuracy_test}")
    print(f"average_precision = {average_precision_test}")
    print(f"f1 = {f1_test}")
    print(f"precision = {precision_test}")
    print(f"recall = {recall_test}")
    print(f"roc_auc = {roc_auc_test}")
    print()

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

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    DetCurveDisplay.from_predictions(
        y_true=y_train,
        y_pred=y_pred_proba_oof,
        name="train",
        ax=ax,
    )
    DetCurveDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_pred_proba_test,
        name="test",
        ax=ax,
    )
    fig.tight_layout()
    save_path = save_dir_pipeline / "det_curve.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    PrecisionRecallDisplay.from_predictions(
        y_true=y_train,
        y_pred=y_pred_proba_oof,
        plot_chance_level=False,
        name="train",
        ax=ax,
    )
    PrecisionRecallDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_pred_proba_test,
        plot_chance_level=True,
        name="test",
        ax=ax,
    )
    fig.tight_layout()
    save_path = save_dir_pipeline / "pr_curve.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    RocCurveDisplay.from_predictions(
        y_true=y_train,
        y_pred=y_pred_proba_oof,
        plot_chance_level=False,
        name="train",
        ax=ax,
    )
    RocCurveDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_pred_proba_test,
        plot_chance_level=True,
        name="test",
        ax=ax,
    )
    fig.tight_layout()
    save_path = save_dir_pipeline / "roc_curve.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))

    # plot_feature_importance_and_partial_dependence(
    #     pipeline=pipeline,
    #     X_train=X_train,
    #     y_train=y_train,
    #     X_test=X_test,
    #     y_test=y_test,
    #     scoring=scoring,
    #     cfg=cfg,
    #     feature_names=feature_names,
    #     n_repeats=cfg.feature_importance.n_repeats,
    #     save_dir=save_dir_pipeline,
    # )
    # plot_calibration_curve(
    #     pipeline=pipeline,
    #     X_train=X_train,
    #     y_train=y_train,
    #     X_test=X_test,
    #     y_test=y_test,
    #     save_dir=save_dir_pipeline,
    # )

    # plot_feature_importance_and_partial_dependence(
    #     pipeline=pipeline_calibrated_sigmoid_prefit,
    #     X_train=X_train,
    #     y_train=y_train,
    #     X_test=X_test,
    #     y_test=y_test,
    #     scoring=scoring,
    #     cfg=cfg,
    #     feature_names=feature_names,
    #     n_repeats=cfg.feature_importance.n_repeats,
    #     save_dir=save_dir_pipeline_calibrated_sigmoid_prefit,
    # )
    # plot_calibration_curve(
    #     pipeline=pipeline_calibrated_sigmoid_prefit,
    #     X_train=X_train,
    #     y_train=y_train,
    #     X_test=X_test,
    #     y_test=y_test,
    #     save_dir=save_dir_pipeline_calibrated_sigmoid_prefit,
    # )

    # plot_feature_importance_and_partial_dependence(
    #     pipeline=pipeline_calibrated_isotonic_prefit,
    #     X_train=X_train,
    #     y_train=y_train,
    #     X_test=X_test,
    #     y_test=y_test,
    #     scoring=scoring,
    #     cfg=cfg,
    #     feature_names=feature_names,
    #     n_repeats=cfg.feature_importance.n_repeats,
    #     save_dir=save_dir_pipeline_calibrated_isotonic_prefit,
    # )
    # plot_calibration_curve(
    #     pipeline=pipeline_calibrated_isotonic_prefit,
    #     X_train=X_train,
    #     y_train=y_train,
    #     X_test=X_test,
    #     y_test=y_test,
    #     save_dir=save_dir_pipeline_calibrated_isotonic_prefit,
    # )


if __name__ == "__main__":
    main()
