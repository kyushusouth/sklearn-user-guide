from pathlib import Path

import hydra
import numpy as np
import omegaconf
import polars as pl
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import (
    enable_halving_search_cv,  # noqa
    enable_iterative_imputer,  # noqa
)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import (
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

import wandb
from src.binary.utils import (
    log_metrics,
    log_multiple_figs,
    log_pr_curve_oof_vs_test,
    log_roc_curve_oof_vs_test,
)

sns.set_style("whitegrid")


@hydra.main(version_base=None, config_path="../../conf", config_name="test")
def main(cfg: omegaconf.DictConfig) -> None:
    wandb.init(
        project=cfg.wandb.project_name,
        group=cfg.wandb.group_name,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),  # type: ignore
    )

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
    )
    y_pred_oof = np.argmax(y_pred_proba_oof, axis=1)

    pipeline.fit(X=X_train, y=y_train)
    y_pred_proba_test = pipeline.predict_proba(X_test)
    y_pred_test = np.argmax(y_pred_proba_test, axis=1)

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

    log_metrics(
        y_train=y_train,
        y_pred_proba_oof=y_pred_proba_oof,
        y_pred_oof=y_pred_oof,
        y_test=y_test,
        y_pred_proba_test=y_pred_proba_test,
        y_pred_test=y_pred_test,
    )
    log_multiple_figs(
        y_train=y_train,
        y_pred_proba_oof=y_pred_proba_oof,
        y_test=y_test,
        y_pred_proba_test=y_pred_proba_test,
    )
    log_roc_curve_oof_vs_test(
        y_train=y_train,
        y_pred_proba_oof=y_pred_proba_oof,
        y_test=y_test,
        y_pred_proba_test=y_pred_proba_test,
    )
    log_pr_curve_oof_vs_test(
        y_train=y_train,
        y_pred_proba_oof=y_pred_proba_oof,
        y_test=y_test,
        y_pred_proba_test=y_pred_proba_test,
    )

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
