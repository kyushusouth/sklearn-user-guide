import functools

import hydra
import lightgbm
import numpy as np
import omegaconf
import optuna
import polars as pl
import seaborn as sns
import wandb
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
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
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

from src.binary.utils import (
    log_metrics,
    log_multiple_figs,
    log_pr_curve_oof_vs_test,
    log_roc_curve_oof_vs_test,
)

sns.set_style("whitegrid")


def objective(
    trial: optuna.trial.Trial,
    cfg: omegaconf.DictConfig,
    X_train: pl.DataFrame,
    y_train: np.ndarray,
    splitter: RepeatedStratifiedKFold,
    step_preprocess: ColumnTransformer,
    step_model: LogisticRegression,
):
    params = {
        "model__learing_rate": trial.suggest_uniform(
            "model__learing_rate", 1.0e-3, 0.1
        ),
        "model__num_leaves": trial.suggest_int("model__num_leaves", 16, 31, step=1),
        "model__reg_alpha": trial.suggest_uniform("model__reg_alpha", 0, 1),
        "model__reg_lambda": trial.suggest_uniform("model__reg_lambda", 0, 1),
    }
    pipeline = Pipeline(
        steps=[
            ("preprocess", step_preprocess),
            ("model", step_model),
        ]
    )
    pipeline.set_params(**params)

    y_pred_proba_oof_list = []
    for train_index, val_index in splitter.split(X_train, y_train):  # type: ignore
        pipeline.fit(X=X_train[train_index], y=y_train[train_index])
        y_pred_proba = pipeline.predict_proba(X_train[val_index])
        y_pred_proba_oof_list.append(y_pred_proba)
    y_pred_proba_oof = np.concatenate(y_pred_proba_oof_list, axis=0)
    y_pred_oof = np.argmax(y_pred_proba_oof, axis=1)

    accuracy_oof = accuracy_score(y_true=y_train, y_pred=y_pred_oof)
    f1_oof = f1_score(y_true=y_train, y_pred=y_pred_oof, average="macro")
    precision_oof = precision_score(y_true=y_train, y_pred=y_pred_oof, average="macro")
    recall_oof = recall_score(y_true=y_train, y_pred=y_pred_oof, average="macro")
    average_precision_oof = average_precision_score(
        y_true=y_train, y_score=y_pred_proba_oof[:, 1], average="macro"
    )
    roc_auc_oof = roc_auc_score(
        y_true=y_train, y_score=y_pred_proba_oof[:, 1], average="macro"
    )

    if cfg.eval_metric == "accuracy_oof":
        return accuracy_oof
    elif cfg.eval_metric == "f1_oof":
        return f1_oof
    elif cfg.eval_metric == "precision_oof":
        return precision_oof
    elif cfg.eval_metric == "recall_oof":
        return recall_oof
    elif cfg.eval_metric == "average_precision_oof":
        return average_precision_oof
    elif cfg.eval_metric == "roc_auc_oof":
        return roc_auc_oof


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
    step_model = lightgbm.LGBMClassifier(
        boosting_type="gbdt",
        n_estimators=1000,
        max_depth=7,
        objective="binary",
        random_state=cfg.seed,
        n_jobs=-1,
        importance_type="gain",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", step_preprocess),
            ("model", step_model),
        ]
    )

    splitter = RepeatedStratifiedKFold(
        n_splits=cfg.splitter.n_splits,
        n_repeats=cfg.splitter.n_repeats,
        random_state=cfg.seed,
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(
        func=functools.partial(
            objective,
            cfg=cfg,
            X_train=X_train,
            y_train=y_train,
            splitter=splitter,
            step_preprocess=step_preprocess,
            step_model=step_model,
        ),  # type: ignore
        n_trials=100,
    )
    pipeline.set_params(**study.best_params)

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


if __name__ == "__main__":
    main()
