import hydra
import omegaconf
import polars as pl
from flaml import AutoML
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split


@hydra.main(version_base=None, config_path="../../conf", config_name="test")
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

    splitter = RepeatedStratifiedKFold(
        n_splits=cfg.splitter.n_splits,
        n_repeats=cfg.splitter.n_repeats,
        random_state=cfg.seed,
    )

    automl = AutoML()
    automl.fit(
        X_train=X_train.to_pandas(),
        y_train=y_train,
        task="classification",
        estimator_list=[
            "lgbm",
            "xgboost",
            "xgb_limitdepth",
            "rf",
            "extra_tree",
            "histgb",
            # "lrl1",
            # "lrl2",
            # "kneighbor",
        ],
        metric="roc_auc",
        time_budget=10,
        # ensemble={
        #     "final_estimator": DecisionTreeClassifier(random_state=cfg.seed),
        #     "passthrough": True,
        # },
        eval_method="cv",
        split_type=splitter,
        seed=cfg.seed,
        n_jobs=-1,
        retrain_full=True,
        early_stop=False,
    )
    breakpoint()


if __name__ == "__main__":
    main()
