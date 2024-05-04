import subprocess


def main():
    eval_metric_list = [
        "accuracy_oof",
        "f1_oof",
        "precision_oof",
        "recall_oof",
        "average_precision_oof",
        "roc_auc_oof",
    ]
    for eval_metric in eval_metric_list:
        subprocess.run(
            [
                "python",
                "/Users/minami/dev/python/sklearn-user-guide/src/binary/lightgbm_optuna.py",
                f"eval_metric={eval_metric}",
            ]
        )


if __name__ == "__main__":
    main()
