import argparse
import timeit
from typing import Literal

import neptune
import neptune.integrations.sklearn as npt_utils
import numpy as np
from pydantic_settings import BaseSettings
from rich import print
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class Settings(BaseSettings):
    NEPTUNE_PROJECT: str
    NEPTUNE_API_TOKEN: str


settings = Settings()


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("[bold]Loading dataset[/bold]")
    iris_dataset = load_iris()
    X, y = iris_dataset.data, iris_dataset.target
    print(f"Loaded dataset with {len(X)} samples.")
    return train_test_split(X, y, test_size=1 / 3)


def train(
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    criterion: Literal["gini", "entropy", "log_loss"] = "gini",
    max_depth: int = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_leaf_nodes: int = 5,
):
    print("[bold]Training a DecisionTreeClassifier[/bold]")
    print("Unpacking training and evaluation data...")
    X_train, X_test, y_train, y_test = data

    print("[bold]Instantiating model...[/bold]")
    model_parameters = {
        "criterion": criterion,
        "splitter": "best",
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_leaf_nodes": max_leaf_nodes,
    }
    print(model_parameters)
    model = DecisionTreeClassifier(**model_parameters)

    print("[bold]Fitting model...[/bold]")
    start = timeit.default_timer()
    model.fit(X_train, y_train)
    fit_duration = timeit.default_timer() - start
    print(f"Fitted model in {fit_duration:.2f} seconds.")

    print("[bold]Evaluating model...[/bold]")
    y_pred = model.predict(X_test)
    evaluation = {
        "f1_score": f1_score(y_test, y_pred, average="macro"),
        "accuracy": accuracy_score(y_test, y_pred),
    }
    print(evaluation)

    print("🚀 [purple bold]Logging data to neptune.ai[/purple bold]")
    run = neptune.init_run(
        project=settings.NEPTUNE_PROJECT,
        api_token=settings.NEPTUNE_API_TOKEN,
    )
    run["model/parameters"] = model_parameters
    run["train/duration"] = fit_duration

    run["evaluation"] = evaluation
    run["visuals/confusion_matrix"] = npt_utils.create_confusion_matrix_chart(
        model, X_train, X_test, y_train, y_test
    )
    run["estimator/pickled-model"] = npt_utils.get_pickled_model(model)

    run["summary"] = npt_utils.create_classifier_summary(
        model, X_train, X_test, y_train, y_test
    )

    run.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--criterion", type=str)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--min-samples-split", type=int)
    parser.add_argument("--min-samples-leaf", type=int)
    parser.add_argument("--max-leaf-nodes", type=int)
    args = parser.parse_args()

    data = load_data()
    train(
        data,
        criterion=args.criterion,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_leaf_nodes=args.max_leaf_nodes,
    )
