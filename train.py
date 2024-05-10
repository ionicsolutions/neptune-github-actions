import argparse
import timeit

import neptune
import neptune.integrations.sklearn as npt_utils
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


def load_data():
    print("[bold]Loading dataset[/bold]")
    data = load_iris()
    X, y = data.data, data.target
    print(f"Loaded dataset with {len(X)} samples.")
    return train_test_split(X, y, test_size=1 / 3)


def train(
    data: tuple,
    criterion: str = "gini",
    min_samples_split: int = 2,
    max_depth: int = 5,
    min_samples_leaf: int = 1,
    max_leaf_nodes: int = 5,
):
    print("Unpacking training and evaluation data...")
    X_train, X_test, y_train, y_test = data

    print("[bold]Instantiating model...[/bold]")
    model_parameters = {
        "criterion": criterion,
        "min_samples_split": min_samples_split,
        "max_depth": max_depth,
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
    run["evaluation/cls_summary"] = npt_utils.create_classifier_summary(
        model, X_train, X_test, y_train, y_test
    )
    run["visuals/confusion_matrix"] = npt_utils.create_confusion_matrix_chart(
        model, X_train, X_test, y_train, y_test
    )

    run["estimator/pickled-model"] = npt_utils.get_pickled_model(model)
    run.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--criterion", type=str)
    parser.add_argument("--min-samples-split", type=int)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--min-samples-leaf", type=int)
    parser.add_argument("--max-leaf_nodes", type=int)

    args = parser.parse_args()

    data = load_data()
    train(
        data,
        args.criterion,
        args.min_samples_split,
        args.max_depth,
        args.min_samples_leaf,
        args.max_leaf_nodes,
    )
