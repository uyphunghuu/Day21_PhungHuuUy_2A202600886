import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    data_path2: str = "data/train_phase2.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params      : dict chua cac sieu tham so cho RandomForestClassifier.
        data_path   : duong dan den file du lieu huan luyen phase 1.
        data_path2  : duong dan den file du lieu huan luyen phase 2 (optional).
        eval_path   : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """
    import os
    df1 = pd.read_csv(data_path)
    if os.path.exists(data_path2):
        df2 = pd.read_csv(data_path2)
        df_train = pd.concat([df1, df2], ignore_index=True)
        print(f"Training with combined data: {len(df_train)} samples ({len(df1)} + {len(df2)})")
    else:
        df_train = df1
        print(f"Training with phase1 only: {len(df_train)} samples")
    df_eval  = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval  = df_eval.drop(columns=["target"])
    y_eval  = df_eval["target"]

    with mlflow.start_run():

        mlflow.log_params(params)

        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc   = accuracy_score(y_eval, preds)
        f1    = f1_score(y_eval, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params, data_path="data/train_phase1.csv", data_path2="data/train_phase2.csv")
