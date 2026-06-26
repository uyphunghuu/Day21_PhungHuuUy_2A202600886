import mlflow
client = mlflow.tracking.MlflowClient()
experiments = client.search_experiments()
for exp in experiments:
    runs = client.search_runs(exp.experiment_id)
    for r in runs:
        p = r.data.params
        m = r.data.metrics
        print(f"--- Run {r.info.run_id[:8]} ---")
        print(f"  Params: {dict(p)}")
        print(f"  Acc: {float(m['accuracy']):.4f} | F1: {float(m['f1_score']):.4f}")
        print()
