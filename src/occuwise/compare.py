"""Build the leaderboard across all tracked runs.

Reads every finished MLflow run and produces, per dataset, a ranked table of
architectures by that dataset's primary metric — this is the "which combination
wins" view. Outputs `outputs/leaderboard.csv` and `outputs/leaderboard.md`.

    py -m occuwise.compare [--tracking-uri ./mlruns]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data.registry import DATASETS


def collect_runs(tracking_uri: str) -> pd.DataFrame:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    rows = []
    for exp in mlflow.search_experiments():
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        if runs.empty:
            continue
        for _, r in runs.iterrows():
            rows.append(r.to_dict())
    return pd.DataFrame(rows)


def build_leaderboard(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    records = []
    for _, r in runs.iterrows():
        name = r.get("tags.mlflow.runName", "")
        if "__" not in str(name):
            continue
        dataset, arch = str(name).split("__", 1)
        spec = DATASETS.get(dataset)
        if spec is None:
            continue
        metric_col = f"metrics.test/{spec.primary_metric}"
        score = r.get(metric_col)
        if pd.isna(score):
            continue
        records.append(
            {"dataset": dataset, "task": spec.task, "modality": spec.modality,
             "arch": arch, "metric": spec.primary_metric, "score": float(score)}
        )
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.sort_values(["dataset", "score"], ascending=[True, False])
    df["rank"] = df.groupby("dataset")["score"].rank(ascending=False, method="min").astype(int)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-uri", default="./mlruns")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    runs = collect_runs(args.tracking_uri)
    board = build_leaderboard(runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if board.empty:
        print("No completed runs with test metrics found yet.")
        return

    board.to_csv(out / "leaderboard.csv", index=False)
    md = ["# Occuwise Leaderboard\n"]
    for dataset, grp in board.groupby("dataset"):
        best = grp.iloc[0]
        md.append(f"\n## {dataset}  ·  metric: `{best['metric']}`  ·  best: "
                  f"**{best['arch']}** ({best['score']:.4f})\n")
        md.append(grp[["rank", "arch", "score"]].to_markdown(index=False))
        md.append("")
    (out / "leaderboard.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nWrote {out/'leaderboard.csv'} and {out/'leaderboard.md'}")


if __name__ == "__main__":
    main()
