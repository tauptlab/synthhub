"""Run the public SynthHub benchmark.

The benchmark intentionally uses a small sklearn-bundled dataset so that it is
network-free, reproducible, and runnable in CI. Optional backends that are not
installed are reported as skipped instead of failing the whole run.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import load_breast_cancer

from synthhub import Synthesizer
from synthhub.errors import SynthHubError


DEFAULT_METHODS = (
    "independent",
    "privbayes",
    "datasynthesizer-independent",
    "aim",
    "mst",
    "mwem",
    "smartnoise-aim",
    "synthcity-privbayes",
)


@dataclass(frozen=True)
class BenchmarkRow:
    method: str
    backend: str
    status: str
    epsilon_spent: str
    utility_similarity: str
    tos_score: str
    reid_risk: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {
            "method": self.method,
            "backend": self.backend,
            "status": self.status,
            "epsilon_spent": self.epsilon_spent,
            "utility_similarity": self.utility_similarity,
            "train_on_synthetic_score": self.tos_score,
            "reid_risk": self.reid_risk,
            "detail": self.detail,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SynthHub's public backend benchmark.")
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--sample-rows", type=int, default=180)
    parser.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    real_df = load_demo_dataframe()
    rows = run_benchmark(
        real_df,
        methods=args.methods,
        epsilon=args.epsilon,
        sample_rows=args.sample_rows,
    )
    write_csv(rows, args.out_dir / "latest.csv")
    write_markdown(rows, args.out_dir / "latest.md", epsilon=args.epsilon, sample_rows=args.sample_rows)
    write_svg(rows, args.out_dir / "utility_vs_risk.svg")


def load_demo_dataframe() -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    df = data.frame.drop(columns=["target"]).copy()
    df["target"] = data.frame["target"].map({0: "malignant", 1: "benign"}).astype("object")
    keep = [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "mean smoothness",
        "worst radius",
        "worst texture",
        "worst area",
        "target",
    ]
    return df.loc[:, keep].rename(columns=lambda name: name.replace(" ", "_"))


def run_benchmark(
    real_df: pd.DataFrame,
    *,
    methods: list[str],
    epsilon: float,
    sample_rows: int,
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for index, method in enumerate(methods):
        try:
            synth = Synthesizer(method=method, epsilon=epsilon, random_state=100 + index)
            synth.fit(real_df)
            synth_df = synth.sample(sample_rows)
            report = synth.evaluate(real_df, synth_df, target="target").to_dict()
            tos = report["utility"]["train_on_synthetic"]
            tos_score = tos.get("accuracy", tos.get("r2"))
            risk = report["privacy"]["membership_inference"]
            accounting = report["privacy"]["accounting"] or {}
            rows.append(
                BenchmarkRow(
                    method=method,
                    backend=str(accounting.get("backend") or "unknown"),
                    status="ok",
                    epsilon_spent=_fmt(accounting.get("epsilon_spent")),
                    utility_similarity=_fmt(report["utility"]["mean_column_similarity"]),
                    tos_score=_fmt(tos_score),
                    reid_risk=_fmt(risk.get("risk_score")),
                    detail="",
                )
            )
        except Exception as exc:
            status = "skipped" if isinstance(exc, SynthHubError) else "failed"
            rows.append(
                BenchmarkRow(
                    method=method,
                    backend="",
                    status=status,
                    epsilon_spent="",
                    utility_similarity="",
                    tos_score="",
                    reid_risk="",
                    detail=str(exc).splitlines()[0],
                )
            )
    return rows


def write_csv(rows: list[BenchmarkRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].as_dict()))
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)


def write_markdown(rows: list[BenchmarkRow], path: Path, *, epsilon: float, sample_rows: int) -> None:
    lines = [
        "# SynthHub Benchmark Results",
        "",
        f"Dataset: `sklearn.datasets.load_breast_cancer` subset. Epsilon: `{epsilon}`. Synthetic rows: `{sample_rows}`.",
        "",
        "| Method | Backend | Status | Epsilon spent | Utility similarity | TSTR score | Re-ID risk | Detail |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {backend} | {status} | {epsilon_spent} | {utility_similarity} | "
            "{train_on_synthetic_score} | {reid_risk} | {detail} |".format(**_escaped(row.as_dict()))
        )
    lines.extend(
        [
            "",
            "`Utility similarity` is the mean per-column distribution similarity from `Synthesizer.evaluate`.",
            "`TSTR score` is train-on-synthetic, test-on-real accuracy for `target`.",
            "`Re-ID risk` is a nearest-neighbor membership-inference heuristic, not a DP proof.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg(rows: list[BenchmarkRow], path: Path) -> None:
    ok_rows = [
        row
        for row in rows
        if row.status == "ok" and row.utility_similarity and row.reid_risk
    ]
    width = 760
    height = 320
    margin_left = 150
    chart_width = 520
    bar_height = 22
    gap = 16
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}" viewBox="0 0 {0} {1}">'.format(width, height),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="32" font-family="Arial, sans-serif" font-size="20" font-weight="700">Utility vs Re-ID Risk</text>',
        '<text x="24" y="56" font-family="Arial, sans-serif" font-size="12" fill="#555">Higher utility is better. Lower risk is better.</text>',
    ]
    y = 90
    for row in ok_rows:
        utility = float(row.utility_similarity)
        risk = float(row.reid_risk)
        utility_width = int(chart_width * utility)
        risk_width = int(chart_width * risk)
        lines.extend(
            [
                f'<text x="24" y="{y + 15}" font-family="Arial, sans-serif" font-size="13">{_xml(row.method)}</text>',
                f'<rect x="{margin_left}" y="{y}" width="{utility_width}" height="{bar_height}" rx="3" fill="#2563eb"/>',
                f'<rect x="{margin_left}" y="{y + bar_height + 3}" width="{risk_width}" height="8" rx="2" fill="#dc2626"/>',
                f'<text x="{margin_left + chart_width + 10}" y="{y + 15}" font-family="Arial, sans-serif" font-size="12" fill="#333">U {_fmt(utility)} / R {_fmt(risk)}</text>',
            ]
        )
        y += bar_height + gap + 12
    lines.extend(
        [
            f'<line x1="{margin_left}" y1="{height - 36}" x2="{margin_left + chart_width}" y2="{height - 36}" stroke="#ddd"/>',
            f'<text x="{margin_left}" y="{height - 16}" font-family="Arial, sans-serif" font-size="11" fill="#555">0</text>',
            f'<text x="{margin_left + chart_width - 8}" y="{height - 16}" font-family="Arial, sans-serif" font-size="11" fill="#555">1</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.3f}"


def _escaped(row: dict[str, str]) -> dict[str, str]:
    return {key: value.replace("|", "\\|") for key, value in row.items()}


def _xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()
