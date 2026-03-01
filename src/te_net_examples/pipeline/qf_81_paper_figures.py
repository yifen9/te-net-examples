from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from te_net_examples.io.csv import read_csv
from te_net_examples.io.json import write_json
from te_net_examples.utils.audit import Audit
from te_net_examples.utils.console import ConsoleSink
from te_net_examples.utils.jlog import jline
from te_net_examples.utils.logger import Logger
from te_net_examples.utils.meta import build_meta
from te_net_examples.utils.versioner import build_version_dir


@dataclass(frozen=True, slots=True)
class Qf81PaperFiguresOut:
    run_dir: str
    meta: dict[str, Any]
    summary_path: str


def _df_from_csv(path: str) -> pd.DataFrame:
    """严格复用 te_net_examples.io.csv 的基础 IO 组件"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required CSV not found: {path}")
    header, rows = read_csv(path, has_header=True)
    if header is None:
        raise ValueError(f"CSV missing header: {path}")
    return pd.DataFrame(rows, columns=header)


class PaperAesthetics:
    @staticmethod
    def apply_to_figure(fig: Figure, figsize: tuple[float, float] = (8, 5)) -> None:
        fig.set_size_inches(figsize)
        fig.set_dpi(300)
        fig.patch.set_facecolor("white")

    @staticmethod
    def apply_to_axes(ax: Any) -> None:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, color="#CCCCCC")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.tick_params(width=1.2, labelsize=10)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        ax.title.set_size(12)
        ax.title.set_weight("bold")


class BespokeFigureBuilder:
    @classmethod
    def _compute_tn_ratio(cls, df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        if "T" in x.columns and "N" in x.columns:
            x["T"] = pd.to_numeric(x["T"], errors="coerce")
            x["N"] = pd.to_numeric(x["N"], errors="coerce")
            x["tn_ratio"] = x["T"] / x["N"]
        else:
            raise ValueError(
                "Input data must contain 'T' and 'N' columns to compute T/N ratio."
            )
        return x

    @classmethod
    def _select_target_dgp(cls, df: pd.DataFrame) -> str:
        if "dgp" not in df.columns:
            return ""
        dgps = df["dgp"].astype(str).unique()
        for cand in ["garch_factor", "planted_var", "gaussian"]:
            if cand in dgps:
                return cand
        return dgps[0] if len(dgps) > 0 else ""

    @classmethod
    def create_figure1_network_recovery(
        cls, report_df: pd.DataFrame, out_path: str
    ) -> None:
        """Figure 1: LASSO-TE vs OLS-TE Sparse VAR Network Recovery"""
        df = cls._compute_tn_ratio(report_df)

        need_cols = [
            "tn_ratio",
            "f1_mean",
            "f1_std",
            "precision_mean",
            "precision_std",
            "recall_mean",
            "recall_std",
        ]
        for col in need_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        target_dgp = cls._select_target_dgp(df)
        if target_dgp:
            df = df[df["dgp"].astype(str) == target_dgp]

        df = df.dropna(subset=["tn_ratio"])
        if df.empty:
            return

        fig = Figure()
        PaperAesthetics.apply_to_figure(fig, figsize=(15, 5))
        axes = fig.subplots(1, 3)

        titles = ["A. F1 Score vs T/N", "B. Precision vs T/N", "C. Recall vs T/N"]
        metrics = ["f1", "precision", "recall"]
        color_map = {"ols": "#ff7f0e", "lasso": "#1f77b4"}

        for ax, metric, title in zip(axes, metrics, titles):
            PaperAesthetics.apply_to_axes(ax)
            te_names = (
                df["te_name"].astype(str).unique()
                if "te_name" in df.columns
                else ["ols"]
            )

            for te_name in sorted(te_names):
                sub_df = df[df["te_name"].astype(str) == te_name]
                if sub_df.empty:
                    continue

                numeric_cols = [f"{metric}_mean", f"{metric}_std"]
                agg_df = sub_df.groupby("tn_ratio")[numeric_cols].mean().reset_index()
                if agg_df.empty:
                    continue

                x_vals = agg_df["tn_ratio"].values
                mu = agg_df[f"{metric}_mean"].values
                sigma = agg_df[f"{metric}_std"].values

                c = color_map.get(te_name.lower(), "#2ca02c")
                label = f"{te_name.upper()}-TE"

                ax.plot(
                    x_vals,
                    mu,
                    marker="o",
                    linewidth=2,
                    color=c,
                    markersize=6,
                    label=label,
                )
                ax.fill_between(
                    x_vals,
                    np.clip(mu - sigma, 0, 1),
                    np.clip(mu + sigma, 0, 1),
                    color=c,
                    alpha=0.2,
                    edgecolor="none",
                )

            ax.set_title(title)
            ax.set_xlabel("T/N Ratio")
            ax.set_ylabel(metric.capitalize())
            ax.set_ylim(-0.05, 1.05)
            # 只有当图里有内容时才调用 legend，防止警告
            if ax.get_legend_handles_labels()[0]:
                ax.legend(
                    loc="upper left" if metric != "recall" else "upper right",
                    frameon=False,
                )

        fig.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        FigureCanvasAgg(fig).print_figure(out_path, dpi=300, bbox_inches="tight")

    @classmethod
    def create_figure2_mechanism_decomposition(
        cls, report_df: pd.DataFrame, out_path: str
    ) -> None:
        """Figure 2: Mechanism Decomposition: Which Component Drives Precision Failure?"""
        df = cls._compute_tn_ratio(report_df)
        if "precision_mean" not in df.columns or "dgp" not in df.columns:
            return

        # 将 NaN 填充为 0
        df["precision_mean"] = pd.to_numeric(
            df["precision_mean"], errors="coerce"
        ).fillna(0.0)
        df = df.dropna(subset=["tn_ratio"])

        if "te_name" in df.columns:
            df = df[df["te_name"].astype(str) == "ols"]
        if df.empty:
            return

        fig = Figure()
        PaperAesthetics.apply_to_figure(fig, figsize=(8, 5))
        ax = fig.subplots(1, 1)
        PaperAesthetics.apply_to_axes(ax)

        dgp_map = {
            "gaussian": ("Gaussian", "#1f77b4"),
            "garch_only": ("GARCH only", "#ff7f0e"),
            "factor_only": ("Factor only", "#2ca02c"),
            "garch_factor": ("GARCH+Factor", "#d62728"),
            "planted_var": ("Planted VAR", "#9467bd"),
        }

        found_any = False
        for dgp_key, (label, color) in dgp_map.items():
            sub_df = df[df["dgp"].astype(str) == dgp_key]
            if sub_df.empty:
                continue

            agg_df = sub_df.groupby("tn_ratio")[["precision_mean"]].mean().reset_index()
            ax.plot(
                agg_df["tn_ratio"],
                agg_df["precision_mean"],
                marker="o",
                linewidth=2,
                color=color,
                label=label,
            )
            found_any = True

        if not found_any:
            agg_df = df.groupby("tn_ratio")[["precision_mean"]].mean().reset_index()
            ax.plot(
                agg_df["tn_ratio"],
                agg_df["precision_mean"],
                marker="o",
                linewidth=2,
                color="black",
                label="All DGPs",
            )

        ax.axvline(
            x=5.0, color="gray", linestyle="--", alpha=0.7, label="T/N=5 Barrier"
        )

        ax.set_title(
            "Mechanism Decomposition: Which Component Drives Precision Failure?"
        )
        ax.set_xlabel("T/N Ratio")
        ax.set_ylabel("OLS-TE Precision")
        ax.set_ylim(-0.05, 1.05)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", frameon=False)

        fig.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        FigureCanvasAgg(fig).print_figure(out_path, dpi=300, bbox_inches="tight")

    @classmethod
    def create_figure4_tn_barrier(
        cls, report_signal_df: pd.DataFrame, out_path: str
    ) -> None:
        """Figure 4: The T/N Barrier - t-statistic vs Estimation Quality"""
        df = cls._compute_tn_ratio(report_signal_df)
        if "tstat_mean" not in df.columns:
            return

        for col in ["tstat_mean", "tstat_std", "tn_ratio"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df = df.dropna(subset=["tn_ratio"])
        if df.empty:
            return

        fig = Figure()
        PaperAesthetics.apply_to_figure(fig, figsize=(8, 5))
        ax = fig.subplots(1, 1)
        PaperAesthetics.apply_to_axes(ax)

        color_map = {
            "tstat_estimated": ("#1f77b4", "Estimated Signal"),
            "tstat_oracle": ("#ff7f0e", "Oracle Signal (True Network)"),
        }
        subsets = (
            df["subset"].astype(str).unique()
            if "subset" in df.columns
            else ["tstat_estimated"]
        )

        for sub in subsets:
            if sub not in color_map:
                continue
            color, label = color_map[sub]
            sub_df = df[df["subset"].astype(str) == sub]
            if sub_df.empty:
                continue

            agg_df = (
                sub_df.groupby("tn_ratio")[["tstat_mean", "tstat_std"]]
                .mean()
                .reset_index()
                .sort_values("tn_ratio")
            )
            x_vals = agg_df["tn_ratio"].values
            mu = agg_df["tstat_mean"].values
            sigma = agg_df["tstat_std"].values

            ax.plot(
                x_vals,
                mu,
                marker="o",
                color=color,
                linewidth=2,
                markersize=6,
                label=label,
            )
            ax.fill_between(x_vals, mu - sigma, mu + sigma, color=color, alpha=0.2)

        ax.axhline(0, color="black", linewidth=1)
        ax.axhline(1.96, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(-1.96, color="red", linestyle="--", linewidth=1, alpha=0.7)

        max_x = max(df["tn_ratio"].max() + 2, 16) if not df.empty else 16
        ax.axvspan(8, max_x, color="green", alpha=0.1, label="T/N ≥ 8 Zone")

        ax.set_title("The T/N Barrier: Signal Significance vs T/N Ratio")
        ax.set_xlabel("T/N Ratio")
        ax.set_ylabel("Mean t-statistic")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", frameon=False)

        fig.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        FigureCanvasAgg(fig).print_figure(out_path, dpi=300, bbox_inches="tight")

    @classmethod
    def create_figure5_nonparametric(
        cls, report_df: pd.DataFrame, out_path: str
    ) -> None:
        """Figure 5: Nonparametric vs Linear TE Precision"""
        df = cls._compute_tn_ratio(report_df)
        if "precision_mean" not in df.columns or "te_name" not in df.columns:
            return

        df["precision_mean"] = pd.to_numeric(
            df["precision_mean"], errors="coerce"
        ).fillna(0.0)
        df["precision_std"] = pd.to_numeric(
            df.get("precision_std", 0), errors="coerce"
        ).fillna(0.0)
        df = df.dropna(subset=["tn_ratio"])

        target_dgp = cls._select_target_dgp(df)
        if target_dgp:
            df = df[df["dgp"].astype(str) == target_dgp]

        te_types = df["te_name"].astype(str).unique()

        fig = Figure()
        PaperAesthetics.apply_to_figure(fig, figsize=(8, 5))
        ax = fig.subplots(1, 1)
        PaperAesthetics.apply_to_axes(ax)

        target_tns = [0.6, 2.5, 5.0, 10.0]
        available_tns = np.sort(df["tn_ratio"].unique())
        selected_tns = []
        for t in target_tns:
            if len(available_tns) == 0:
                break
            idx = np.abs(available_tns - t).argmin()
            selected_tns.append(available_tns[idx])
        selected_tns = sorted(list(set(selected_tns)))

        if not selected_tns:
            return

        bar_width = 0.35
        x_indices = np.arange(len(selected_tns))

        lasso_means, lasso_errs = [], []
        knn_means, knn_errs = [], []

        for tn in selected_tns:
            sub = df[np.isclose(df["tn_ratio"], tn)]

            lasso_sub = sub[sub["te_name"].astype(str) == "lasso"]
            if not lasso_sub.empty:
                lasso_means.append(lasso_sub["precision_mean"].mean())
                lasso_errs.append(lasso_sub["precision_std"].mean())
            else:
                lasso_means.append(0)
                lasso_errs.append(0)

            knn_sub = sub[sub["te_name"].astype(str) == "knn"]
            if not knn_sub.empty:
                knn_means.append(knn_sub["precision_mean"].mean())
                knn_errs.append(knn_sub["precision_std"].mean())
            else:
                knn_means.append(0)
                knn_errs.append(0)

        if "lasso" in te_types:
            ax.bar(
                x_indices - bar_width / 2,
                lasso_means,
                bar_width,
                yerr=lasso_errs,
                capsize=5,
                label="LASSO-TE",
                color="#1f77b4",
                edgecolor="black",
            )
        if "knn" in te_types:
            ax.bar(
                x_indices + bar_width / 2,
                knn_means,
                bar_width,
                yerr=knn_errs,
                capsize=5,
                label="KNN (Nonparametric)",
                color="#ff7f0e",
                edgecolor="black",
            )

        ax.axhline(
            0.05, color="red", linestyle="--", alpha=0.7, label="Random Baseline (5%)"
        )

        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"{t:.1f}" for t in selected_tns])
        ax.set_title("Nonparametric vs Linear TE: Precision Comparison")
        ax.set_xlabel("T/N Ratio")
        ax.set_ylabel("Precision")
        ax.set_ylim(0, 1.0)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", frameon=False)

        fig.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        FigureCanvasAgg(fig).print_figure(out_path, dpi=300, bbox_inches="tight")


def run_qf_81_paper_figures(
    *,
    input_dir_report_figures: str,
    input_dir_report_signal: str,
    output_root: str,
    src_dir: str,
    config_path: str,
    script_path: str,
    component: str,
) -> Qf81PaperFiguresOut:
    input_dir_report_figures = os.path.abspath(input_dir_report_figures)
    input_dir_report_signal = os.path.abspath(input_dir_report_signal)
    output_root_abs = os.path.abspath(output_root)
    cfg_path = os.path.abspath(config_path)
    script_path_abs = os.path.abspath(script_path)

    repo_root = Path(script_path_abs).parent.parent.parent.parent
    env_path = str(repo_root / "uv.lock")

    params: dict[str, Any] = {
        "input_dir_report_figures": input_dir_report_figures,
        "input_dir_report_signal": input_dir_report_signal,
        "output_root": output_root_abs,
        "config_path": cfg_path,
        "component": component,
    }

    meta = build_meta(
        params=params, env=env_path, script=script_path_abs, cfg=cfg_path, src=src_dir
    )
    run_dir = build_version_dir(output_root_abs, meta)
    audit = Audit.create(run_dir, meta)
    logger = Logger(sinks=[ConsoleSink(), audit])

    try:
        logger.info(jline("stage", component, "start", run_dir=run_dir))

        report_fig_csv = os.path.join(input_dir_report_figures, "report.csv")
        report_sig_csv = os.path.join(input_dir_report_signal, "report.csv")

        # 完美复用你的 utils，杜绝任何外部依赖导致的报错
        df_fig = _df_from_csv(report_fig_csv)
        df_sig = _df_from_csv(report_sig_csv)

        fig_dir = os.path.join(run_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        fig1_path = os.path.join(fig_dir, "figure_1_network_recovery.png")
        BespokeFigureBuilder.create_figure1_network_recovery(df_fig, fig1_path)
        logger.info(jline("output", component, "figure_1", path=fig1_path))

        fig2_path = os.path.join(fig_dir, "figure_2_mechanism.png")
        BespokeFigureBuilder.create_figure2_mechanism_decomposition(df_fig, fig2_path)
        logger.info(jline("output", component, "figure_2", path=fig2_path))

        fig4_path = os.path.join(fig_dir, "figure_4_tn_barrier.png")
        BespokeFigureBuilder.create_figure4_tn_barrier(df_sig, fig4_path)
        logger.info(jline("output", component, "figure_4", path=fig4_path))

        fig5_path = os.path.join(fig_dir, "figure_5_nonparametric.png")
        BespokeFigureBuilder.create_figure5_nonparametric(df_fig, fig5_path)
        logger.info(jline("output", component, "figure_5", path=fig5_path))

        summary = {
            "component": component,
            "run_dir": run_dir,
            "outputs": {
                "figure_1": fig1_path,
                "figure_2": fig2_path,
                "figure_4": fig4_path,
                "figure_5": fig5_path,
            },
        }
        summary_path = os.path.join(run_dir, "summary.json")
        write_json(summary_path, summary)

        audit.finish_success()
        return Qf81PaperFiguresOut(
            run_dir=run_dir, meta=meta, summary_path=summary_path
        )

    except BaseException as e:
        audit.finish_error(e)
        raise
