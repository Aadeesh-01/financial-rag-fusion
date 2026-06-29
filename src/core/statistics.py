"""
financial_rag_eval/src/evaluation/statistics.py
Calculates statistical significance (Wilcoxon) and generates publication-ready plots.
"""

import os
import logging
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

logger = logging.getLogger("StatisticsManager")

class StatisticsManager:
    """
    Handles significance testing and vector graphic generation for publications.
    """
    def __init__(self, workspace: str):
        self.workspace = os.path.abspath(workspace)
        self.stats_dir = os.path.join(self.workspace, "statistics")
        self.plots_dir = os.path.join(self.workspace, "plots")
        self.tables_dir = os.path.join(self.workspace, "tables")
        
        # Set publication-quality plot aesthetics
        sns.set_theme(style="whitegrid", context="paper")
        plt.rcParams.update({
            'font.family': 'serif',
            'figure.dpi': 300,
            'savefig.bbox': 'tight'
        })

    def run_wilcoxon_tests(self, df_metrics: pd.DataFrame, baseline_col: str, candidate_cols: List[str], metric: str = "ndcg@7") -> pd.DataFrame:
        """
        Runs Wilcoxon signed-rank tests between a baseline and candidate pipelines.
        Essential for non-normally distributed retrieval metrics.
        """
        results = []
        baseline_data = df_metrics[baseline_col].dropna()
        
        for candidate in candidate_cols:
            candidate_data = df_metrics[candidate].dropna()
            
            # Ensure paired data matches in length
            if len(baseline_data) == len(candidate_data) and len(baseline_data) > 0:
                stat, p_val = stats.wilcoxon(baseline_data, candidate_data, zero_method='wilcox', correction=False)
                effect_size = stat / (len(baseline_data) * (len(baseline_data) + 1) / 2) # Simplified rank biserial
                
                results.append({
                    "Baseline": baseline_col,
                    "Candidate": candidate,
                    "Metric": metric,
                    "W-Statistic": stat,
                    "p-value": p_val,
                    "Significant (p<0.05)": p_val < 0.05,
                    "Effect Size": effect_size
                })
            else:
                logger.warning(f"Length mismatch or empty data for paired test: {baseline_col} vs {candidate}")
                
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.stats_dir, f"wilcoxon_{metric}.csv"), index=False)
        return results_df

    def plot_latency_vs_relevance(self, summary_df: pd.DataFrame, x_metric: str = "latency_ms", y_metric: str = "ndcg@7"):
        """
        Generates a scatter plot showing the trade-off between speed and retrieval quality.
        """
        plt.figure(figsize=(8, 6))
        ax = sns.scatterplot(
            data=summary_df, 
            x=x_metric, 
            y=y_metric, 
            hue="pipeline", 
            style="pipeline", 
            s=150, 
            palette="deep"
        )
        
        # Add labels for each point
        for i in range(summary_df.shape[0]):
            plt.text(
                summary_df[x_metric].iloc[i] + (summary_df[x_metric].max()*0.02), 
                summary_df[y_metric].iloc[i], 
                summary_df["pipeline"].iloc[i], 
                horizontalalignment='left', 
                size='small', 
                color='black', 
                weight='semibold'
            )
            
        plt.title(f"Trade-off: {y_metric.upper()} vs {x_metric.replace('_', ' ').title()}")
        plt.xlabel(f"Total Inference Time ({x_metric})")
        plt.ylabel(y_metric.upper())
        plt.grid(True, linestyle='--', alpha=0.7)
        
        output_path = os.path.join(self.plots_dir, "latency_vs_relevance_tradeoff.pdf")
        plt.savefig(output_path, format="pdf")
        plt.close()
        logger.info(f"Trade-off plot saved to {output_path}")

    def generate_latex_table(self, summary_df: pd.DataFrame, filename: str = "main_results.tex"):
        """
        Exports the aggregated metrics dataframe directly to a LaTeX table format.
        """
        filepath = os.path.join(self.tables_dir, filename)
        
        latex_str = summary_df.to_latex(
            index=False, 
            float_format="%.4f", 
            caption="Comparison of Retrieval Pipelines on Financial Corpus",
            label="tab:main_results",
            column_format="l" + "c" * (len(summary_df.columns) - 1)
        )
        
        with open(filepath, "w") as f:
            f.write(latex_str)
            
        logger.info(f"LaTeX table saved to {filepath}")