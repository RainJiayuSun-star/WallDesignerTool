#!/usr/bin/env python3
"""
Visualization script for comparing segmentation model benchmark results.

This script loads benchmark CSV files for Mask2Former, OneFormer, NYU, and SegFormer
and creates comprehensive visualizations comparing all models across IoU,
Dice Coefficient, and Runtime metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
from itertools import combinations

# Set style for better-looking plots
plt.style.use(
    "seaborn-v0_8-darkgrid"
    if "seaborn-v0_8-darkgrid" in plt.style.available
    else "default"
)
plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["font.size"] = 10

# Model colors for consistent visualization
MODEL_COLORS = {
    "Mask2Former": "#3498db",  # Blue
    "OneFormer": "#e74c3c",  # Red
    "NYU": "#2ecc71",  # Green
    "SegFormer": "#f39c12",  # Orange
}

MODEL_ORDER = ["Mask2Former", "OneFormer", "NYU", "SegFormer"]


def load_benchmark_data(csv_paths):
    """Load all benchmark CSV files."""
    dataframes = {}

    for model_name, path in csv_paths.items():
        if path.exists():
            df = pd.read_csv(path)
            df["model"] = model_name
            dataframes[model_name] = df
            print(f"  Loaded {len(df)} {model_name} results")
        else:
            print(f"  Warning: {path} not found, skipping {model_name}")

    # Combine all dataframes
    if dataframes:
        df_combined = pd.concat(list(dataframes.values()), ignore_index=True)
    else:
        raise ValueError("No benchmark data files found!")

    return dataframes, df_combined


def calculate_statistics(dataframes):
    """Calculate summary statistics for all models."""
    stats = {}
    metrics = ["iou", "dice_coefficient", "runtime_sec"]

    for metric in metrics:
        stats[metric] = {}
        for model_name, df in dataframes.items():
            stats[metric][model_name] = {
                "mean": df[metric].mean(),
                "median": df[metric].median(),
                "std": df[metric].std(),
                "min": df[metric].min(),
                "max": df[metric].max(),
                "q25": df[metric].quantile(0.25),
                "q75": df[metric].quantile(0.75),
            }

    return stats


def plot_distribution_comparison(df_combined, output_dir):
    """Create distribution plots for all metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Distribution Comparison: All Models",
        fontsize=16,
        fontweight="bold",
    )

    metrics = [
        ("iou", "IoU Score", "Intersection over Union"),
        ("dice_coefficient", "Dice Coefficient", "Dice Similarity Coefficient"),
        ("runtime_sec", "Runtime (seconds)", "Inference Time"),
    ]

    models_in_data = sorted(df_combined["model"].unique())

    for idx, (metric, title, ylabel) in enumerate(metrics):
        # Histogram
        ax1 = axes[0, idx]
        for model in models_in_data:
            data = df_combined[df_combined["model"] == model][metric]
            color = MODEL_COLORS.get(model, "#95a5a6")
            ax1.hist(
                data,
                bins=30,
                alpha=0.6,
                label=model,
                edgecolor="black",
                linewidth=0.5,
                color=color,
            )
        ax1.set_xlabel(title)
        ax1.set_ylabel("Frequency")
        ax1.set_title(f"{title} Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Density plot
        ax2 = axes[1, idx]
        for model in models_in_data:
            data = df_combined[df_combined["model"] == model][metric]
            color = MODEL_COLORS.get(model, "#95a5a6")
            ax2.hist(
                data,
                bins=30,
                alpha=0.5,
                label=model,
                density=True,
                edgecolor="black",
                linewidth=0.5,
                color=color,
            )
        ax2.set_xlabel(title)
        ax2.set_ylabel("Density")
        ax2.set_title(f"{title} Density")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "distribution_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_box_plots(df_combined, output_dir):
    """Create box plots for all metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Box Plot Comparison: All Models", fontsize=16, fontweight="bold")

    metrics = [
        ("iou", "IoU Score"),
        ("dice_coefficient", "Dice Coefficient"),
        ("runtime_sec", "Runtime (seconds)"),
    ]

    models_in_data = sorted(df_combined["model"].unique())

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        data_to_plot = [
            df_combined[df_combined["model"] == model][metric].values
            for model in models_in_data
        ]

        bp = ax.boxplot(
            data_to_plot,
            tick_labels=models_in_data,
            patch_artist=True,
            showmeans=True,
        )

        # Color the boxes
        for patch, model in zip(bp["boxes"], models_in_data):
            color = MODEL_COLORS.get(model, "#95a5a6")
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel(title)
        ax.set_title(f"{title} Comparison")
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "box_plot_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_scatter_comparison(dataframes, output_dir):
    """Create scatter plots comparing all model pairs."""
    models = sorted(dataframes.keys())

    if len(models) < 2:
        print("Need at least 2 models for scatter comparison")
        return

    # Create pairwise comparisons
    model_pairs = list(combinations(models, 2))

    metrics = [
        ("iou", "IoU Score"),
        ("dice_coefficient", "Dice Coefficient"),
        ("runtime_sec", "Runtime (seconds)"),
    ]

    for metric, title in metrics:
        n_pairs = len(model_pairs)
        n_cols = min(3, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
        if n_pairs == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()

        fig.suptitle(f"{title} Pairwise Comparison", fontsize=16, fontweight="bold")

        for idx, (model1, model2) in enumerate(model_pairs):
            ax = axes[idx] if n_pairs > 1 else axes[0]

            # Merge on image_file to compare same images
            df1 = dataframes[model1][["image_file", metric]]
            df2 = dataframes[model2][["image_file", metric]]
            df_merged = pd.merge(
                df1, df2, on="image_file", suffixes=(f"_{model1}", f"_{model2}")
            )

            x = df_merged[f"{metric}_{model1}"]
            y = df_merged[f"{metric}_{model2}"]

            # Scatter plot
            ax.scatter(x, y, alpha=0.5, s=30, edgecolors="black", linewidth=0.5)

            # Diagonal line (y=x)
            min_val = min(x.min(), y.min())
            max_val = max(x.max(), y.max())
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                label="y=x (equal performance)",
                linewidth=2,
                alpha=0.7,
            )

            ax.set_xlabel(f"{model1} {title}")
            ax.set_ylabel(f"{model2} {title}")
            ax.set_title(f"{model1} vs {model2}")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add correlation coefficient
            if len(x) > 1:
                corr = np.corrcoef(x, y)[0, 1]
                ax.text(
                    0.05,
                    0.95,
                    f"Correlation: {corr:.3f}",
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

        # Hide unused subplots
        for idx in range(n_pairs, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"scatter_comparison_{metric}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close()


def plot_statistics_summary(stats, output_dir):
    """Create a bar chart comparing mean and median statistics."""
    models = sorted(list(stats["iou"].keys()))

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Summary Statistics Comparison", fontsize=16, fontweight="bold")

    metrics = [
        ("iou", "IoU Score"),
        ("dice_coefficient", "Dice Coefficient"),
        ("runtime_sec", "Runtime (seconds)"),
    ]

    stat_types = [("mean", "Mean"), ("median", "Median")]

    for row, (stat_type, stat_label) in enumerate(stat_types):
        for col, (metric, title) in enumerate(metrics):
            ax = axes[row, col]

            values = [stats[metric][model][stat_type] for model in models]
            colors = [MODEL_COLORS.get(model, "#95a5a6") for model in models]

            bars = ax.bar(
                models,
                values,
                color=colors,
                alpha=0.7,
                edgecolor="black",
                linewidth=1.5,
            )

            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{val:.4f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=9,
                )

            ax.set_ylabel(title)
            ax.set_title(f"{stat_label} {title}")
            ax.grid(True, alpha=0.3, axis="y")
            ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "statistics_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def print_statistics_table(stats):
    """Print a formatted statistics table."""
    models = sorted(list(stats["iou"].keys()))

    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)

    metrics = [
        ("iou", "IoU Score"),
        ("dice_coefficient", "Dice Coefficient"),
        ("runtime_sec", "Runtime (seconds)"),
    ]

    for metric, title in metrics:
        print(f"\n{title}:")
        print("-" * 100)

        # Header
        header = f"{'Statistic':<15}"
        for model in models:
            header += f"{model:<20}"
        print(header)
        print("-" * 100)

        stat_names = ["mean", "median", "std", "min", "max", "q25", "q75"]
        for stat_name in stat_names:
            row = f"{stat_name.capitalize():<15}"
            for model in models:
                val = stats[metric][model][stat_name]
                row += f"{val:<20.6f}"
            print(row)

    print("\n" + "=" * 100)


def main():
    """Main function to run all visualizations."""
    # Get script directory
    script_dir = Path(__file__).parent
    benchmark_output_dir = script_dir / "benchmark_output"

    # Define paths for all models
    csv_paths = {
        "Mask2Former": benchmark_output_dir
        / "benchmark_results_home_or_hotel_mask2former_preprocessed.csv",
        "OneFormer": benchmark_output_dir
        / "benchmark_results_home_or_hotel_oneformer_preprocessed.csv",
        "NYU": benchmark_output_dir / "benchmark_results_home_or_hotel_nyu_preprocessed.csv",
        "SegFormer": benchmark_output_dir
        / "benchmark_results_home_or_hotel_segformer_preprocessed.csv",
    }

    # Create output directory for visualizations
    viz_output_dir = benchmark_output_dir / "visualizations"
    viz_output_dir.mkdir(exist_ok=True)

    # Check if at least one file exists
    existing_files = {k: v for k, v in csv_paths.items() if v.exists()}
    if not existing_files:
        print("Error: No benchmark data files found!")
        print("Expected files:")
        for model, path in csv_paths.items():
            print(f"  - {path}")
        sys.exit(1)

    print("Loading benchmark data...")
    dataframes, df_combined = load_benchmark_data(existing_files)

    print(f"\nTotal records loaded: {len(df_combined)}")

    # Calculate statistics
    print("\nCalculating statistics...")
    stats = calculate_statistics(dataframes)

    # Print statistics table
    print_statistics_table(stats)

    # Create visualizations
    print("\nGenerating visualizations...")
    plot_distribution_comparison(df_combined, viz_output_dir)
    plot_box_plots(df_combined, viz_output_dir)
    plot_scatter_comparison(dataframes, viz_output_dir)
    plot_statistics_summary(stats, viz_output_dir)

    print(f"\n✓ All visualizations saved to: {viz_output_dir}")
    print("\nGenerated files:")
    print("  - distribution_comparison.png")
    print("  - box_plot_comparison.png")
    print("  - scatter_comparison_iou.png")
    print("  - scatter_comparison_dice_coefficient.png")
    print("  - scatter_comparison_runtime_sec.png")
    print("  - statistics_summary.png")


if __name__ == "__main__":
    main()
