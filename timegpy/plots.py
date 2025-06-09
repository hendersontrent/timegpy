import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .evaluate_expression import evaluate_expression

def feature_hist(expr, X, y, bins=10):
    """
    Plots a histogram of the feature (from best_info_df) by class, with vertical lines at class means.
    
    Parameters:
    - expr: string representation of the time-average feature expression
    - X: 2D numpy array of shape (n_samples, n_timepoints)
    - y: array-like of class labels
    - bins: Number of histogram bins (default: 10)
    
    Returns:
    - Matplotlib figure
    """
    feature_values = evaluate_expression(expr, X)

    # Get class labels
    classes = np.unique(y)
    colors = plt.cm.get_cmap('viridis', len(classes))

    plt.figure(figsize=(10, 6))
    for idx, cls in enumerate(classes):
        cls_values = feature_values[y == cls]
        color = colors(idx)

        # Histogram
        plt.hist(cls_values, bins=bins, alpha=0.4, label=f"Class {cls}", color=color, edgecolor='black')

        # Vertical line for class mean
        cls_mean = np.mean(cls_values)
        plt.axvline(cls_mean, color=color, linestyle='--', linewidth=2, label=f"Mean Class {cls}")

    plt.title(expr)
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt

def pareto(df_all, use_parsimony=True, jitter_strength=0.1):
    """
    Plots all programs as points in a Pareto frontier scatter plot.

    Parameters:
    - df_all: DataFrame from evolve_features() containing 'program_size' and fitness columns.
    - use_parsimony: If True, plots 'fitness_parsimony'; else plots 'fitness'.
    - jitter_strength: Float, controls how much horizontal jitter is applied (default: 0.2).

    Returns:
    - Matplotlib figure object.
    """
    metric_col = 'fitness_parsimony' if use_parsimony else 'fitness'

    # Drop NaNs
    df = df_all.dropna(subset=['program_size', metric_col])

    # Jitter program size for visual clarity
    x = df['program_size'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(df))
    y = df[metric_col]

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, alpha=0.6, edgecolor='k', linewidth=0.5)
    ax.set_title("Pareto front of program size vs fitness", fontsize=14)
    ax.set_xlabel("Program size", fontsize=12)
    ax.set_ylabel("Fitness (adjusted)" if use_parsimony else "Fitness", fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    return fig

def fitness_gen(df_all, use_parsimony=True):
    """
    Plots mean fitness (or adjusted fitness) by generation with ±1 SD error bars.
    X-axis is forced to show integer ticks for generations.

    Parameters:
    - df_all: DataFrame from evolve_features(), must contain 'generation' and fitness columns.
    - use_parsimony: If True, plots 'fitness_parsimony'; else plots 'fitness'.

    Returns:
    - Matplotlib figure
    """
    metric_col = 'fitness_parsimony' if use_parsimony else 'fitness'

    # Drop NaNs
    df = df_all.dropna(subset=['generation', metric_col])

    # Group by generation
    grouped = df.groupby('generation')[metric_col]
    means = grouped.mean()
    stds = grouped.std()
    generations = means.index

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(generations, means, yerr=stds, fmt='o-', capsize=5, linewidth=2, markersize=6)

    ax.set_title("Mean fitness by generation", fontsize=14)
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Fitness (adjusted)" if use_parsimony else "Fitness", fontsize=12)
    ax.grid(True)

    # Ensure x-axis ticks are integers
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    return fig
