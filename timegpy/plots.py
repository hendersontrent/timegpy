import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .evaluate_expression import evaluate_expression

def plot_hist(expr, X, y, bins=10):
    """
    Plots a histogram of a feature by class, with vertical lines at class means.
    
    Parameters:
    - expr: string representation of the time-average feature expression
    - X: 2D numpy array of shape (n_samples, n_timepoints)
    - y: array-like of class labels
    - bins: Number of histogram bins (default: 10)
    
    Returns:
    - Matplotlib figure.
    """

    #-------- Prepare data --------

    feature_values = evaluate_expression(expr, X)

    # Get class labels

    classes = np.unique(y)
    colors = plt.cm.get_cmap('viridis', len(classes))

    #-------- Draw plot --------

    plt.figure(figsize=(10, 6))

    for idx, cls in enumerate(classes):
        cls_values = feature_values[y == cls]
        color = colors(idx)

        # Add histogram

        plt.hist(cls_values, bins=bins, alpha=0.5, label=f"Class {cls}", color=color, edgecolor='black')

        # Add vertical lines for class mean

        cls_mean = np.mean(cls_values)
        plt.axvline(cls_mean, color=color, linestyle='--', linewidth=2, label=f"Mean Class {cls}")

    plt.title(expr)
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt

def plot_pareto(df_all, df_best, use_parsimony=True, level=0.95):
    """
    Plots a ribbon of fitness percentiles and mean fitness per program size, with the best feature highlighted.

    Parameters:
    - df_all: DataFrame from tsgp() containing 'program_size' and fitness columns.
    - df_best: DataFrame from tsgp() containing the best expression found.
    - use_parsimony: If True, plots 'fitness_parsimony'; else plots 'fitness'.
    - level: Confidence level for the ribbon (e.g., 0.95 for a 95 per cent interval).

    Returns:
    - Matplotlib figure.
    """
    metric_col = 'fitness_parsimony' if use_parsimony else 'fitness'

    #-------- Prepare data --------

    # Drop NaNs

    df = df_all.dropna(subset=['program_size', metric_col])

    # Calculate lower and upper percentiles based on the confidence level

    lower_pct = (1.0 - level) / 2 * 100
    upper_pct = (1.0 + level) / 2 * 100

    # Compute summary stats for each program size

    summary = df.groupby('program_size')[metric_col].agg(
        mean='mean',
        lower=lambda x: np.percentile(x, lower_pct),
        upper=lambda x: np.percentile(x, upper_pct)
    ).reset_index()

    #-------- Draw plot --------

    fig, ax = plt.subplots(figsize=(9, 6))

    # Add interval as a ribbon

    ax.fill_between(summary['program_size'], summary['lower'], summary['upper'],
                    alpha=0.3, label=f'{int(level * 100)}% Interval', color='skyblue')

    # Add mean line

    ax.plot(summary['program_size'], summary['mean'], label='Mean Fitness', color='blue', linewidth=2)

    # Add a point to signify best individual feature

    best_size = df_best['program_size'].iloc[0]
    best_fitness = df_best[metric_col].iloc[0]
    ax.scatter([best_size], [best_fitness], color='red', s=80, marker='o', label='Best Feature', zorder=5)

    ax.set_title("Pareto front: Program size vs fitness", fontsize=14)
    ax.set_xlabel("Program size", fontsize=12)
    ax.set_ylabel("Fitness (adjusted)" if use_parsimony else "Fitness", fontsize=12)
    ax.grid(True)
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    return fig

def plot_gen(df_all, use_parsimony=True):
    """
    Plots mean fitness (or adjusted fitness) by generation with ±1 SD error bars.
    X-axis is forced to show integer ticks for generations.

    Parameters:
    - df_all: DataFrame from evolve_features(), must contain 'generation' and fitness columns.
    - use_parsimony: If True, plots 'fitness_parsimony'; else plots 'fitness'.

    Returns:
    - Matplotlib figure.
    """
    metric_col = 'fitness_parsimony' if use_parsimony else 'fitness'

    #-------- Prepare data --------

    # Drop NaNs

    df = df_all.dropna(subset=['generation', metric_col])

    # Group by generation

    grouped = df.groupby('generation')[metric_col]
    means = grouped.mean()
    stds = grouped.std()
    generations = means.index

    #-------- Draw plot --------

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
