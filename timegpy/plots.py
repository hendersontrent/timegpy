import numpy as np
import matplotlib.pyplot as plt
from evaluate_expression import evaluate_expression

def plot_feature(expr, X, y, bins=10):
    """
    Plots a histogram of the feature (from best_info_df) by class.
    
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
    colors = plt.cm.get_cmap('tab10', len(classes))

    plt.figure(figsize=(10, 6))
    for idx, cls in enumerate(classes):
        cls_values = feature_values[y == cls]
        plt.hist(cls_values, bins=bins, alpha=0.6, label=f"Class {cls}", color=colors(idx), edgecolor='black')

    plt.title(expr)
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt