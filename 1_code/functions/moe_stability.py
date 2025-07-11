
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def class_stability(df_moe):
    import pandas as pd
    import numpy as np

    estimate_col = 'NOMINAL'
    indicator = 'Index_Rep'
    geoid_col = 'GEOID'  
    
    columns = [col for col in df_moe.columns if col.startswith(indicator)]
    
    squared_differences = (df_moe[columns] - df_moe[estimate_col].values.reshape(-1, 1)) ** 2
    variance = (4 / 80) * squared_differences.sum(axis=1, min_count=1)
    df_moe['MOE'] = 1.645 * np.sqrt(variance)  # 90% confidence interval

    rank_columns = [f'Rank_{col}' for col in [estimate_col] + columns]
    ranks = pd.concat([df_moe[geoid_col], pd.concat([df_moe[col].rank(method='min').rename(f'Rank_{col}') for col in [estimate_col] + columns], axis=1)], axis=1)

    quantiles = [0, 0.25, 0.5, 0.75, 1]

    quantiles_df = pd.concat([df_moe[geoid_col], pd.concat([pd.qcut(ranks[f'Rank_{col}'], quantiles, labels=False).rename(f'Quantile_{col}') 
                                for col in [estimate_col] + columns], axis=1)], axis=1)

    df_moe = pd.merge(df_moe, ranks, on=geoid_col, how='left')
    df_moe = pd.merge(df_moe, quantiles_df, on=geoid_col, how='left')

    df_moe[f'Lower_CI_{estimate_col}'] = df_moe[estimate_col] - df_moe['MOE']
    df_moe[f'Upper_CI_{estimate_col}'] = df_moe[estimate_col] + df_moe['MOE']

    # Quantile classification for the confidence intervals
    lower_upper_quantiles = pd.concat([
        pd.qcut(df_moe[f'Lower_CI_{estimate_col}'], quantiles, labels=False).rename(f'Lower_Quantile_{estimate_col}'),
        pd.qcut(df_moe[f'Upper_CI_{estimate_col}'], quantiles, labels=False).rename(f'Upper_Quantile_{estimate_col}')
    ], axis=1)

    df_moe = pd.concat([df_moe, lower_upper_quantiles], axis=1)
    stability_cols = [f'Quantile_{col}' for col in columns]  
    df_moe['Stability_NOMINAL'] = (df_moe[stability_cols] == df_moe[f'Quantile_{estimate_col}'].values.reshape(-1, 1)).all(axis=1)

    quantile_diffs = df_moe[stability_cols].sub(df_moe[f'Quantile_{estimate_col}'], axis=0)

    move_1_class = (quantile_diffs.abs() == 1).sum(axis=1) / len(columns)
    move_2_class = (quantile_diffs.abs() == 2).sum(axis=1) / len(columns)
    move_3_class = (quantile_diffs.abs() == 3).sum(axis=1) / len(columns)

    move_1_class_percentage = (move_1_class > 0).mean() * 100
    move_2_class_percentage = (move_2_class > 0).mean() * 100
    move_3_class_percentage = (move_3_class > 0).mean() * 100

    stay_in_quantile_percentage = df_moe['Stability_NOMINAL'].mean() * 100
    rank_nominal_col = f'Rank_{estimate_col}' 
    abs_rank_diffs = df_moe[rank_columns].sub(df_moe[rank_nominal_col], axis=0).abs()
    df_moe['Mean_Absolute_Difference'] = abs_rank_diffs.mean(axis=1)
    overall_mean_of_means = df_moe['Mean_Absolute_Difference'].mean()
    overall_std_of_means = df_moe['Mean_Absolute_Difference'].std()

    print(f'Mean of the mean absolute rank differences across all tracts: {overall_mean_of_means:.4f}')
    print(f'Standard deviation of the mean absolute rank differences across all tracts: {overall_std_of_means:.4f}')

    return df_moe


def plot_moe(df_index_moe, y_limit, x_limit=100):
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    # Define custom model-based colors directly
    color_blue = '#264653'   # Hierarchical (deep blue)
    color_yellow = '#E9C46A' # Inductive (warm yellow)
    color_red = '#E76F51'    # Deductive (warm red)
    
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_moe_colormap",
        [color_blue, color_yellow, color_red]
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    plot_data = df_index_moe.dropna(subset=['NOMINAL', 'MOE'])
    plot_data = plot_data[plot_data['MOE'] <= 100]
    
    if x_limit:
        plot_data = plot_data[plot_data['NOMINAL'] <= x_limit]
    
    sns.histplot(
        data=plot_data,
        x='NOMINAL',
        y='MOE',
        bins=50,  
        cbar=True,  
        cmap=custom_cmap, 
        ax=ax
    )
    
    ax.set_ylim(0, y_limit)
    if x_limit:
        ax.set_xlim(0, x_limit)
  
    ax.set_xlabel('Social Vulnerability Index-Score')
    ax.set_ylabel('MOE of Vulnerability Index-Score')
    ax.grid(True)  # Enable grid for better readability
    
    mean_value = np.mean(plot_data['MOE'])
    median_value = np.median(plot_data['MOE'])
    std_dev = np.std(plot_data['MOE'])
    
    print(f"Statistics for MOE:")
    print(f"Mean: {mean_value:.4f}")
    print(f"Median: {median_value:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")
    print("\n") 
    
    return fig

def plot_panel(df_index_moe, y_limit, ax=None):
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    plot_data = df_index_moe.dropna(subset=['NOMINAL', 'MOE'])
    plot_data = plot_data[plot_data['MOE'] <= 100]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = None 

    sns.histplot(
        data=plot_data,
        x='NOMINAL',
        y='MOE',
        bins=50,
        cbar=True,
        cmap="twilight",
        ax=ax
    )

    ax.set_ylim(0, y_limit)
    ax.set_xlabel('Social Vulnerability Index-Score')
    ax.set_ylabel('MOE of Vulnerability Index-Score')
    ax.grid(True)

    # Print stats
    mean_value = np.mean(plot_data['MOE'])
    median_value = np.median(plot_data['MOE'])
    std_dev = np.std(plot_data['MOE'])

    print(f"Statistics for MOE:")
    print(f"Mean: {mean_value:.4f}")
    print(f"Median: {median_value:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")
    print("\n")

    return fig


def plot_moe_distribution_per_bin(df_index_moe, bin_width=20, plot_type='box', y_max=None, x_max=None,
                                   quantile_bins=False, n_quantiles=None, min_n=10, save_path=None):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Your preferred colors
    color_list = ['#2A9D8F', '#264653', '#E9C46A', '#E76F51']

    # Clean and prepare data
    data = df_index_moe.dropna(subset=['NOMINAL', 'MOE']).copy()

    if x_max is not None:
        data = data[data['NOMINAL'] <= x_max]

    if quantile_bins:
        if n_quantiles is None:
            raise ValueError("You must specify `n_quantiles` when using quantile_bins=True.")

        quantiles = np.linspace(0, 1, n_quantiles + 1)
        bins = data['NOMINAL'].quantile(quantiles).unique()

        vulnerability_labels = ["Low", "Medium Low", "Medium High", "High"]
        data['bin'] = pd.cut(data['NOMINAL'], bins=bins, labels=vulnerability_labels[:n_quantiles], include_lowest=True)
        data['bin'] = pd.Categorical(data['bin'], categories=vulnerability_labels[:n_quantiles], ordered=True)

        # Filter out bins with low counts
        bin_counts = data['bin'].value_counts().sort_index()
        valid_bins = bin_counts[bin_counts >= min_n].index
        data = data[data['bin'].isin(valid_bins)]
        labeled_bins = [f"{b}\n(n={bin_counts.get(b, 0)})" for b in data['bin'].cat.categories if b in valid_bins]
        palette = dict(zip(vulnerability_labels[:n_quantiles], color_list[:n_quantiles]))

    else:
        bins = np.arange(0, 101, bin_width)
        labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
        data['bin'] = pd.cut(data['NOMINAL'], bins=bins, labels=labels, include_lowest=True)
        data['bin'] = pd.Categorical(data['bin'], categories=labels, ordered=True)

        # Filter out bins with low counts
        bin_counts = data['bin'].value_counts().sort_index()
        valid_bins = bin_counts[bin_counts >= min_n].index
        data = data[data['bin'].isin(valid_bins)]
        labeled_bins = [f"{b}\n(n={bin_counts.get(b, 0)})" for b in data['bin'].cat.categories if b in valid_bins]
        palette = dict(zip(labels, color_list * ((len(labels) // len(color_list)) + 1)))

    # Create figure and plot
    fig, ax = plt.subplots(figsize=(10, 7))

    if plot_type == 'box':
        sns.boxplot(data=data, x='bin', y='MOE', palette=palette, ax=ax, showfliers=False)
    elif plot_type == 'violin':
        sns.violinplot(data=data, x='bin', y='MOE', palette=palette, cut=0, ax=ax, inner='box', scale='width')
    else:
        raise ValueError("Invalid plot_type. Use 'box' or 'violin'.")

    ax.tick_params(axis='both', labelsize=14)
    ax.set_xticklabels(valid_bins, rotation=0, fontsize=14)

    y_max = y_max if y_max is not None else data['MOE'].max() * 0.9
    ax.set_ylim(0, y_max)

    ax.set_xlabel("Vulnerability Level", fontsize=16)
    ax.set_ylabel("Margin of Error", fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig