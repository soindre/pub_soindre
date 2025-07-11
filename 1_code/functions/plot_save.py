def plot_sensitivity(SAresults):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import matplotlib as mpl
    from IPython.display import display

    mpl.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['axes.labelsize'] = 13
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['legend.title_fontsize'] = 13

    if not isinstance(SAresults, dict):
        raise ValueError("SAresults should be a dictionary")

    Sdf = SAresults['Sensitivity']
    if Sdf is None:
        raise ValueError("Sensitivity indices not found. Did you run get_sensitivity with SA_type = 'SA'?")

    if 'Si' not in Sdf.columns or 'STi' not in Sdf.columns:
        raise KeyError("Both 'Si' and 'STi' columns must be present in the Sensitivity DataFrame")

    # Calculate effects
    Sdf.rename(columns={'Si': 'MainEffect'}, inplace=True)
    Sdf['Interactions'] = (Sdf['STi'] - Sdf['MainEffect']).clip(lower=0)
    Sdf[['MainEffect', 'Interactions']] = Sdf[['MainEffect', 'Interactions']].clip(upper=1)
    Sdf['Total'] = Sdf['MainEffect'] + Sdf['Interactions']

    # Rename for plotting
    name_map = {
        'normalization': 'Normalization',
        'aggregation': 'Aggregation',
        'moe': 'Margin of Error',
        'weights': 'Dimension Weights'
    }
    Sdf['Variable'] = Sdf['Variable'].map(name_map).fillna(Sdf['Variable'])

    # Show numeric table
    table_data = Sdf[['Variable', 'MainEffect', 'Interactions', 'Total']].set_index('Variable').round(2)
    display(table_data)

    bardf = Sdf.melt(id_vars=['Variable'], value_vars=['MainEffect', 'Interactions'],
                     var_name='EffectType', value_name='Value')

    sns.set_palette("colorblind")
    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca()

    bar_width = 0.8
    colors = sns.color_palette("colorblind", 2)
    bottom = None

    # Draw stacked bars
    for i, effect_type in enumerate(['MainEffect', 'Interactions']):
        subset = bardf[bardf['EffectType'] == effect_type]
        label = 'Main Effect' if effect_type == 'MainEffect' else 'Interaction Effect'
        if bottom is None:
            bars = ax.bar(subset['Variable'], subset['Value'], width=bar_width, color=colors[i], label=label)
            bottom = subset['Value'].values
        else:
            bars = ax.bar(subset['Variable'], subset['Value'], width=bar_width, bottom=bottom, color=colors[i], label=label)
            bottom += subset['Value'].values

    # Axes labels and limits
    ax.set_xlabel('Sensitivity Parameter', fontsize=13)
    ax.set_ylabel('Sensitivity Index', fontsize=13)
    ax.set_ylim(0, min(1.1, Sdf['Total'].max() + 0.1))

    # Ticks
    ax.set_xticks(range(len(Sdf['Variable'])))
    ax.set_xticklabels(Sdf['Variable'], rotation=45, ha='right', fontsize=12)

    # Add legend with box
    legend = ax.legend(title="Effect Type", loc='upper right', frameon=True)
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_alpha(0.9)

    # Final layout
    fig.tight_layout()

    # Print totals
    print(f"Total Main Effect: {Sdf['MainEffect'].sum():.3f}")
    print(f"Total Interactions: {Sdf['Interactions'].sum():.3f}")

    return fig