import numpy as np
import pandas as pd

def construct_nominal_index(df_variables, normalization, index_normalization, aggregation, weight_func, weights):
    df = df_variables.copy()

    d_columns = df.columns[df.columns.str.startswith('D_')]

    min_threshold = 1.5 
    
    if normalization == 'minmax':
        for col in d_columns:
            df[f'N{col}'] = 1 + ((df[col] - df[col].min()) / (df[col].max() - df[col].min())) * 99
            if aggregation == 'geometric':
                df.loc[df[f'N{col}'] < min_threshold, f'N{col}'] = np.nan
                
    elif normalization == 'pct':
        for col in d_columns:
            df[f'N{col}'] = df[col].rank(pct=True) * 100
            if aggregation == 'geometric':
                df.loc[df[f'N{col}'] < min_threshold, f'N{col}'] = np.nan
    
    themes = {
        'THEME1': ['ND_HOCOB', 'ND_POVTY', 'ND_NOHIG', 'ND_NOHEA'],
        'THEME2': ['ND_AGE65', 'ND_AGE17', 'ND_DISBL', 'ND_SNGPH', 'ND_LANGU'],
        'THEME3': ['ND_MINRTY'],
        'THEME4': ['ND_MUUNS', 'ND_MOHOM', 'ND_CROWD', 'ND_NOVEH']
    }
    
    if weights is not None:
        theme_weights = weights.copy()
    
    for theme, components in themes.items():
        if aggregation == 'arithmetic':
            df[theme] = df[components].sum(axis=1) / len(components)

        elif aggregation == 'multiplicative':
            df[theme] = df[components].apply(
                lambda row: np.prod([
                    row[col] if row[col] == 1 else row[col] * (1/len(components))
                    for col in components
                    if pd.notna(row[col])
                ]) if any(pd.notna(row[col]) for col in components) else np.nan,
                axis=1
            )

        elif aggregation == 'geometric':
            df[theme] = df[components].apply(
                lambda row: np.power(
                    np.prod([
                        row[col] 
                        for col in components
                        if pd.notna(row[col])
                    ]),
                    1/sum(pd.notna(row[components]))
                ) if any(pd.notna(row[col]) for col in components) else np.nan,
                axis=1
            )

        else:
            print("Aggregation method not implemented")
        
        if index_normalization == True:
            if normalization == 'minmax':
                df[theme] = 1 + ((df[theme] - df[theme].min()) / (df[theme].max() - df[theme].min())) * 99
            elif normalization == 'pct':
                df[theme] = df[theme].rank(pct=True) * 100
    
    if aggregation == 'arithmetic':
        if weight_func == 'equal':
            df['S_NOMINAL'] = df[list(themes.keys())].sum(axis=1) / len(themes)
        else:
            df['S_NOMINAL'] = df[list(themes.keys())].apply(
                lambda row: sum(row[theme] * theme_weights.get(theme, 1) for theme in themes) /
                sum(theme_weights.get(theme, 1) for theme in themes),
                axis=1
            )
    elif aggregation == 'multiplicative':
        if weight_func == 'equal':
            df['S_NOMINAL'] = df[list(themes.keys())].apply(
                lambda row: np.prod([
                    row[theme] if row[theme] == 1 else row[theme] * (1/len(themes))
                    for theme in themes
                    if pd.notna(row[theme])
                ]) if any(pd.notna(row[theme]) for theme in themes) else np.nan,
                axis=1
            )
        else:
            sum_weights = sum(theme_weights.get(theme, 1) for theme in themes)
            df['S_NOMINAL'] = df[list(themes.keys())].apply(
                lambda row: np.prod([
                    row[theme] if row[theme] == 1 else row[theme] * (theme_weights.get(theme, 1)/sum_weights)
                    for theme in themes
                    if pd.notna(row[theme])
                ]) if any(pd.notna(row[theme]) for theme in themes) else np.nan,
                axis=1
        )
    elif aggregation == 'geometric':
        if weight_func == 'equal':
            df['S_NOMINAL'] = df[list(themes.keys())].apply(
                lambda row: np.exp(
                    np.mean([
                        np.log(max(row[theme], 1e-10))
                        for theme in themes
                        if pd.notna(row[theme])
                    ])
                ) if any(pd.notna(row[theme]) for theme in themes) else np.nan,
                axis=1
            )
        else:
            sum_weights = sum(theme_weights.get(theme, 1) for theme in themes)
            df['S_NOMINAL'] = df[list(themes.keys())].apply(
                lambda row: np.exp(
                    sum([
                        np.log(max(row[theme], 1e-10)) * (theme_weights.get(theme, 1)/sum_weights)
                        for theme in themes
                        if pd.notna(row[theme])
                    ]) / sum([
                        theme_weights.get(theme, 1)/sum_weights
                        for theme in themes
                        if pd.notna(row[theme])
                    ])
                ) if any(pd.notna(row[theme]) for theme in themes) else np.nan,
                axis=1
            )
    else:
        print("Blubp - tell me something new")

    if index_normalization:
        if normalization == 'minmax':
            df['S_NOMINAL'] = 1 + ((df['S_NOMINAL'] - df['S_NOMINAL'].min()) /
                                (df['S_NOMINAL'].max() - df['S_NOMINAL'].min())) * 99
        elif normalization == 'pct':
            df['S_NOMINAL'] = df['S_NOMINAL'].rank(pct=True) * 100
 
    df['RS_NOMINAL'] = df['S_NOMINAL'].rank(ascending=False)


    return df