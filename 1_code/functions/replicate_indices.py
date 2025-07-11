import os
import pandas as pd
import numpy as np

def construct_nominal_index(df, normalization, index_normalization, aggregation, weight_func, weights):

    d_columns = df.columns[df.columns.str.startswith('D_')]

    if normalization == 'minmax':
        for col in d_columns:
            df[f'N{col}'] = 1 + ((df[col] - df[col].min()) / (df[col].max() - df[col].min())) * 99
    elif normalization == 'pct':
        for col in d_columns:
            df[f'N{col}'] = df[col].rank(pct=True) * 100


    # Define and calculate theme scores
    themes = {
        'THEME1': ['ND_HOCOB', 'ND_POVTY', 'ND_NOHIG', 'ND_NOHEA'],
        'THEME2': ['ND_AGE65', 'ND_AGE17', 'ND_DISBL', 'ND_SNGPH', 'ND_LANGU'],
        'THEME3': ['ND_MINRTY'],
        'THEME4': ['ND_MUUNS', 'ND_MOHOM', 'ND_CROWD', 'ND_NOVEH']
    }

    if weights != None:
        theme_weights = weights.copy()

    # Calculate the weighted sum for each theme
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

    # Calculate the total score from the themes
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
            
    # Rank the total score with the highest value as rank 1
    df['RS_NOMINAL'] = df['S_NOMINAL'].rank(ascending=False)

    return df

def calculate_index_80_times(df_nominal, calculated_indicator_path, denominator_table_map, indicator_table_map, indicators_denominators, normalization, index_pct, aggregation, weight_func, weights):
    
    df_nominal = df_nominal[['GEOID', 'S_NOMINAL']].copy()
    
    df_nominal.rename(columns={'S_NOMINAL': 'NOMINAL'}, inplace=True)

    # Loop through each replicate (Var_Rep1 to Var_Rep80)
    for i in range(1, 81):
        df_rep = pd.DataFrame({'GEOID': df_nominal['GEOID']}).copy()
        var_rep_1 = pd.DataFrame({'GEOID': df_nominal['GEOID']}).copy()

        df_denominators = pd.DataFrame({'GEOID': df_nominal['GEOID']}).copy()
        for denom, denom_table in denominator_table_map.items():

            path_denom = os.path.join(os.path.dirname(os.getcwd()), '0_data', 'gitignore', 'denominator_tables')
            table_file_path = os.path.join(path_denom, f'{denom}.csv')
            df_denom = pd.read_csv(table_file_path)

            var_rep_col = f'Var_Rep{i}'
            denom_col_name = f'{denom}_Rep{i}'
            df_denom.rename(columns={var_rep_col: denom_col_name}, inplace=True)

            df_denominators = df_denominators.merge(df_denom[['GEOID', denom_col_name]], on='GEOID', how='left')
            if i == 1:
                var_rep_1 = var_rep_1.merge(df_denominators, on='GEOID', how='left')

        for indicator, table_name in indicator_table_map.items():

            table_file_path = os.path.join(calculated_indicator_path, f'{indicator}.csv')
            df_indicator = pd.read_csv(table_file_path)

            var_rep_col = f'Var_Rep{i}'
            indicator_col_name = f'{indicator}_Rep{i}'
            df_indicator.rename(columns={var_rep_col: indicator_col_name}, inplace=True)
            if i == 1:
             var_rep_1 = var_rep_1.merge(df_indicator[['GEOID', indicator_col_name]], on='GEOID', how='left')

            df_indicator = df_indicator.merge(df_denominators, on='GEOID', how='left')
            
            denominator = indicators_denominators.get(indicator, None)
            if denominator is not None:
                denominator_col_name = f'{denominator}_Rep{i}'
                
                if denominator_col_name in df_indicator.columns:
                    df_indicator[f'D_{indicator}'] = np.where(
                        df_indicator[denominator_col_name] == 0,
                        np.nan,  # not 0! this is key
                        df_indicator[indicator_col_name] / df_indicator[denominator_col_name] * 100
                    )
                    original = df_indicator[f'D_{indicator}'].copy()

                    df_indicator[f'D_{indicator}'] = np.clip(df_indicator[f'D_{indicator}'], 0, 100)

                    too_low = original[original < -50]
                    too_high = original[original > 150]

                    if len(too_low) > 0 or len(too_high) > 0:
                        print(f"Diagnostics for indicator: {indicator}")
                        print(f"Indicator: {indicator}")
                        print(f"  Values < -50: {len(too_low)}")
                        print(f"  Values > 150: {len(too_high)}")
                        
                        # Get corresponding indicator and denominator values
                        denom_col = denominator_col_name
                        ind_col = indicator_col_name

                        print("  Sample of extreme values with indicator and denominator:")
                        sample_indices = pd.concat([too_low.head(5), too_high.head(5)]).index
                        print(df_indicator.loc[sample_indices, [f'D_{indicator}', ind_col, denom_col]])
                        print("-" * 40)

                else:
                    print(f'Denominator column {denominator_col_name} not found in the merged dataframe for indicator {indicator}.')
            else:
                print(f'No denominator found for indicator {indicator}.')

            df_rep = df_rep.merge(df_indicator[['GEOID', f'D_{indicator}']], on='GEOID', how='left')

        d_columns = df_rep.columns[df_rep.columns.str.startswith('D_')]
        
        df_rep = construct_nominal_index(df_rep, normalization, index_pct, aggregation, weight_func, weights)

        if i == 2:
            index_norm_label = 'indexpct' if index_pct else 'noindexpct'

            output_path = os.path.join(
                os.path.dirname(os.getcwd()),
                '0_data', 'gitignore',
                f'var_rep_2_{normalization}_{aggregation}_{index_norm_label}_validation.csv'
            )

            df_rep.to_csv(output_path, index=False)

        df_rep.rename(columns={'S_NOMINAL': f'Index_Rep{i}'}, inplace=True)
        df_nominal = df_nominal.merge(df_rep[['GEOID', f'Index_Rep{i}']], on='GEOID', how='left')
        

    return df_nominal

def construct_replicate_indices(df_nominal, normalization, index_pct, aggregation, weight_func, weights):
   
    calculated_indicator_path = os.path.join(os.path.dirname(os.getcwd()), '0_data', 'gitignore', 'calculated_indicator_tables')

    # Define the table map for denominators
    denominator_table_map = {
        'TOTHH': 'B11012',  # Total Households
        'TOTHU': 'B25024',  # Total Housing Units
        'TOTPOP': 'B01001'  # Total Population
    }

    # Define the table map for indicators
    indicator_table_map = {
        'POVTY': 'B17001',  # Poverty
        'HOCOB': 'B25070',  # Housing Cost Burden
        'NOHIG': 'B15002',  # No Highschool Diploma
        'NOHEA': 'B27001',  # No Health Insurance
        'AGE65': 'B01001',  # Age 65 and Older
        'AGE17': 'B01001',  # Age 17 and Younger
        'DISBL': 'B18101',  # Disability
        'SNGPH': 'B11012',  # Single-Parent Households
        'LANGU': 'C16001',  # Language Proficiency
        'MINRTY': 'B03002',  # Minorities (handled separately)
        'MUUNS': 'B25024',  # Multi-unit Structure
        'MOHOM': 'B25024',  # Mobile Homes
        'CROWD': 'B25014',  # Crowding
        'NOVEH': 'B25044'   # No Vehicle
    }

    # Define the list of indicators and corresponding denominators
    indicators_denominators = {
        'POVTY': 'TOTPOP',  # Poverty / Population
        'HOCOB': 'TOTHU',   # Housing Cost Burden / Housing Units
        'NOHIG': 'TOTPOP',  # No Highschool / Population
        'NOHEA': 'TOTPOP',  # No Health Insurance / Population
        'AGE65': 'TOTPOP',  # 65+ / Population
        'AGE17': 'TOTPOP',  # 17- / Population
        'DISBL': 'TOTPOP',  # Disability / Population
        'SNGPH': 'TOTHH',   # Single Parent Household / Universe Households
        'LANGU': 'TOTPOP',  # Language / Population
        'MINRTY': 'TOTPOP', # Minority / Population (allow larger than total population)
        'MUUNS': 'TOTHU',   # Multi Unit Structure / Housing Units
        'MOHOM': 'TOTHU',   # Mobile Homes / Universe Housing Units
        'CROWD': 'TOTHU',   # Crowding / Housing Units
        'NOVEH': 'TOTHU'    # No Vehicle / Universe Occupied housing units
    }

    # Call the function to calculate the index 80 times
    df_final_nominal = calculate_index_80_times(df_nominal, calculated_indicator_path, denominator_table_map, indicator_table_map, indicators_denominators, normalization, index_pct, aggregation, weight_func, weights)

    return df_final_nominal

