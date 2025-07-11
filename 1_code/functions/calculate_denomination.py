
import numpy as np


def estimate_denomination(indicators, denominators):

    df_indicators = indicators.copy()
    df_denominators = denominators.copy()
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

    # Loop through each indicator and corresponding denominator, only for 'E_' prefix
    for indicator, denominator in indicators_denominators.items():
        indicator_col = f'E_{indicator}'
        denominator_col = f'E_{denominator}'

        if indicator_col in df_indicators.columns and denominator_col in df_denominators.columns:
            ratio_col_name = f'D_{indicator}'

            df_merged = df_indicators[['GEOID', indicator_col]].merge(
                df_denominators[['GEOID', denominator_col]], on='GEOID', how='left'
            )

            df_merged[ratio_col_name] = np.where(
                df_merged[denominator_col] == 0,
                0,
                df_merged[indicator_col] / df_merged[denominator_col] * 100
            )
        
            min_threshold = 1 
            max_threshold = 100

            mask_small_values = (df_merged[ratio_col_name] < min_threshold)
            df_merged.loc[mask_small_values, ratio_col_name] = np.nan
            df_merged[ratio_col_name] = df_merged[ratio_col_name].clip(upper=max_threshold)
            df_indicators = df_indicators.merge(df_merged[['GEOID', ratio_col_name]], on='GEOID', how='left')
            
            # Assert that all values are within the range of 0 to 100
            invalid_rows = df_indicators[(df_indicators[ratio_col_name] < 0) | (df_indicators[ratio_col_name] > 100)]
            assert invalid_rows.empty, f"Out of range values found in {ratio_col_name}:\n{invalid_rows[['GEOID', ratio_col_name]]}"

    print("All denominated indicators (D_ prefixed) are within the range of 0 to 100.")

    return df_indicators
