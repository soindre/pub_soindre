
import numpy as np
import pandas as pd


# Function to generate random weights for a list of indicators ensuring they sum to 1
def generate_random_weights(num_indicators):
    weights = np.random.rand(num_indicators)
    return weights / weights.sum()

def generate_weights(dimensions):
    
    weights_sets = []

    # Equal weights for all indicators across all themes
    equal_weights = {}
    for theme, indicators in dimensions.items():
        equal_weight = 1 / len(indicators)  
        for indicator in indicators:
            equal_weights[indicator] = equal_weight
    weights_sets.append(equal_weights)  

    # Generate 4 sets of random weights for each theme
    for i in range(4):
        random_weights = {}
        for theme, indicators in dimensions.items():
            theme_weights = generate_random_weights(len(indicators))  # Generate random weights
            for indicator, weight in zip(indicators, theme_weights):
                random_weights[indicator] = weight
        weights_sets.append(random_weights)  # Append random weights only

    
    return weights_sets


def dimension_weights():
    
    # Define the weight sets as a list of dictionaries
    weights_list = [
        {"THEME1": 0.37, "THEME2": 0.43, "THEME3": 0.18, "THEME4": 0.2},  # literature
        {"THEME1": 0.20, "THEME2": 0.20, "THEME3": 0.40, "THEME4": 0.20},  # minority
        {"THEME1": 0.25, "THEME2": 0.25, "THEME3": 0.25, "THEME4": 0.25},  # equal
        {"THEME1": 0.30, "THEME2": 0.15, "THEME3": 0.30, "THEME4": 0.15}   # empiric
    ]

    
    return weights_list