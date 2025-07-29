"""
Base level determination logic for format application.
"""
import pandas as pd
import numpy as np


def determine_base_level(univariate_table, weight_col, non_base_labels=None):
    """
    Determine the default base level from univariate table.
    
    Args:
        univariate_table (pd.DataFrame): Table with 'label' column and weight columns
        weight_col (str): Name of the weight column to use for selection
        non_base_labels (list): Labels to exclude from base level selection
        
    Returns:
        str or None: Selected base level label, or None if no eligible levels
    """
    if non_base_labels is None:
        non_base_labels = ["Missing", "Other", "NP"]
    
    # Filter out non-eligible labels
    eligible_levels = univariate_table[~univariate_table['label'].isin(non_base_labels)]
    
    if not eligible_levels.empty:
        # Return the level with the highest weight
        return eligible_levels.loc[eligible_levels[weight_col].idxmax()]['label']
    else:
        return None


def get_base_level_from_dict(formats_dict, selected_fac, univariate_table, weight_col):
    """
    Get base level from formats_dict, falling back to calculated default if needed.
    
    Args:
        formats_dict (pd.DataFrame): Formats dictionary dataframe
        selected_fac (str): Selected factor name
        univariate_table (pd.DataFrame): Table with factor levels
        weight_col (str): Weight column name
        
    Returns:
        str or None: Base level to use
    """
    # Try to get base level from dictionary
    factor_row = formats_dict[formats_dict.factor == selected_fac]
    
    if not factor_row.empty:
        stored_base_level = factor_row.iloc[0].get('base_level', '')
        
        # Check if stored base level exists in current levels
        if stored_base_level and stored_base_level in univariate_table['label'].values:
            return stored_base_level
    
    # Fall back to calculated default
    return determine_base_level(univariate_table, weight_col)