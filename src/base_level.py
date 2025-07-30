"""
Base level determination logic for format application.
"""
import polars as pl


def determine_base_level(univariate_table, weight_col, non_base_labels=None):
    """
    Determine the default base level from univariate table.
    
    Args:
        univariate_table (pl.DataFrame): Table with 'label' column and weight columns
        weight_col (str): Name of the weight column to use for selection
        non_base_labels (list): Labels to exclude from base level selection
        
    Returns:
        str or None: Selected base level label, or None if no eligible levels
    """
    if non_base_labels is None:
        non_base_labels = ["Missing", "Other", "NP"]
    
    # Filter out non-eligible labels
    eligible_levels = univariate_table.filter(~pl.col('label').is_in(non_base_labels))
    
    if len(eligible_levels) > 0:
        # Return the level with the highest weight
        return eligible_levels.sort(weight_col, descending=True).row(0, named=True)['label']
    else:
        return None


def get_base_level_from_dict(formats_dict, selected_fac, univariate_table, weight_col):
    """
    Get base level from formats_dict, falling back to calculated default if needed.
    
    Args:
        formats_dict (pl.DataFrame): Formats dictionary dataframe
        selected_fac (str): Selected factor name
        univariate_table (pl.DataFrame): Table with factor levels
        weight_col (str): Weight column name
        
    Returns:
        str or None: Base level to use
    """
    # Try to get base level from dictionary
    factor_row = formats_dict.filter(pl.col('factor') == selected_fac)
    
    if len(factor_row) > 0:
        stored_base_level = factor_row.row(0, named=True).get('base_level', '')
        
        # Check if stored base level exists in current levels
        if stored_base_level and stored_base_level in univariate_table['label'].to_list():
            return stored_base_level
    
    # Fall back to calculated default
    return determine_base_level(univariate_table, weight_col)