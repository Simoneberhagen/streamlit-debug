import pandas as pd
import numpy as np
import scipy.stats as ss
import gpc_utils.emblem as eu
import gpc_utils.sas as su



def cap_format(format, cap=np.nan, floor=np.nan):

    # find format name
    fmt_name = format.FMTNAME.unique()
    assert(len(fmt_name)==1)
    fmt_name = fmt_name[0]

    new_rows = []

    for _, row in format.iterrows():

        if row["LABEL"] in ["Missing", "Other"]:
            # If the row's start is less than or equal to x, keep it
            new_rows.append(row)

        else:
            # format START and END as floats to compare them with cap and floor
            row = row.replace("low", -np.infty).replace("high", np.infty)
            row["START"] = float(row["START"])
            row["END"] = float(row["END"])

            # Check if cap is within the interval considering inclusivity/exclusivity
            if cap == np.nan:
                cap_level = False
            else:
                cap_start_condition = (row['START'] < cap) if row['SEXCL'] == 'Y' else (row['START'] <= cap)
                cap_end_condition = (cap < row['END']) if row['EEXCL'] == 'Y' else (cap <= row['END'])
                cap_level = cap_start_condition & cap_end_condition

            # Check if floor is within the interval considering inclusivity/exclusivity
            if floor == np.nan:
                floor_level = False
            else:
                floor_start_condition = (row['START'] < floor) if row['SEXCL'] == 'Y' else (row['START'] <= floor)
                floor_end_condition = (floor < row['END']) if row['EEXCL'] == 'Y' else (floor <= row['END'])
                floor_level = floor_start_condition & floor_end_condition

            # Check if both floor and cap are within the interval
            if floor_level and cap_level:
                new_row = row.copy()
                new_row['START'] = floor
                new_row['END'] = cap
                new_row['SEXCL'] = 'N'
                new_row['EEXCL'] = 'N'
                new_row["LABEL"] = f"[{floor}; {cap}]"
                new_row["HLO"] = "S"
                new_rows.append(new_row)

            # Ignores levels lower than floor
            elif row["END"] < floor:
                continue

            # If floor is included in the level, cut the level keeping only values >= floor
            elif floor_level:
                new_row = row.copy()
                new_row['START'] = floor
                new_row['SEXCL'] = 'N'  # Include floor
                new_row["LABEL"] = f"[{floor}; {new_row['END']:.2f}]"
                new_row["HLO"] = "S"
                new_rows.append(new_row)

            # If cap is included in the level, cut the level keeping only values <= cap
            elif cap_level:  
                new_row = row.copy()
                new_row['END'] = cap
                new_row['EEXCL'] = 'N'  # Include cap
                new_row["LABEL"] = f"({new_row['START']:.2f}; {cap}]"
                new_row["HLO"] = "S"
                new_rows.append(new_row)

            # Ignores levels higher than cap
            elif row["START"] > cap:
                continue
            
            # Append all the other levels without modifying them
            else:
                new_rows.append(row)
            
    format["START"].replace(float(-np.infty), "low")
    format["END"].replace(float(np.infty), "high")
    
    return pd.DataFrame(new_rows)



def add_missing_values(format, missing_values=[], label="Missing"):
    """
    
    """
    # find format name
    fmt_name = format.FMTNAME.unique()
    assert(len(fmt_name)==1)
    fmt_name = fmt_name[0]

    # set type
    var_type = format.TYPE.unique()
    assert(len(var_type)==1)
    var_type = var_type[0]

    # add missing rules
    for mv in missing_values:
        missing_row = pd.Series(data=[mv, mv, fmt_name, var_type, label, "S", "", ""], index=format.columns)
        format = pd.concat([format, missing_row.to_frame().T])

    return format



def define_format(df, var, weight, distribution, cap=np.nan, floor=np.nan, highest=np.nan, lowest=np.nan, 
                  missing_values=[], np_values=[], num_bins=np.nan, formatter="integer", *args, **kwars):

    # subset the df keeping only values meaningful for the format definition
    filter_mask = (df[var] < floor) | (df[var] > cap) | (df[var].isin(missing_values)) | (df[var].isin(np_values))
    df_subset = df[~filter_mask]
    # cut the values higher and lower than the extremes to generate levels only inside this interval 
    df_subset.loc[df[var] > highest, var] = highest
    df_subset.loc[df[var] < lowest, var] = lowest

    # Set number of bins
    if np.isnan(num_bins):
        unique_num = len(df_subset[var].unique())
        num_bins = np.min([33, unique_num]) -1

    # set distribution type
    dist = None
    if distribution == "uniform":
        dist = ss.uniform()
    elif distribution == "normal":
        dist = ss.norm(0,1)
    elif distribution == "discrete":
        # For discrete, we can allow the discretizer to handle it, often by frequency.
        pass
    else:
        raise ValueError(f"Distribution {distribution} not supported")
    
    # set label format
    if formatter == "integer":
        label_format = "{a:.0f}"
    elif formatter == "decimal":
        label_format = "{a:.2f}"
    else:
        raise ValueError(f"Formatter {formatter} not supported")
    
    # generate format table
    _, fmt_table = su.discretizer_w_distribution(vec=df_subset[var],
                                                weights=df_subset[weight],
                                                nbins=num_bins,
                                                dist=dist,
                                                fmtname="FMT_"+var,
                                                formatter=label_format,
                                                label_centroids=False,
                                                stdize=True)
    # cap format table
    fmt_table = cap_format(fmt_table, cap=cap, floor=floor)
    # add missing value and np value
    if missing_values != []:
        fmt_table = add_missing_values(fmt_table, missing_values, label="Missing")
    if np_values != []:
        fmt_table = add_missing_values(fmt_table, np_values, label="NP")

    return fmt_table



def parse_format_dict_row(row):
    params = row.to_dict()

    if np.isnan(params["missing_values"]):
        params["missing_values"] = []
    else:
        params["missing_values"] = [float(x) for x in str(params["missing_values"]).split(", ")]

    if np.isnan(params["np_values"]):
        params["np_values"] = []
    else:
        params["np_values"] = [float(x) for x in str(params["np_values"]).split(", ")]

    return params
