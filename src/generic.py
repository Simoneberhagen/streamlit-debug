from os import environ
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, when


def init_spark_session(jars_path):
    environ['JAVA_HOME'] = "C:\\Program Files\\Java\\jre1.8.0_202\\"

    driver = '50g'
    executor = '50g'
    executor_cores = 4

    spark = SparkSession \
        .builder \
        .enableHiveSupport() \
        .config("spark.driver.memory", driver)\
        .config("spark.executor.memory", executor) \
        .config("spark.executor.cores", executor_cores) \
        .config("spark.sql.parquet.datetimeRebaseModeInWrite",'LEGACY')\
        .config("spark.sql.parquet.datetimeRebaseModeInRead",'LEGACY')\
        .config("spark.sql.parquet.compression.codec", "snappy")\
        .config("spark.driver.maxResultSize", "20g")\
        .config("spark.jars", jars_path) \
        .appName("Credit-Score") \
        .master("local[*]")\
        .getOrCreate()

    return spark



def exp_1dia(df, etiqK: str, etiqCac: str, etiqStro: str):
    df = df.withColumn(
        f"EXP_CORR_{etiqK}",
        when(
            ((col(f"STRO_CORR_{etiqStro}")>0) | (round(col(f"CUPD_CORR_{etiqCac}"), 2) > 0)) & (col(f"EXP_CORR_{etiqK}") == 0),
            col("EXPOSICION")
        ).otherwise(col(f"EXP_CORR_{etiqK}"))
    )
    return df



def stros_sinexp(df,name,etiqK,etiqCac,etiqStro):
    df= df.withColumn("FLAG_STRO_SINEXP_"+name , when((((col("STRO_CORR_"+etiqStro)>0) & (col("STRO_CORR_"+etiqStro).isNotNull())) |
                                              ((round(col("CUPD_CORR_"+etiqCac),2)>0) & (round(col("CUPD_CORR_"+etiqCac),2).isNotNull()))) &
                                               (((col("EXP_CORR_"+etiqK)==0) | (col("EXP_CORR_"+etiqK).isNull())) & ((col("EXPOSICION")==0) 
                                               | (col("EXPOSICION").isNull()))) ,
                                               1).otherwise(0))
    return df



def create_reconciliation_df(df_path, spark):

    df = spark.read.parquet(df_path)

    var_clasification = ['YEAR', 'RAMO', 'NP_RN'] + \
                        [col for col in df.columns if col.startswith('ETIQ_')]  + \
                        [col for col in df.columns if col.startswith('FLAG_STRO_SINEXP')]

    var_analysis = [col for col in df.columns if 
                        col.startswith('K_') or 
                        col.startswith('CUPD_') or 
                        col.startswith('STRO_') or 
                        col.startswith('EXP_')
                   ] + ['EXPOSICION']

    var_analysis = [col for col in var_analysis if not
                    (col.startswith("STRO_1A") or
                    col.startswith("STRO_2A") or
                    col.startswith("STRO_3A") or
                    col.startswith("STRO_4A") or
                    col.startswith("STRO_5A"))]

    df = df.select(var_clasification + var_analysis)

    rec_df = df.groupBy(*var_clasification).agg(*[sum(col(col_name)).alias(col_name) for col_name in var_analysis])
    rec_df =  rec_df.toPandas()

    rec_df = rec_df[var_clasification + var_analysis]

    rec_df.to_excel(df_path+"_reconciliation.xlsx")

    return rec_df



def set_factors_hyperparameters(data_dict, pd_gar, year):

    variable_mapping = {}

    for factor in data_dict.index: 

        if data_dict.loc[factor,'BASE_FORZADA']==1:
                    
            if factor=='YEAR':
                level_base = str(int(year)-1)
                base_level = pd_gar[factor].cat.categories.get_loc(level_base)+1

            else:
                try:
                    level_base = 'NP'
                    base_level = pd_gar[factor].cat.categories.get_loc(level_base)+1
                except:
                    level_base = '99. NP'
                    base_level = pd_gar[factor].cat.categories.get_loc(level_base)+1
            
        else:
            
            if data_dict.loc[factor,'CONTINUA']==0: ## Columna discreta

                base_level_position = pd_gar[factor].mode().cat.codes.iloc[0]+1
                base_level_value = pd_gar[factor].cat.categories[base_level_position-1]
        
                if base_level_value in ["Missing", "Desconocido"]:
            
                    base_level_previo = pd_gar[factor].value_counts().index[1]
                    base_level = pd_gar[factor].cat.categories.get_loc(base_level_previo)+1
            
                    if base_level_previo in ["Missing", "Desconocido"]:
                        
                        base_level_previo = pd_gar[factor].value_counts().index[2]
                        base_level = pd_gar[factor].cat.categories.get_loc(base_level_previo)+1
                else:   
                     base_level=pd_gar[factor].mode().cat.codes.iloc[0]+1
                    
            else: #### Columna Continua
                
                base_level_position = pd_gar[factor].mode().cat.codes.iloc[0]+1
                base_level_value = pd_gar[factor].cat.categories[base_level_position-1]
                
                # Define characters that identify binned or special categories to be avoided as base level
                excluded_chars = ['[', '(', ')', ']', '>', '<']
                excluded_labels = ["Missing", "Desconocido", "Other", "NP"]

                # Check if the most frequent value (mode) is a binned or excluded category
                if any(c in str(base_level_value) for c in excluded_chars) or base_level_value in excluded_labels:
                    # If it is, find the next most frequent value that is not a binned or excluded category
                    base_level = base_level_position # Fallback to mode
                    for idx in range(1, len(pd_gar[factor].value_counts())):
                        next_most_frequent_value = pd_gar[factor].value_counts().index[idx]
                        if not any(c in str(next_most_frequent_value) for c in excluded_chars) and next_most_frequent_value not in excluded_labels:
                            base_level_value = next_most_frequent_value
                            base_level = pd_gar[factor].cat.categories.get_loc(base_level_value) + 1
                            break
                else:
                    # If the mode is not a binned or excluded category, use it as the base level
                    base_level = base_level_position
                      
        variable_mapping[factor] = { 'format': data_dict.loc[factor,"format"].upper(),
                                     'base_level':base_level ,
                                     'label_name': data_dict.loc[factor, 'LABEL']} 
    return variable_mapping



def crear_random(df, resp, w, rand_factor):

    split_metrics = {}

    # running the seed search to properly split the dframe in 10 different levels with homogeneous response kpi
    for seed in range(1, 20):
        # define random state
        rng = np.random.RandomState(seed)
        # generate random train-test split
        df[rand_factor] = rng.randint(1, 11, len(df))
        trn_subset = df[rand_factor] <= 8
        test_subset = df[rand_factor] > 8
        # compute average wheighted response over train and tes
        obs_train = df.loc[trn_subset, resp].sum()/df.loc[trn_subset, w].sum()
        obs_test = df.loc[test_subset, resp].sum()/df.loc[test_subset, w].sum()
        rel_dif = np.abs(obs_train - obs_test)/obs_train

        split_metrics[seed] = {"obs_train": obs_train, "obs_test": obs_test, "rel_dif": rel_dif}

        if rel_dif < 0.01:
            break
    
    best_seed = min(split_metrics, key=split_metrics.get("rel_dif"))
    rng = np.random.RandomState(best_seed)
    df[rand_factor] = rng.randint(1, 11, len(df))
    df[rand_factor] = df[rand_factor].apply(lambda x: str(x).zfill(2))
    
    return df[rand_factor], split_metrics[best_seed]

def determine_default_base_level(univariate_table, weight, non_base_labels):
    """
    Determines the default base level from a univariate table.

    Args:
        univariate_table (pd.DataFrame): The univariate table.
        weight (str): The name of the weight column.
        non_base_labels (list): A list of labels to exclude.

    Returns:
        str: The default base level.
    """
    import re

    excluded_chars = ['[', '(', ')', ']', '>', '<']

    # Filter out non-base labels
    eligible_levels = univariate_table[~univariate_table['label'].isin(non_base_labels)]

    # Further filter out labels containing any of the excluded characters
    for char in excluded_chars:
        eligible_levels = eligible_levels[~eligible_levels['label'].astype(str).str.contains(re.escape(char))]

    if not eligible_levels.empty:
        default_base_level = eligible_levels.loc[eligible_levels[weight].idxmax()]['label']
    else:
        default_base_level = None
    
    return default_base_level