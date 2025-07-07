import os
import pyarrow
import numpy as np
import pandas as pd
import scipy.stats as ss
import gpc_utils.emblem as eu
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
from itertools import product
from collections import deque
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import lit, when, col




def confindence_interval(y, ths):
    # with this function we compute the interval of confindence from the Fisher information matrix (FI)
    # http://people.missouristate.edu/songfengzheng/Teaching/MTH541/Lecture%20notes/Fisher_info.pdf
    sub_sel = y[np.where( y > ths )]
    nobs = len(sub_sel)
    sub_sel = sub_sel - ths
    smpl_mean = sub_sel.mean()
    smpl_var = sub_sel.var()
    params = ss.genpareto.fit(sub_sel,floc=0,scale=1)            
    # computing The errors
    Fcc = lambda x: (csi**2*x**2*(csi + 1) + 2*csi*x*(csi*x + sigma) - 2*(csi*x + sigma)**2*np.log((csi*x + sigma)/sigma))/(csi**3*(csi*x + sigma)**2)
    Fss = lambda x: (-csi*x**2 + sigma**2 - 2*sigma*x)/(sigma**2*(csi**2*x**2 + 2*csi*sigma*x + sigma**2))
    Fcs = lambda x: x*(csi*x + sigma - x*(csi + 1))/(sigma*(csi*x + sigma)**2)
    csi, sigma = params[0], params[2]        
    FI = [[-Fcc(sub_sel).sum(), -Fcs(sub_sel).sum()], [-Fcs(sub_sel).sum(), -Fss(sub_sel).sum()]]
    FID = np.linalg.inv(FI)
    Eigen = np.diag(FID)**0.5
    dcsi = Eigen[0]    
    dsigma = Eigen[1]
    return  dcsi



def plot_qqplots(cost):
    
    dists = {'GPD': ss.genpareto, 'Gamma': ss.gamma}
    fits = {k: v.fit(cost, floc=0) for k, v in dists.items()}
    q_obs = cost.sort_values().values

    len_x = cost.quantile(0.999)

    fig, axs = plt.subplots(1,2,figsize=(10,5))
    nsin = len(q_obs)
    qteos = {k: v.ppf((np.arange(nsin)+1)/(nsin+1), *fits[k]) for k, v in dists.items()}
    axs = dict(zip(dists.keys(), axs))

    {k: axs[k].scatter(qteos[k],q_obs) for k,v in dists.items()}
    {k: axs[k].plot(qteos[k],qteos[k],color='r')for k,v in dists.items()}
    {k: axs[k].tick_params(axis='x',rotation=30) for k,v in dists.items()}
    {k: axs[k].set_xlabel(f'{k} quantile' ) for k,v in dists.items()},{k:axs[k].set_ylabel('Observed amount') for k,v in dists.items()}

    deque(map(lambda x: x.set_xlim(0,len_x),axs.values())),deque(map(lambda x: x.set_ylim(0,len_x),axs.values()))
    _={k: axs[k].set_title(f'Q-Q Plot: Observed vs {k}') for k,v in dists.items() }
    fig.tight_layout()



def format_data_for_analysis(df, coste, stros, ths_num, qmin=0.0, qmax=0.995):
    """
    cost: pd.Series - the aggregated cost
    ths_num: int - number of threshold to be analized
    """

    def ks(dist, cost, th):
        """
        implement ks test to compare the excess with a theoretical distribution
        """
        exc = cost[cost > th] - th
        ks = ss.kstest(rvs=exc, cdf=dist.name, args=dist.fit(exc, floc=0))
        return ks[0]
    
    cost = df[coste]
    claims = df[stros]

    # min and max threshold selection
    min_th = np.round(cost.quantile(qmin), 0)
    max_th = np.round(cost.quantile(qmax), 0)
    # distance between consecutive thresholds
    ths_step = np.floor((max_th-min_th)/ths_num)
    # thresholds list
    ths = np.arange(min_th, max_th, ths_step)

    cost = cost.to_numpy()

    # define the table needed to generate the LL analysis plots
    table = pd.DataFrame(ths, columns=["thresholds"])
    table["count"] = table.thresholds.apply(lambda x: (cost>x).sum())
    table["shape"] = table.thresholds.apply(lambda x: ss.genpareto.fit(cost[cost>x]-x)[0])
    table["sigma"] = table.thresholds.apply(lambda x: confindence_interval(cost, x))
    table["average excess"] = ((cost[:, None] - ths) * (cost[:, None] > ths)).sum(0) / (cost[:, None] > ths).sum(0)
    table["ks_gdp"] = table.thresholds.apply(lambda x: ks(ss.genpareto, cost, x))
    table["ks_gamma"] = table.thresholds.apply(lambda x: ks(ss.gamma, cost, x))
    table["Total Cost"] = cost.sum()
    table["Cost Above"] = table.thresholds.apply(lambda x: cost[cost > x].sum())
    table["Perc Cost Above"] = table["Cost Above"]/table["Total Cost"]
    table["Total Claims"] = claims.sum()
    table["Claims Above"] = table.thresholds.apply(lambda x: df[df[coste] > x][stros].sum())
    table["Perc Claims Above"] = table["Claims Above"]/table["Total Claims"]

    return table



def cuts_by_threshold(df_plot, threshold=None):

    # Create DataFrame
    tabla = df_plot[["thresholds", "Total Cost", "Cost Above", "Perc Cost Above", "Total Claims", "Claims Above", "Perc Claims Above"]]

    # Create Plotly table
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(tabla.columns), align='left'),
        cells=dict(values=[tabla[col] for col in tabla.columns], 
                   align='left', 
                   format=[['.0f'], ['$.2f'], ['$.2f'], ['.2%'], ['.0f'], ['.0f'], ['.2%']]))
    ])

    return fig



def plot_shape_parameter(df_plot, threshold=0):
   
    # Create the plot
    fig = go.Figure()

    # Add confidence interval fill
    ths = np.array(df_plot["thresholds"])
    fig.add_trace(go.Scatter(
        x=np.concatenate([ths, ths[::-1]]),
        y=np.concatenate([df_plot['shape'] + 1.96 * df_plot['sigma'], (df_plot['shape'] - 1.96 * df_plot['sigma'])[::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='Confidence Interval'
    ))

    # Add bar plot for the number of claims
    fig.add_trace(go.Bar(
        x=df_plot["thresholds"],
        y=df_plot['count'],
        name='Number of Claims',
        yaxis='y2',
        marker=dict(color='rgba(50, 171, 96, 0.6)')
    ))

    # Add shape parameter line
    fig.add_trace(go.Scatter(
        x=df_plot["thresholds"],
        y=df_plot['shape'],
        mode='lines+markers',
        name='Shape Parameter $\\xi$',
        line=dict(color='rgba(31, 119, 180, 1)')
    ))

    if threshold > 0:
        
        fig.add_shape(
            type="line",
            x0=threshold, x1=threshold,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="red", dash="dash"),
            name='Selected Threshold'
        )

    # Update layout
    fig.update_layout(
        title='Shape Parameter vs Threshold',
        xaxis=dict(title='Threshold'),
        yaxis=dict(title='Shape Parameter $\\xi$'),
        yaxis2=dict(
            title='Number of Claims',
            overlaying='y',
            side='right',
            range=[0, 2 * max(df_plot['count'])]  # Set y-axis max to twice the max value
        ),
        legend=dict(x=0, y=1)
    )

    return fig



def plot_average_excess(df_plot, threshold=0):
    # Get the threshold list from the dataframe index
    ths = np.array(df_plot.index)

    # Create the plot
    fig = go.Figure()

    # Add bar plot for the number of claims
    fig.add_trace(go.Bar(
        x=df_plot["thresholds"],
        y=df_plot['count'],
        name='Number of Claims',
        yaxis='y2',
        marker=dict(color='rgba(50, 171, 96, 0.6)')
    ))

    # Add average excess line
    fig.add_trace(go.Scatter(
        x=df_plot["thresholds"],
        y=df_plot['average excess'],
        mode='lines+markers',
        name='Average Excess',
        line=dict(color='rgba(31, 119, 180, 1)')
    ))

    if threshold > 0:    
        fig.add_shape(
            type="line",
            x0=threshold, x1=threshold,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="red", dash="dash"),
            name='Selected Threshold'
        )

    # Update layout
    fig.update_layout(
        title='Average Excess vs Threshold',
        xaxis=dict(title='Threshold'),
        yaxis=dict(title='Average Excess'),
        yaxis2=dict(
            title='Number of Claims',
            overlaying='y',
            side='right',
            range=[0, 2 * max(df_plot['count'])]  # Set y-axis max to twice the max value
        ),
        legend=dict(x=0, y=1)
    )

    return fig



def plot_ks_test(df_plot, threshold=0):

    # Create the plot
    fig = go.Figure()

    # Add bar plot for the number of claims
    fig.add_trace(go.Bar(
        x=df_plot["thresholds"],
        y=df_plot['count'],
        name='Number of Claims',
        yaxis='y2',
        marker=dict(color='rgba(50, 171, 96, 0.6)')
    ))

    # Add KS test lines for Gamma and GPD
    fig.add_trace(go.Scatter(
        x=df_plot["thresholds"],
        y=df_plot['ks_gamma'],
        mode='lines+markers',
        name='KS Test - Gamma',
        line=dict(color='rgba(31, 119, 180, 1)')
    ))
    fig.add_trace(go.Scatter(
        x=df_plot["thresholds"],
        y=df_plot['ks_gdp'],
        mode='lines+markers',
        name='KS Test - GPD'
    ))

    if threshold > 0:
        fig.add_shape(
            type="line",
            x0=threshold, x1=threshold,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="red", dash="dash"),
            name='Selected Threshold'
        )

    # Update layout
    fig.update_layout(
        title='KS Test vs Threshold',
        xaxis=dict(title='Threshold'),
        yaxis=dict(title='KS Test'),
        yaxis2=dict(
            title='Number of Claims',
            overlaying='y',
            side='right',
            range=[0, 2 * max(df_plot['count'])]  # Set y-axis max to twice the max value
        ),
        legend=dict(x=0, y=1)
    )

    return fig



def create_report_large_losses(df, gar, coste, stros, filename):

    # Generate the table needed for the plots
    df_plot = format_data_for_analysis(df, coste, stros, ths_num=100)

    print("table created")

    # Generate the plots and the table needed for the EVT analysis
    shape_plot = plot_shape_parameter(df_plot)
    exc_plot = plot_average_excess(df_plot)
    ks_plot = plot_ks_test(df_plot)
    table = cuts_by_threshold(df_plot)
    
    # Create a HTML file to store the analisys
    with open(filename, 'w') as f:
        
        title = f"{gar} - Large Loss Analisys"
        
        f.write(f'<html><head><title>{title}</title>')
        f.write('<style>body { margin: 0; padding: 40px; }</style>')  # Add margin for padding
        f.write('</head><body>')
        f.write(f'<h1 style="text-align:center;">{title}</h1>')
        
        # Write each figure to the HTML
        for fig in [shape_plot, exc_plot, ks_plot, table]:
            f.write('<div style="margin: 40px 80px;">')  # Add margin for padding
            f.write(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))
            f.write('</div>')
        
        f.write('</body></html>')

    # Add a column with the cover name
    df_plot["gar"] = gar

    return df_plot



def cap_cost(df_hogar, garantia, corte):
    #TODO handle missing threshold properly
    
    df_hogar = df_hogar.withColumn("CORTE", lit(corte).cast(IntegerType()))
    
    df_hogar = df_hogar.withColumn("CUPD_CAP_"+garantia, col("CUPD_"+garantia))
    df_hogar = df_hogar.withColumn("CUPD_CORR_"+garantia, col("CUPD_"+garantia))

    df_hogar = df_hogar.withColumn("CUPD_CORR_" + garantia, 
                               when((col("ETIQ_" + garantia) == "Coste negativo") | 
                                    (col("ETIQ_" + garantia) == "Coste nulo con siniestro"), lit(0))
                               .otherwise(col("CUPD_CORR_" + garantia)))
    
    df_hogar = df_hogar.withColumn("STRO_G_"+garantia, lit(0).cast(IntegerType()))
    df_hogar = df_hogar.withColumn("CUPD_EXC_"+garantia, lit(0).cast(IntegerType()))

    df_hogar = df_hogar.withColumn("CUPD_CAP_"+garantia, when((col('CUPD_'+garantia) > col('CORTE')) & (col('CORTE') > 0), col('CORTE'))
                                                    .otherwise(col("CUPD_CAP_"+garantia)))

    df_hogar = df_hogar.withColumn("CUPD_CAP_CORR_"+garantia, col("CUPD_CAP_"+garantia))

    df_hogar = df_hogar.withColumn("CUPD_CAP_CORR_"+garantia, 
                               when((col("ETIQ_" + garantia) == "Coste negativo") | 
                                    (col("ETIQ_" + garantia) == "Coste nulo con siniestro"), lit(0))
                               .otherwise(col("CUPD_CAP_CORR_"+garantia)))
    
    df_hogar = df_hogar.withColumn("STRO_G_"+garantia, when((col('CUPD_'+garantia) > col('CORTE')) & (col('CORTE') > 0), lit(1).cast(IntegerType()))
                                                   .otherwise(col("STRO_G_"+garantia)))
    df_hogar = df_hogar.withColumn("CUPD_EXC_"+garantia, when((col('CUPD_'+garantia) > col('CORTE')) & (col('CORTE') > 0), col('CUPD_'+garantia)-col('CORTE'))
                                                    .otherwise(col("CUPD_EXC_"+garantia)))

    df_hogar =df_hogar.drop(col('CORTE'))

    return df_hogar
    