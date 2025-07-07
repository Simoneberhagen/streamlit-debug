"""

Utilities to interface with Emblem using Python.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from itertools import groupby
from math import ceil, floor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from gpc_utils.emblem import parse_fac


def _b_roundup(number,digits):
    return ceil(number/(10**digits))*(10**digits)

def _b_rounddown(number,digits):
    return floor(number/(10**digits))*(10**digits)

def univariate(df,x=None,y=None,w=None,output=False,ax1=None,each=None,split_sentence=10,angle=45,legend=None):

    """
        Print out a oneway plot of a binned variable. A matplotlib.pyplot graph is generated.

        Parameters
        ----------
        df   : pandas DataFrame
            A table containing the data to use for the oneway creation
        x: string
            Name of the feature in df whose levels are used to group the observations by and create the oneway
        y : list
            List containing the names of the response variables in df  (e.g. 'observed_num_claims' or ['response','predicted'])
        w  : string
            Name of the weight variable in df (e.g. 'exposure')
        output: boolean
            Flag to return the oneway plot and the table as pandas DataFrame
        ax1 : Matplotlib Axis Object
            Name of the Matplotlib Axis Object wherein the user wants to plot the oneway. 
            If it is not assign, a new matplotlib.plot is created.
        each : int
            It indicates which labels to be printed in the x-axis: a label every "each" will be printed out.
            If None, an assignation depending on the number of x levels is done.
        split_sentence : int
            Unit used to split the level name of x in case they are categorical and too long to be displayed in a single line.
        angle : int
            Angle to orientate the x-labels in oneway plot
        legend : str or bool
            if True legend will be inseted into the graph, if string standard loc position of a matplotlib legend are supported (https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend.html), as well as 'outer right'
            
            
        Returns
        -------
        table: pandas Dataframe 
            (optional) If output is True, oneway table is returned as well 
	ax1 : Matplotlib Axis Object
		    Matplotlib Axis Object wherein the weight histogram binned by x levels is plotted. 
        ax2 : Matplotlib Axis Object
            Matplotlib Axis Object wherein the response line graph is plotted.
        
            
        Examples
        --------
	
		.. code-block:: python

            import gpc_utils.eu as eu
       
            import numpy as np
            import random
            import pandas as pd
            import gpc_utils.emblem as eu
            
            #Creation Random Database for input
            np.random.seed(0)
            df=pd.DataFrame(np.random.rand(100,5),columns=[f'num_'+str(i) for i in range(5)])
            df['x']=np.array([chr(x) for x in np.random.randint(97,122,size=100)],dtype="object")
            
            #Create plot
            eu.univariate(df,filename,'x','num_0','num_2')
            
            #Create plot considering a list of response variable
            eu.univariate_excel(df,'x',['num_0','num_1'],'num_2')
            
            #Create plot considering a list of response variable and returning as output the oneway table as well
            output=eu.univariate_excel(df,'x',['num_0','num_1'],'num_2',True)
            table=output[0]
            
    """  
    # Check whether y is a list of features. It returns a list containing x,y (including all the elements) and w.
    if isinstance(y,list):
        cols=[c for c in [x,*y,w] if c is not None]
    else:
        cols = [c for c in [x, y, w] if c is not None]
    
    #If x is not assigned to a feature in df, the script assigns it to the number of the observation, i.e. x=[0,1,2,...len(df)]
    if x is None: x=df.index.to_series()
    
    #Selection only of the cols of interest in df
    ddf=df[cols]
    #creation of an array sr1 contening the percentage of w per each level of x. If w is not specified by the user, it returns the percentage of observation per each level of x.
    sr1=ddf.groupby(x, observed = False)[w].sum().div(ddf[w].sum()) if w is not None else ddf.groupby(x, observed = False)[x].count().div(len(ddf))
    table=pd.DataFrame(sr1)
    if w is not None:
        table[w+'_Sum']=ddf.groupby(x, observed = False)[w].sum()
    else:
        table['Count']=ddf.groupby(x, observed = False)[x].count()
    #Plot of w histogram binned by x levels. The bars are centrally alligned with respect to the ticks and the graph is created on ax1 axis. Colours are chosen keeping the EMB ones.
    ax1=sr1.plot.bar(align='center',ax=ax1,color='#f4f00e',use_index=False,edgecolor='#909293')
    
    #Set y-axis limit
    ymax=sr1.max()*2.5
    ax1.set_ylim([0,ymax])
    
    #Settings for the plot
    ax1.yaxis.grid(False) #No grid in the plot
    ax1.set_xlabel('') #No x-axis label
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) #Set the formatter of the major ticker, in particular the nummber of decimals
    
    
    xmin,xmax=ax1.get_xlim() #Get the limits of the x-axis  
    if y is not None:
        #If y is assigned, creation of a pandas array sr2 (if y is a list) contening the weighted average of y per each level of x. If w is not specified by the user, it returns the average of y per each level of x. 
        sr2=ddf.groupby(x, observed = False)[y].mean() if w is None else ddf.groupby(x, observed = False)[y].aggregate(lambda y: np.ma.average(y, weights=ddf.loc[ddf.index.isin(y.index),w]))
        table[y]=sr2
        ax2=ax1.twinx() # Create a twin Axes sharing the xaxis
        ymin,ymax=sr2.min()-sr2.mean()*.4,sr2.max()*1.1 #Change y-axis limit
        ax2=sr2.plot(ax=ax2,linestyle='--',marker='o',use_index=False) #Line plot of the sr2 table containing the weighted average response per each x level
        if isinstance(ymin,pd.Series):
            ymin,ymax=min(ymin),max(ymax) #Change y-axis limit
        #Setting for secondary chart
        ax2.set_ylim([ymin,ymax])
        ax2.set_xlim([xmin-.5,xmax+.5])
        ax2.yaxis.tick_left()
    ax1.set_title(x if isinstance(x,str) else x.name) # Title Settings
    ax1.yaxis.tick_right()
    
    #Function used to split the x-label whether the x levels are string
    splitter=lambda x: "\n".join([" ".join([gg for kk,gg in g]) for k,g in groupby(enumerate(x.split(" ")),lambda xx: ceil((xx[0]+1)/split_sentence))])
    
    # if ech is None, a value is assigned depending on the number of levels of x-variable: 1 (less than 12), 2 (less than 20), 4 otherwise.
    if each is None: each=1 if len(sr1.index)<= 11 else 2 if len(sr1.index)<20 else 4
    
    canbesplitted=False
    
    #Recognize whether the x levels are categorical and change the flag canbeplitted accordingly
    if isinstance(sr1.index.dtype,pd.CategoricalDtype):
        if pd.api.types.infer_dtype(sr1.index.categories)=="string":
            canbesplitted = True
    elif pd.api.types.infer_dtype(sr1.index)=="string":
        canbesplitted=True

    if canbesplitted:
        ax1.set_xticklabels([ splitter(x) if n % each ==0 else '' for n,x in enumerate(sr1.index)])
    else:
        ax1.set_xticklabels([ x if n % each ==0 else '' for n,x in enumerate(sr1.index)])
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=angle, ha="right")
    plt.tight_layout()
    
    if legend is not None:
        if isinstance(legend,bool):
            ax2.legend(loc='best')
        elif legend=='outer right':
            ax2.legend(loc='center left',bbox_to_anchor=(1.2, 1.0))
        else:
            ax2.legend(loc=legend)

    if output is True: 
        return (table,ax1,ax2)
    else:
        return (ax1,ax2)


def univariate_plotly(df,
            x,
            y,
            w,
            path_fac=None,
            response_format_table=".4f",
            response_format_plot=None,
            exposure_format=".2%",
            fig_title="",
            xtick_rotation=-90,
            fig_w=1000,
            fig_h=1000,
            show_fig=True,
            tickfont=dict(size=12),
            optimize_labels=False,
            html_output=None,
            retfig=False,
            output=False):
    
    """
        Print out a oneway plot of a binned variable. A matplotlib.pyplot graph is generated.

        Parameters
        ----------
        df   : pandas DataFrame
            A table containing the data to use for the oneway creation
        x: string, list
            Name or list of the features in df whose levels are used to group the observations by and create the oneway (e.g. 'year' or ['year','bonus_malus'])
            It can be None only if path_fac is provided. In such case the oneway for all rating factors in the .Fac file are produced
            If x is a list, a dropdown menu is added at the top left of the graph allowing to switch among rating factors. 
        y : string,list
            string or list containing the names of the response variables in df  (e.g. 'observed_num_claims' or ['response','predicted'])
            If a list is provided, all the response variables are plotted in the graph. 
            Single response can be activated/deactivated by clicking on the correspondig name in the graph legend. The y-axis scale is then adjusted accordingly.
        w  : string
            Name of the weight variable in df (e.g. 'exposure')
        path_fac : string
            path to Emblem .Fac file. If provided, rating factors level are read from here. 
            If provided then x paramater can be set to None. In such case the oneway for all rating factors in the .Fac file are produced.
        response_format_table : string
            A format string that specify how response variable is formatted in the oneway table
            The format string must follow the d3 format convention:
            See https://github.com/d3/d3-format/blob/master/README.md#locale_format
        response_format_plot: string
            A format string that specify how response variable is formatted in the oneway graph
            The format string must follow the d3 format convention:
            See https://github.com/d3/d3-format/blob/master/README.md#locale_format
        exposure_format: string
            A format string that specify how the weight variable is formatted in the oneway graph and table
            Default: '.2%' i.e. percentage with 2 decimal digits
            The format string must follow the d3 format convention:
            See https://github.com/d3/d3-format/blob/master/README.md#locale_format
        xtick_rotation : int
            Rotation of xticks labels (Default to -90)
        fig_w : int
            Width of the output figure (Default to 1000)
        fig_h : int
            Height of the output figure (Default to 1000)
        show_fig: boolean
            Flag to show the produced graph and table (Default to True)
        tickfont: dict
            Specify xaxis font style. (Default to dict(size=12) )
            Other options, such as family font and color, can be found at https://plotly.com/python/reference/#scatter-marker-colorbar-tickfont
        optimize_labels: boolean
            Flag to optimize the number of ticks on the xaxis. 
            Avoid ticks overlapping when having many levels by removing some of them (Default to False)
        html_output : str 
            Path to the html output file to store the output graph and table
        retfig: boolean
            Flag to return the plotly figure object.
            If both retfig and output are True, then a dictionary of tables and a plot are returned 
        output: boolean
            Flag to return the dictionary of oneway tables as pandas DataFrame
            
        Returns
        -------
        ow_tables: dictionary of pandas Dataframe 
            (optional) If output is True, a dictionary whose keys are the feature names (e.g. 'year') is returned as well
        ow_tables, fig: dictionary of pandas Dataframe, plotly figure object 
            (optional) If both ouput and retfig are True
        
            
        Examples
        --------
	
		.. code-block:: python

            import gpc_utils.eu as eu
       
            import numpy as np
            import random
            import pandas as pd
            import gpc_utils.emblem as eu
            
            #Creation Random Database for input
            np.random.seed(0)
            df=pd.DataFrame(np.random.rand(100,5),columns=[f'num_'+str(i) for i in range(5)])
            df['x']=np.array([chr(x) for x in np.random.randint(97,122,size=100)],dtype="object")
            
            #Create plot
            eu.univariate_plotly(df,'x','num_0','num_2')
            
            #Create plot considering a list of response variable
            eu.univariate_plotly(df,'x',['num_0','num_1'],'num_2')
            
            #Create plot considering a list of response variable and returning as output the oneway table as well
            output=eu.univariate_plotly(df,x='x',y=['num_0','num_1'],w='num_2',output=True)
            table=output.get('x')[0]
            base_level=output.get('x')[1] #base level is None if path_fac is None
            
            #Produce oneway from .Fac and .Bid files
            bid,_=eu.parse_bid_fac(path_to_bid,path_to_fac)
            eu.univariate_plotly(bid,x=None,y='Response',w='Weight',path_fac=path_to_fac)
            
            #Create an HTML report
            eu.univariate_plotly(df,x='x',y=['num_0','num_1'],w='num_2',html_output='oneway.html')
            
    """
    if x and not(isinstance(x,str) or isinstance(x,list)):
        raise TypeError('x must be a string or a list of strings')
    if x and isinstance(x,list) and (not all(isinstance(elem, str) for elem in x)):
        raise TypeError('All elements of x must be strings')
    if path_fac:
        names=parse_fac(path_fac)[-1]    
    if x and isinstance(x,str):
        loop_on=[x]
    elif x and isinstance(x,list):
        loop_on=x
    elif not x:
        if not path_fac:
            raise ValueError('\"x\" can be None only if \"path_fac\" is also specified')
        else:
            loop_on=list(names.keys())
    
    if not isinstance(df,pd.DataFrame):
        raise TypeError('\"df\" variable must be a pandas dataframe. Type '+str(type(df))+' received.')
    
    if isinstance(y,list):
        cols=[c for c in loop_on+[*y,w] if c is not None]
    else:
        cols = [c for c in loop_on+ [y, w] if c is not None]
        
    ddf=df[cols].copy()
    ddf[y]=ddf[y].multiply(ddf[w],axis="index")
    ow_tables=dict()
    sum_of_weights=ddf[w].sum()
    for rf_name in loop_on:
        if path_fac:
            if not rf_name in names:
                raise KeyError('Rating factor \"'+rf_name + '\" not found in fac file')
            rf_num_levels=len(names.get(rf_name).get('levels'))
            rf_labels=names.get(rf_name).get('levels')
            rf_base_level=names.get(rf_name).get('base_level')
        
        #Compute OneWay
        sr1=ddf.groupby(rf_name)[w].sum().div(ddf[w].sum()) if w is not None else ddf.groupby(rf_name)[rf_name].count().div(len(ddf))
        obs_ow=pd.DataFrame(sr1)
        obs_ow[w+'_Sum']=ddf.groupby(rf_name)[w].sum() if w is not None else ddf.groupby(rf_name)[rf_name].count()
        #sr2=ddf.groupby(rf_name)[y].mean() if w is None else ddf.groupby(rf_name)[y].aggregate(lambda y: np.ma.average(y, weights=ddf.loc[ddf.index.isin(y.index),w]))

        sr2=ddf.groupby(rf_name)[y].mean() if w is None else ddf.groupby(rf_name)[y].sum()
        obs_ow[y]=sr2
        obs_ow[y]=obs_ow[y].divide(obs_ow[w+'_Sum'],axis="index")
        if path_fac:
            #Handle levels not seen in database.
            #The OW table is merged with a simple dataframe contaiing all levels of the given RF. If a level is not seen then 0 is assigned as weighted average and sum weights
            ow=pd.DataFrame({rf_name:np.arange(int(rf_num_levels))+1,'label':rf_labels})
            ow=ow.merge(obs_ow,how='left',left_on=rf_name,right_index=True).fillna(0)
            ow_tables.update({rf_name:[ow,int(rf_base_level)]})
        else:
            ow=obs_ow
            ow['label']=ow.index
            ow['label']=ow['label'].astype(str)
            ow_tables.update({rf_name:[ow,None]})
    
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=1, cols=1,
                        specs=[[{"secondary_y": True}]],  # Remove the table spec
                        vertical_spacing=0.15,
                        row_heights=[1],  # Adjust row height since there's only one row now
                        start_cell="top-left"
                    )
    visible = True
    labels_dict = dict()
    labels_index_dict = dict()
    resp_dict = dict()

    for irf, ow_table in ow_tables.items():
        table = ow_table[0]
        if path_fac:
            base_level = ow_table[1] - 1  # Bins in emblem are counted from 1
        else:
            base_level = None
        x = np.arange(0, len(table), 1)
        data2 = table[w]
        for resp_name in [k for k in cols if k is not w and k not in loop_on]:
            resp_dict.update({resp_name: table[resp_name]})
        labels = table['label']

        if optimize_labels:
            # Try to find best number of labels to avoid text overlap
            avg_length = (len(labels.sum()) * int(tickfont['size']) + 4 * tickfont['size'] * len(labels)) / len(labels)
            max_levels = int(fig_w / avg_length)
            max_levels = max_levels if max_levels <= len(x) else len(x)
            new_labels_indexes = np.linspace(start=0, stop=len(x) - 1, num=max_levels, dtype=int)
            new_labels = labels.iloc[new_labels_indexes]
            labels_dict.update({irf: new_labels})
            labels_index_dict.update({irf: new_labels_indexes})
        else:
            labels_dict.update({irf: labels})
            labels_index_dict.update({irf: x})

        # Add traces
        opacity = 0.6
        width = 0.5
        colors = ['yellow',] * len(x)
        ntraces = 0
        for resp_name, resp_data in resp_dict.items():
            fig.add_trace(
                go.Scatter(x=x, y=resp_data,
                        name=resp_name,
                        marker_color='rgb(49,130,189)' if len(resp_dict) == 1 else None,
                        mode='lines+markers',
                        marker_size=8,
                        visible=visible),
                secondary_y=True,
                row=1,
                col=1
            )
            ntraces += 1

        # Base level
        if path_fac:
            fig.add_trace(
                go.Bar(x=[x[base_level]], y=[data2[base_level]], name="Base Level",
                    marker_color='red',
                    visible=visible,
                    opacity=opacity,
                    width=width,
                    text='Base Level',
                    textposition='auto',
                    marker=dict(
                        line=dict(color='black', width=1)
                    )),
                secondary_y=False,
                row=1, col=1
            )

        # All levels
        if path_fac:
            x_rf = np.delete(x, base_level)
            data2_rf = data2.drop(index=base_level)
        else:
            x_rf = x
            data2_rf = data2

        fig.add_trace(
            go.Bar(x=x_rf, y=data2_rf, name="% Weight",
                marker_color=colors,
                visible=visible,
                opacity=opacity,
                width=width,
                marker=dict(
                    line=dict(color='black', width=1)
                )),
            secondary_y=False,
            row=1, col=1
        )

        visible = False

    if path_fac:
        ntraces = ntraces + 3
    else:
        ntraces = ntraces + 2

    idx = np.arange(len(ow_tables) * ntraces)
    visible = {rf: np.where(((idx >= i * ntraces) & (idx <= i * ntraces + (ntraces - 1))), True, False) for i, rf in enumerate(ow_tables)}

    if len(ow_tables) > 1:
        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    yanchor="top",
                    xanchor="left",
                    y=1.1,
                    x=0.,
                    buttons=[{
                        'label': re.sub("(.{31})", "\\1<br>", rf, 0, re.DOTALL),
                        'method': "update",
                        'args': [
                            {"visible": viz.tolist()},
                            {"title": f"Rating Factor: {rf}",
                            "xaxis": {'title': rf,
                                    'tickmode': 'array',
                                    'tickvals': list(labels_index_dict[rf]),
                                    'ticktext': list(labels_dict[rf]),
                                    'tickangle': xtick_rotation,
                                    'tickfont': tickfont}},
                            {'width': fig_w}
                        ]} for rf, viz in visible.items()]
                )
            ]
        )

    # Add figure title
    fig.update_layout(
        xaxis=dict(
            # title=next(iter(ow_tables.items()))[0],
            tickangle=xtick_rotation,
            tickfont=tickfont,
            tickmode='array',
            tickvals=next(iter(labels_index_dict.values())),  # by default print first rating factor ticks
            ticktext=next(iter(labels_dict.values()))  # by default print first rating factor ticks
        ),
        width=fig_w,
        height=fig_h,
        yaxis2=dict(side='left', tickformat=response_format_plot),
        yaxis=dict(side='right', tickformat=exposure_format),
        title={
            'text': fig_title,
            'y': 0.95, 
            'x': 0,
            # 'xanchor': 'center', 
            # 'yanchor': 'top',
            'font': {'size': 25}
            }
    )

    # Set y-axes titles
    fig.update_yaxes(title_text="% Weight", secondary_y=False)
    fig.update_yaxes(title_text="Response average", secondary_y=True)

    fig.update_layout(
        legend=dict(x=0.8, y=1.1, bgcolor='rgba(255, 255, 255, 0.5)')
    )

    if html_output:
        fig.write_html(html_output, default_width='100%')
    if show_fig:
        fig.show(renderer='notebook')
    if retfig:
        if output:
            return ow_tables, fig
        else:
            return fig
    elif output:
        return ow_tables
    else:
        return

    
__all__=["univariate","univariate_plotly"]
