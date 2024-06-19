import numpy as np
import pandas as pd
import os

import scipy
import scipy.stats as st

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib 

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from matplotlib import cm

####
# Data processing and bootstrapping
####

# Bootstrapping function
def bootstrap_sample(df, n_samples=1000):

    measurements = df.values
    medians = []

    for i in range(n_samples):

        samples = np.random.choice(measurements, size = len(measurements))
        medians.append(np.median(samples))

    medians = np.asarray(medians)

    return medians

def bootstrap_sample_df(df,factor,ctl_label):

    '''
    Generate bootstrapped sample and return as dataframe, to be plotted with seaborn
    '''

    # Calculate the differences for each category and save them into dataframes for visualizing in Seaborn or Matplotlib
    bootstrap_diff_df = pd.DataFrame()

    # Get the control bootstrap
    ctl_bootstrap = bootstrap_sample(df[factor][df['Condition'] == ctl_label])

    for i in range(0,len(pd.unique(df['Condition']))):

        # Use the ctl_bootstrap if we're now on that condition, otherwise will create a new bootstrap sample that won't be the same.
        if(pd.unique(df['Condition'])[i] == ctl_label):
            bootstrap = ctl_bootstrap
        else:
            bootstrap = bootstrap_sample(df[factor][df['Condition'] == pd.unique(df['Condition'])[i]])

        difference = bootstrap - ctl_bootstrap
        this_cond =  pd.unique(df['Condition'])[i]
        this_diff_df = pd.DataFrame(data={'Difference':difference, 'Condition':this_cond})
        bootstrap_diff_df = bootstrap_diff_df.append(this_diff_df)

        
    # Calculate and print mean effect size for each condition
    mean_effect_sizes = bootstrap_diff_df.groupby('Condition')['Difference'].mean()
    print("Mean Effect Size for Each Condition Compared to Control:")
    for condition, mean_effect_size in mean_effect_sizes.items():
        print(f"The effect size for {condition} compared with control is: {mean_effect_size}")

    return bootstrap_diff_df

# #####
# Plotting Functions
# #####

def plots_of_differences_plotly(df_in,factor='Value', ctl_label = 'wt', palette='tab10', plot_type='swarm',conditions_to_include='all'):
    
    '''
    A function to create the plots of differences plots with bootstrapping CIs on sample data. 
    Based on the method and code from Joachim Goedhart doi: https://doi.org/10.1101/578575
        https://www.biorxiv.org/content/10.1101/578575v1.full.pdf+html
        and related code in R: 
        https://github.com/JoachimGoedhart/PlotsOfDifferences/blob/master/app.R
    '''
    
    df = df_in.copy()

    if ctl_label == -1:
        grouping = 'label'
        assert len(df[grouping].unique()) < 30, str(len(df[grouping].unique()))+ ' groups will be difficult to display, try optimizing the clustering.'

    else:
        
        grouping = 'Condition'

        if conditions_to_include  == 'all':
            conditions_to_include = df['Condition'].unique()

        # Sort the dataframe by custom category list to set draw order
        df[grouping] = pd.Categorical(df[grouping], conditions_to_include)


    df.sort_values(by=grouping, inplace=True, ascending=True)

    
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True)

    # Resize points based on number of samples to reduce overplotting.
    if(len(df) > 1000):
        pt_size = 1
    else:
        pt_size = 3


    # Get the control bootstrap

    # Get a colormap the length of unique condition (or whatever they're being grouped by)
    colors = np.asarray(sns.color_palette(palette, n_colors=len(pd.unique(df[grouping]))))

    ctl_bootstrap = bootstrap_sample(df[factor][df['Condition'] == ctl_label])

    # Store the calculated CIs in a list of shapes to add to the plot using shape in update layout.
    shape_list = []    

    for i in range(0,len(pd.unique(df[grouping]))):

        if (plot_type=='swarm'):

            # Plot the points
            fig.add_trace(go.Violin(
                                    # Select the subset of the dataframe we need by chaining another [] as condition to plot the susbset of the dataframe.
                                    x=df[factor][df[grouping] == pd.unique(df[grouping])[i]],
                                    y=df[grouping][df[grouping] == pd.unique(df[grouping])[i]],
                                    line={
                                        'width': 0
                                    },
                                    points="all",
                                    pointpos=0,
                                    marker={
                                         'color': 'rgb' + str(tuple(colors[i,:])),#colors[i],#'black' #diff_df['Value'].values
                                         'size': pt_size
                                         #'color': colors[np.where(cond_list == diff_df['Condition'])[0]]
                                    },
                                    orientation="h",
                                    jitter=1,
                                    fillcolor='rgba(0,0,0,0)',
                                    width= 0.75, # Width of the violin, will influence extent of jitter
                                   ),

                                    row=1, col=1)

        elif (plot_type=='violin'):

            # Plot the points
            fig.add_trace(go.Violin(
                                    # Select the subset of the dataframe we need by chaining another [] as condition to plot the susbset of the dataframe.
                                    x=df[factor][df[grouping] == pd.unique(df[grouping])[i]],
                                    y=df[grouping][df[grouping] == pd.unique(df[grouping])[i]],
                                    line={
                                        'width': 1
                                    },
                                    pointpos=0,
                                    marker={
                                         'color': 'rgb' + str(tuple(colors[i,:])),#colors[i],#'black' #diff_df['Value'].values
                                         'size': pt_size

                                    },
                                    orientation="h",
                                    side='positive',
                                    meanline_visible=True,
                                    points=False,
                                    width= 0.75, # Width of the violin, will influence extent of jitter
                                   ),

                                    row=1, col=1)

        elif (plot_type=='box'):

            # Plot the points
            fig.add_trace(go.Box(
                                # Select the subset of the dataframe we need by chaining another [] as condition to plot the susbset of the dataframe.
                                x=df[factor][df[grouping] == pd.unique(df[grouping])[i]],
                                y=df[grouping][df[grouping] == pd.unique(df[grouping])[i]],
                                line={
                                    'width': 1
                                },
#                                 pointpos=0,
                                marker={
                                     'color': 'rgb' + str(tuple(colors[i,:])),#colors[i],#'black' #diff_df['Value'].values
                                     'size': pt_size

                                },
                                orientation="h",
                                boxpoints='all',
                               ),

                                row=1, col=1)



        # Use the ctl_bootstrap if we're now on that condition, otherwise will create a new bootstrap sample that won't be the same.
        if(pd.unique(df['Condition'])[i] == ctl_label):
            bootstrap = ctl_bootstrap
        else:
            bootstrap = bootstrap_sample(df[factor][df['Condition'] == pd.unique(df['Condition'])[i]])

        difference = bootstrap - ctl_bootstrap

        # Calculate the confidence interval
        sample = df[factor][df['Condition'] == pd.unique(df['Condition'])[i]]
        raw_ci = st.t.interval(0.95, len(sample)-1, loc=np.mean(sample), scale=st.sem(sample))
        diff_ci = np.percentile(difference, [2.5,97.5])

        fig.add_trace(go.Violin(
                                # Select the subset of the dataframe we need by chaining another [] as condition to plot the susbset of the dataframe. 
                                x=difference,
                                y=df['Condition'][df['Condition'] == pd.unique(df['Condition'])[i]],    

                                side="positive",
                                orientation="h",
                                points=False,
                                # line_color = colors[i] #'black'
                                line_color = 'rgb' + str(tuple(colors[i,:])),#colors[i], #'black'

                               ), 
                                row=1, col=2)

 

        # Add raw CI to the list of dicts
        shape_list.append(dict(type="line", xref="x1", yref="y1",
                                 x0=raw_ci[0], y0=i, x1=raw_ci[1], y1=i, line_width=6))
        # Add diff CI to the list of dicts
        shape_list.append(dict(type="line", xref="x2", yref="y2",
                                 x0=diff_ci[0], y0=i, x1=diff_ci[1], y1=i, line_width=6))


    shape_list.append(dict(type="line", xref="x2", yref="y2",x0=0, y0=-1, x1=0, y1=3, line_width=2)) # Thick line at x=0 for the difference plot

    fig.update_layout(height=500, width=800,showlegend=False,
                      title_text="Plots of Differences",
                      shapes=shape_list # Accepts a list of dicts.
                     )
    fig.update_xaxes(title_text=factor, row=1, col=1)
    fig.update_xaxes(title_text="Difference", row=1, col=2)
    fig.show() 
    

    
def plots_of_differences_sns(df_in, factor='Value', ctl_label = 'wt', conditions_to_include='all'):

    '''
    A function to create the plots of differences plots with bootstrapping CIs on sample data. 
    Based on the method and code from Joachim Goedhart doi: https://doi.org/10.1101/578575
        https://www.biorxiv.org/content/10.1101/578575v1.full.pdf+html
        and related code in R: 
        https://github.com/JoachimGoedhart/PlotsOfDifferences/blob/master/app.R
    '''
    
    plt.rcParams.update({'font.size': 12})
    plt.clf()

    assert ctl_label in df_in['Condition'].values, ctl_label + ' is not in the list of conditions'

    df = df_in.copy()
    
    if conditions_to_include  == 'all':
        conditions_to_include = df['Condition'].unique()

    # Sort values according to custom order for drawing plots onto graph
    df['Condition'] = pd.Categorical(df['Condition'], conditions_to_include)
    df.sort_values(by='Condition', inplace=True, ascending=True)

    bootstrap_diff_df = bootstrap_sample_df(df,factor,ctl_label)

    # Use Matplotlib to create subplot and set some properties
    fig_width = 11 # Inches
    aspect = 2

    fig, axes = plt.subplots(1, 2, figsize=(fig_width,fig_width/aspect))
    # plt.rcParams['savefig.facecolor'] = 'w'
    fig.suptitle('Plots of Differences')

    # Resize points based on number of samples to reduce overplotting.
    if(len(df) > 1000):
        pt_size = 1
    else:
        pt_size = 5

    #sns.swarmplot(ax=axes[0], x=factor, y="Condition",size=2, data=df)#, ax=g.ax) # Built with sns.swarmplot (no ci arg.)
    sns.stripplot(ax=axes[0], x=factor, y="Condition",size=pt_size,jitter=0.25, data=df)

    # Draw confidence intervalswith point plot onto scatter plot
    sns.pointplot(ax=axes[0], x=factor, y="Condition", data=df,color='black', ci = 95, join=False, errwidth=10.0) #calamansi kind="swarm",

    # Right subplot: differences

    sns.violinplot(ax=axes[1], x="Difference", y="Condition",kind="violin", inner='box', data=bootstrap_diff_df, split=True, ci = 'sd',linewidth=2)
    axes[1].axvline(0, ls='-', color='black')
    axes[1].set(ylabel=None)
    axes[1].set(yticklabels=[])

    # Invert both y axis to be consistent with original plots of difference
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()


    return fig

