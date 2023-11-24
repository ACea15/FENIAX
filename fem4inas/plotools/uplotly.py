from dataclasses import dataclass, field
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
import numpy as np 
from enum import Enum
import fem4inas.plotools.utils as utils
# see https://plotly.com/python/line-charts/

@dataclass
class ScatterSettings:

    mode: str = 'lines'  # 'lines+markers'
    name: str = None
    line: dict = dict()  #dict(color=colors[i], width=line_size[i])
    marker: dict = dict()  #dict(color=colors[i], size=mode_size[i])
    connectgaps: bool = False 
    line_shape: ='spline'  # linear, spline, hv, vh
    
@dataclass
class UpdateLayout:

    xaxis: dict = dict()
    # xaxis=dict(
    #     showline=True,
    #     showgrid=False,
    #     zeroline=True,
    #     showticklabels=True,
    #     linecolor='rgb(204, 204, 204)',
    #     linewidth=2,
    #     ticks='outside',
    #     tickfont=dict(
    #         family='Arial',
    #         size=12,
    #         color='rgb(82, 82, 82)'))
    yaxis: dict = dict()
    autosize: bool = True,
    margin: dict = dict()
    # margin=dict(
    #     autoexpand=False,
    #     l=100,
    #     r=20,
    #     t=110,
    # )
    showlegend: bool = True,
    legend: dict = dict()
    # legend=dict(y=0.5, traceorder='reversed', font_size=16)
    plot_bgcolor: str ='white'
    annotations: list = []
    # # Adding labels
    # for y_trace, label, color in zip(y_data, labels, colors):
    #     # labeling the left_side of the plot
    #     annotations.append(dict(xref='paper', x=0.05, y=y_trace[0],
    #                                   xanchor='right', yanchor='middle',
    #                                   text=label + ' {}%'.format(y_trace[0]),
    #                                   font=dict(family='Arial',
    #                                             size=16),
    #                                   showarrow=False))
    #     # labeling the right_side of the plot
    #     annotations.append(dict(xref='paper', x=0.95, y=y_trace[11],
    #                                   xanchor='left', yanchor='middle',
    #                                   text='{}%'.format(y_trace[11]),
    #                                   font=dict(family='Arial',
    #                                             size=16),
    #                                   showarrow=False))
    
def lines2d(x, y,
            fig=None,
            scatter_settings=None,
            update_layout=None,
            update_traces=None):

    if fig is None:
        fig = go.Figure()
    if scatter_settings is None:
        scatters = ScatterSettings()
        fig.add_trace(go.Scatter(x=x, y=y, **scatters.__dict__))        
    else:
        fig.add_trace(go.Scatter(x=x, y=y, **scatter_settings))
    if update_layout is not None:
        fig.update_layout(**update_layout)
    if update_traces is not None:
        scatters = ScatterSettings()
        fig.add_trace(go.Scatter(x=x, y=y, **scatters.__dict__))        
    else:
        fig.add_trace(go.Scatter(x=x, y=y, **update_traces))

    return fig

def lines3d(x, y, z,
            fig=None,
            scatter_settings=None,
            update_layout=None,
            update_traces=None):

    if fig is None:
        fig = go.Figure()
    if scatter_settings is None:
        scatters = ScatterSettings()
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, **scatters.__dict__))        
    else:
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, **scatter_settings))
    if update_layout is not None:
        fig.update_layout(**update_layout)
    if update_traces is not None:
        scatters = ScatterSettings()
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, **scatters.__dict__))        
    else:
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, **update_traces))

    return fig

def iterate_lines2d(x, y,
                    scatter_settings=None,
                    update_layout=None,
                    update_traces=None):

    fig= None
    for i, xi in enumerate(x):
        if i < len(x) - 1:
            fig = lines2d(xi, y[i],
                          fig=fig,
                          scatter_settings=scatter_settings[i])
        else:
            fig = lines2d(xi, y[i],
                          fig=None,
                          scatter_settings=scatter_settings[i],
                          update_layout=update_layout,
                          update_traces=update_traces)
    return fig

def iterate_lines3d(x, y, z,
                    scatter_settings=None,
                    update_layout=None,
                    update_traces=None):

    fig= None
    for i, xi in enumerate(x):
        if i < len(x) - 1:
            fig = lines2d(xi, y[i], z[i], 
                          fig=fig,
                          scatter_settings=scatter_settings[i])
        else:
            fig = lines2d(xi, y[i], z[i],
                          fig=None,
                          scatter_settings=scatter_settings[i],
                          update_layout=update_layout,
                          update_traces=update_traces)
    return fig

