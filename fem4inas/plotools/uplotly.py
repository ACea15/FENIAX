from dataclasses import dataclass, field
import plotly.express as px
import plotly.graph_objects as go
import fem4inas.plotools.utils as putils
import numpy as np 
from enum import Enum
# see https://plotly.com/python/line-charts/

@dataclass
class ScatterSettings:

    mode: str = 'lines'  # 'lines+markers'
    name: str = None
    line: dict = field(default_factory=lambda:{})  #dict(color=colors[i], width=line_size[i])
    marker: dict = field(default_factory=lambda:{})  #dict(color=colors[i], size=mode_size[i])
    connectgaps: bool = False 
    line_shape:str ='spline'  # linear, spline, hv, vh
    
@dataclass
class UpdateLayout:

    xaxis: dict = field(default_factory=lambda:{})
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
    yaxis: dict = field(default_factory=lambda:{})
    autosize: bool = True,
    margin: dict = field(default_factory=lambda:{})
    # margin=dict(
    #     autoexpand=False,
    #     l=100,
    #     r=20,
    #     t=110,
    # )
    showlegend: bool = True,
    legend: dict = field(default_factory=lambda:{})
    # legend=dict(y=0.5, traceorder='reversed', font_size=16)
    plot_bgcolor: str ='white'
    annotations: list = field(default_factory=lambda:[])
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
        fig.update_traces(**update_traces)
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
        fig.update_traces(**update_traces)

    return fig

def iterate_lines2d(x, y,
                    scatter_settings=None,
                    update_layout=None,
                    update_traces=None,
                    fig=None):
    if scatter_settings is None:
        scatter_settings = [{} for i in x]
    elif isinstance(scatter_settings, dict):
        scatter_settings = [scatter_settings for i in x]
    for i, xi in enumerate(x):
        if i < len(x) - 1:
            fig = lines2d(xi, y[i],
                          fig=fig,
                          scatter_settings=scatter_settings[i])
        else:
            fig = lines2d(xi, y[i],
                          fig=fig,
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

def render2d_struct(structcomp: putils.IntrinsicStructComponent,
                    label=None,
                    scatter_settings=None,
                    update_layout=None,
                    update_traces=None,
                    fig=None,
                    xaxis=0,
                    yaxis=1):

    if label is None:
        structure = list(structcomp.map_components.values())[0]
    else:
        structure = structcomp.map_components[label]
    # components = list(structcomp.map_components.keys())
    num_components = len(structure)
    if scatter_settings is None:
        scatter_settings = {}
    #import pdb; pdb.set_trace()
    for i, v in enumerate(structure):
        if isinstance(scatter_settings, list):
            scsettings = scatter_settings[i]
        elif isinstance(scatter_settings, dict):
            scsettings = scatter_settings
        if i < num_components - 1:
            fig = lines2d(v[:, xaxis], v[:, yaxis],
                          fig=fig,
                          scatter_settings=scsettings)
        else:
            fig = lines2d(v[:, xaxis], v[:, yaxis],
                          fig=fig,
                          scatter_settings=scsettings,
                          update_layout=update_layout,
                          update_traces=update_traces)
    return fig

def render3d_struct(structcomp: putils.IntrinsicStructComponent,
                    label=None,
                    scatter_settings=None,
                    update_layout=None,
                    update_traces=None,
                    fig=None):

    if label is None:
        structure = list(structcomp.map_components.values())[0]
    else:
        structure = structcomp.map_components[label]
    # components = list(structcomp.map_components.keys())
    num_components = len(structure)
    if scatter_settings is None:
        scatter_settings = {}
    # import pdb; pdb.set_trace()
    for i, v in enumerate(structure):
        if isinstance(scatter_settings, list):
            scsettings = scatter_settings[i]
        elif isinstance(scatter_settings, dict):
            scsettings = scatter_settings
        if i < num_components - 1:
            fig = lines3d(v[:, 0], v[:, 1], v[:, 2],
                          fig=fig,
                          scatter_settings=scsettings)
        else:
            fig = lines3d(v[:, 0], v[:, 1], v[:, 2],
                          fig=fig,
                          scatter_settings=scsettings,
                          update_layout=update_layout,
                          update_traces=update_traces)
    return fig

def render3d_multi(structcomp: putils.IntrinsicStructComponent,
                   labels=None,
                   scatter_settings=None,
                   update_layout=None,
                   update_traces=None,
                   fig=None):

    if labels is None:
        labels = list(structcomp.map_components.keys())
    for i, li in enumerate(labels):
        if isinstance(scatter_settings, list):
            scsettings = scatter_settings[i]
        else:
            scsettings = scatter_settings
        fig = render3d_struct(structcomp,
                              li,
                              scsettings,
                              update_layout,
                              update_traces,
                              fig)
    return fig

def render2d_multi(structcomp: putils.IntrinsicStructComponent,
                   labels=None,
                   scatter_settings=None,
                   update_layout=None,
                   update_traces=None,
                   fig=None,
                   xaxis=0,
                   yaxis=1):

    if labels is None:
        labels = list(structcomp.map_components.keys())
    for i, li in enumerate(labels):
        if isinstance(scatter_settings, list):
            scsettings = scatter_settings[i]
        else:
            scsettings = scatter_settings
        fig = render2d_struct(structcomp,
                              li,
                              scsettings,
                              update_layout,
                              update_traces,
                              fig,
                              xaxis,
                              yaxis)
    return fig
