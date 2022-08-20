import plotly
import igraph as ig
import numpy as np
import ast
from  chart_studio.plotly import iplot
import plotly.graph_objs  as go
import plotly.io as pio


def get_Plotly_netw(G, graph_layout,  title='Plotly Interactive Network', flip='lr', width=950, height=850):
    #G is an igraph.Graph instance
    #graph_layout is an igraph.Layout instance
    #title - the network title
    #flip is one of the strings 'lr', 'ud' to perform a pseudo-flip effect
    #the igraph.Layout is referenced to the screen system of coords, and is  upside-down flipped chonging the sign of y-coords
    #the global HDN looks better with the left-right  flipped layout, by changing the x-coords sign.
    #width and height are the sizes of the plot area
    
    graph_layout=np.array(graph_layout)
    
    if flip =='lr':
        graph_layout[:, 0] = -graph_layout[:,0]
    elif flip == 'ud':
        graph_layout[:, 1] = -graph_layout[:,1] 
    else: 
        raise ValueError('There is no such a flip type')
        
    m = len(G.vs)
    graph_edges = [e.tuple for e in G.es]# represent edges as tuples of end vertex indices
    
    Xn = [graph_layout[k][0] for k in range(m)]#x-coordinates of graph nodes(vertices)
    Yn = [graph_layout[k][1] for k in range(m)]#y-coordinates -"-
        
    Xe = []#list of edge ends x-coordinates
    Ye = []#list of edge ends y-coordinates
    for e in graph_edges:
        Xe.extend([graph_layout[e[0]][0],graph_layout[e[1]][0], None])
        Ye.extend([graph_layout[e[0]][1],graph_layout[e[1]][1], None]) 

    size = [10 for vertex in G.vs]#[vertex['size'] for vertex in G.vs]
    #define the Plotly graphical objects
    
    plotly_edges = go.Scatter(x=Xe,
                              y=Ye,
                              mode='lines',
                              line=dict(color='rgb(105,105,105)', width=0.2),
                              text=[f"{edge['weight']}" for edge in G.es],
                              hoverinfo='text'
                       )
    plotly_vertices = go.Scatter(x=Xn,
                                 y=Yn,
                                 mode='markers',
                                 name='',
                                 marker=dict(symbol='circle-dot',
                                             size=size, 
                                             color=[vertex['color'] for vertex in G.vs], 
                                                    line=dict(color='rgb(50,50,50)', width=0.5)
                                                   ),
                                text=[f"{vertex['state']}<br>{vertex['party']}<br>{vertex['name']}" for vertex in G.vs],
                                hoverinfo='text'
                   )
    
    #Define the Plotly plot layout:
    
    plotly_plot_layout = dict(title=title, 
                              width=width,
                              height=height,
                              showlegend=False,
                              xaxis=dict(visible=False),
                              yaxis=dict(visible=False), 
         
                              margin=dict(t=100),
                              hovermode='closest',
                              template='none',
                              paper_bgcolor='rgb(235,235,235)'
                                    )
    return go.FigureWidget(data=[plotly_edges, plotly_vertices], layout=plotly_plot_layout)  