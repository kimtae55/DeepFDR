import warnings
# Suppress specific warnings from a particular module
warnings.filterwarnings("ignore",  category=Warning, module='dash_slicer')
warnings.filterwarnings("ignore",  category=Warning, module='VolumeSlicer')

import os
import plotly.graph_objects as go
import dash
from dash import html
from dash import dcc
from dash_slicer import VolumeSlicer
from dash.dependencies import Input, Output, State, ALL
from skimage.measure import marching_cubes
import imageio
import numpy as np
import plotly.express as px
import multiprocessing
from multiprocessing import Queue, Event
import train_gui_helper
import train 
from flask import request
import signal
from dash_core_components import Graph, Slider, Store, Interval
import argparse
from configure import Config
import logging
import time

def setup_dash():
    # Read volumes and create slicer objects
    vol = np.random.rand(30,30,30)

    slicer0 = VolumeSlicer(app, vol, axis=0)
    slicer1 = VolumeSlicer(app, vol, axis=1)
    slicer2 = VolumeSlicer(app, vol, axis=2)

    your_2d_plot1_figure = px.line(x=[None], y=[None], title="Loss function monitor", width=400, height=200, markers=True)
    your_2d_plot2_figure = px.line(x=[None], y=[None], title="FDP overestimation monitor", width=400, height=200, markers=True)
    your_2d_plot1_figure.update_layout(
        margin=dict(l=30, r=30, t=30, b=30) # Adjust margins as needed
    )
    your_2d_plot2_figure.update_layout(
        margin=dict(l=30, r=30, t=30, b=30)  # Adjust margins as needed
    )

    try:
        # Calculate isosurface and create a figure with a mesh object
        verts, faces, _, _ = marching_cubes(vol)
        x, y, z = verts.T
        i, j, k = faces.T
        mesh = go.Mesh3d(x=z, y=y, z=x, opacity=0.5, i=k, j=j, k=i)
        fig = go.Figure(data=[mesh])
        fig.update_layout(uirevision="anything", width=400, height=400)  # prevent orientation reset on update
    except Exception as e:
        print('exception: failed to get isosurface')
        # Create an empty figure or a figure with a placeholder
        fig = go.Figure(data=[go.Mesh3d()])
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers',
                                   marker=dict(size=10, color='red'),
                                   text=["No data available"],
                                   hoverinfo='text'))
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),  # Reduces the margin on all sides
        # Adjust layout properties as needed
    )

    def create_first_row_layout():
        # This function creates the content of the first row
        return [
            # First column with the 3D graph
            html.Div(
                [
                    html.Center(html.H2("3D Isosurface")),
                    dcc.Graph(id="3Dgraph", figure=fig),  # fig should be a defined Plotly figure
                ],
                style={"gridColumn": "1", "gridRow": "1"}  # Occupies the first column, first row
            ),
            html.Div(
                [
                    # Upper plot (assume this should take up more space, e.g., 60% of the grid cell)
                    html.Div(
                        dcc.Graph(id="2Dplot1", figure=your_2d_plot1_figure),
                        style={"height": "50%"}
                    ),
                    # Lower plot (assume this should take up less space, e.g., 40% of the grid cell)
                    html.Div(
                        dcc.Graph(id="2Dplot2", figure=your_2d_plot2_figure),
                        style={"height": "50%"}
                    ),
                ],
                style={"gridColumn": "2", "gridRow": "1", "height": "100%"}  # Ensure the container fills the row
            ),
            # Third column with the square area
            html.Div(
                [
                    html.Center(html.H2("DeepFDR Training Tool")),
                    html.Form(
                        [
                            html.Button("Continue Epoch", id="continue_button"),
                            html.Button("Save Epoch", id="save_button"),
                            html.Button("Save and Exit", id="save_exit_button"),
                        ],
                        style={
                            "display": "flex",
                            "flexDirection": "column",  # Stack buttons vertically
                            "alignItems": "center",  # Center items horizontally
                            "gap": "10px",  # Space between items
                        }
                    )
                ],
                style={
                    "gridColumn": "3", 
                    "gridRow": "1",
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "center",  # Center items horizontally
                    "backgroundColor": "#f0f0f0",
                    "alignItems": "center",
                    "height": "100%",  # Adjust height to fill the grid area
                    "gap": "10px",  # Space between items
                }  # Occupies the third column, first row
            ),
        ]


    def create_second_row_layout():
        return [
            html.Div(
                [
                    html.Center(html.H2("Axial")),
                    slicer0.graph,
                    html.Br(),
                    html.Div(slicer0.slider, style={"display": "none"}),
                    *slicer0.stores,
                ],
                style={
                "gridColumn": "1", 
                "gridRow": "2",
                "maxWidth": "70%",  # Limit the maximum width
                "maxHeight": "70%",  # Limit the maximum height
                },
                id='slicer0-div',
            ), html.Div(
                [
                    html.Center(html.H2("Coronal")),
                    slicer1.graph,
                    html.Br(),
                    html.Div(slicer1.slider, style={"display": "none"}),
                    *slicer1.stores,
                ],
                style={
                "gridColumn": "2", 
                "gridRow": "2",
                "maxWidth": "70%",  # Limit the maximum width
                "maxHeight": "70%",  # Limit the maximum height
                },            
                id='slicer1-div',
            ), 
            html.Div(
                [
                    html.Center(html.H2("Sagittal")),
                    slicer2.graph,
                    html.Br(),
                    html.Div(slicer2.slider, style={"display": "none"}),
                    *slicer2.stores,
                ],
                style={
                "gridColumn": "3", 
                "gridRow": "2",
                "maxWidth": "70%",  # Limit the maximum width
                "maxHeight": "70%",  # Limit the maximum height
                },            
                id='slicer2-div',
            ), 
        ]


    # Combine the components in the layout
    app.layout = html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr 1fr",  # Three columns, each taking one fraction of the grid's width
            "gridTemplateRows": "1fr 1fr",       # Two rows, first takes 40vh, second takes 60vh of the viewport height
            "height": "calc(100vh - 100px)",       # Subtracting top and bottom padding from the viewport height
            "box-sizing": "border-box",            # Padding and border are included in the width and height of the element
            "margin": "0", 
            "padding": "50px",  # Adding padding inside the grid
            "font-family": "'Helvetica Neue', sans-serif",
        },
        children=create_first_row_layout() 
                + create_second_row_layout() + [html.Div(id='dummy-div', style={'display': 'none'}),  # Add the hidden div
                   dcc.Interval(id='interval_updater', interval=1000, n_intervals=0)]  # Add the interval component
    )

    # Callback to display slicer view positions in the 3D view
    app.clientside_callback(
        """
    function update_3d_figure(states, ori_figure) {
        let traces = [ori_figure.data[0]]
        for (let state of states) {
            if (!state) continue;
            let xrange = state.xrange;
            let yrange = state.yrange;
            let xyz = [
                [xrange[0], xrange[1], xrange[1], xrange[0], xrange[0]],
                [yrange[0], yrange[0], yrange[1], yrange[1], yrange[0]],
                [state.zpos, state.zpos, state.zpos, state.zpos, state.zpos]
            ];
            xyz.splice(2 - state.axis, 0, xyz.pop());
            let s = {
                type: 'scatter3d',
                x: xyz[0], y: xyz[1], z: xyz[2],
                mode: 'lines', line: {color: state.color},
                hoverinfo: 'skip',
                showlegend: false,
            };
            traces.push(s);
        }
        let figure = {...ori_figure};
        figure.data = traces;
        return figure;
    }
        """,
        Output("3Dgraph", "figure"),
        [Input({"scene": slicer0.scene_id, "context": ALL, "name": "state"}, "data")],
        [State("3Dgraph", "figure")],
    )

    # Define a callback to update the training flag when the button is clicked
    @app.callback(
        Output("continue_button", "n_clicks"),
        Input("continue_button", "n_clicks"),
        prevent_initial_call=True
    )
    def continue_training(n_clicks):    
        # Triggered when the "Train one more epoch" button is clicked
        continue_training_flag.set()
        return n_clicks

    @app.callback(
        Output("dummy-div", "children"),  # This can be a dummy output
        Input("save_button", "n_clicks"),
        prevent_initial_call=True
    )
    def save(n_clicks):
        # Signal to save and exit
        save_flag.set()
        return "Saving..."

    @app.callback(
        Output("dummy-div", "children", allow_duplicate=True),  # This can be a dummy output
        Input("save_exit_button", "n_clicks"),
        prevent_initial_call=True
    )
    def save_and_exit(n_clicks):
        # Signal to save and exit
        save_exit_flag.set()
        # TODO: for now just wait couple seconds for train_process to completely finish, change this to make it exit more gracefully
        while save_exit_flag.is_set():
            time.sleep(0.5)
        stop_server()
        print('stopped server')
        return "Exiting..."

    def stop_server():
        # wait until train is finished 
        time.sleep(2)
        os._exit(0)

    def create_div_for_VS(slicer, sid, vtype='Axial'):
        return html.Div(
                    [
                        html.Center(html.H2(vtype)),
                        slicer.graph,
                        html.Br(),
                        html.Div(slicer.slider, style={"display": "none"}),
                        *slicer.stores,
                    ],
                    style={"width": "100%", "float": "left"},
                    id='slicer'+ str(sid) + '-div',
                )

    @app.callback(
        [
            Output("slicer0-div", "children"),
            Output("slicer1-div", "children"),
            Output("slicer2-div", "children"),
            Output("3Dgraph", "figure", allow_duplicate=True),
            Output("2Dplot1", "figure"),
            Output("2Dplot2", "figure"),
        ],
        [Input('interval_updater', 'n_intervals')],
        prevent_initial_call=True
    )
    def update_visualization(n_epoch):
        if n_epoch is not None:
            # Implement your logic to update the 2D plots here based on the training progress (e.g., using data from the queue)
            # Replace this with your actual visualization update code
            if q.empty():
                return (dash.no_update,) * 6

            plot1_y = q.get_nowait()
            plot2_y = q.get_nowait()
            vol = q.get_nowait()
            vol = vol.reshape(Config.inputsize)

            updated_2d_plot1_figure = px.line(x=list(range(1, len(plot1_y) + 1)), y=plot1_y, title="Loss function monitor", width=400, height=200, markers=True)
            updated_2d_plot2_figure = px.line(x=list(range(1, len(plot1_y) + 1)), y=plot2_y, title="FDP overestimation monitor", width=400, height=200, markers=True)

            # Create the graph (graph is a Dash component wrapping a Plotly figure) 
            try:
                slicer0._volume = vol
                slicer1._volume = vol
                slicer2._volume = vol

                new_div0 = create_div_for_VS(slicer0, sid=0, vtype='Axial')
                new_div1 = create_div_for_VS(slicer1, sid=1, vtype='Coronal')
                new_div2 = create_div_for_VS(slicer2, sid=2, vtype='Sagittal')
                
                # Calculate isosurface and create a figure with a mesh object
                verts, faces, _, _ = marching_cubes(vol)
                x, y, z = verts.T
                i, j, k = faces.T
                mesh = go.Mesh3d(x=z, y=y, z=x, opacity=0.5, i=k, j=j, k=i)
                fig = go.Figure(data=[mesh])
                fig.update_layout(uirevision="anything")  # prevent orientation reset on update
                
                your_2d_plot1_figure = updated_2d_plot1_figure
                your_2d_plot2_figure = updated_2d_plot2_figure

                your_2d_plot1_figure.update_layout(
                    margin=dict(l=30, r=30, t=30, b=30) # Adjust margins as needed
                )
                your_2d_plot2_figure.update_layout(
                    margin=dict(l=30, r=30, t=30, b=30)  # Adjust margins as needed
                )
                fig.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),  # Reduces the margin on all sides
                    # Adjust layout properties as needed
                )

            except Exception as e:
                print('failed creating new slicers')

        return new_div0, new_div1, new_div2, fig, your_2d_plot1_figure, your_2d_plot2_figure

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepFDR using W-NET')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--l2', default=1e-5, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--labelpath', default='./', type=str)
    parser.add_argument('--savepath', default='./', type=str)
    parser.add_argument('--loss', default='pv_mse', type=str) # x_mse or pv_mse
    parser.add_argument('--mode', default=2, type=int) # 0 for train, 1 for inference, 2 for both
    _args = parser.parse_args()

    q = Queue()
    continue_training_flag = Event()
    save_exit_flag = Event()
    save_flag = Event()

    # Create and start the training process
    train_process = multiprocessing.Process(target=train_gui_helper.train, args=(q,continue_training_flag, save_flag, save_exit_flag, _args))
    train_process.start()

    logger = logging.getLogger('werkzeug')  # Werkzeug is the underlying server for Dash
    logger.setLevel(logging.ERROR)
    app = dash.Dash(__name__, update_title=None)
    server = app.server

    setup_dash()
    app.run_server(debug=False)


