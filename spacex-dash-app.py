# Import required libraries
import pandas as pd
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px

# Read the SpaceX data into pandas dataframe
spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Create a dash application
app = dash.Dash(__name__)

# Create an app layout
app.layout = html.Div(children=[
    html.H1(
        'SpaceX Launch Records Dashboard',
        style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}
    ),

    html.Br(),

    # TASK 1: Dropdown for Launch Site selection
    dcc.Dropdown(
        id='site-dropdown',
        options=[{'label': 'All Sites', 'value': 'ALL'}] +
                [{'label': site, 'value': site} for site in spacex_df['Launch Site'].unique()],
        value='ALL',
        placeholder='Select a Launch Site here',
        searchable=True
    ),

    html.Br(),

    # TASK 2: Pie chart for success count
    html.Div(dcc.Graph(id='success-pie-chart')),
    html.Br(),

    html.P("Payload range (Kg):"),

    # TASK 3: Range Slider to select payload
    dcc.RangeSlider(
        id='payload-slider',
        min=0,
        max=10000,
        step=1000,
        marks={i: str(i) for i in range(0, 10001, 1000)},
        value=[min_payload, max_payload]
    ),
    html.Br(),

    # TASK 4: Scatter chart for payload vs success
    html.Div(dcc.Graph(id='success-payload-scatter-chart'))
])

# TASK 2: Callback for updating pie chart
@app.callback(
    Output('success-pie-chart', 'figure'),
    Input('site-dropdown', 'value')
)
def update_pie_chart(selected_site):
    if selected_site == 'ALL':
        # Pie chart of total successes by site
        fig = px.pie(
            spacex_df,
            names='Launch Site',
            values='class',
            title='Total Success Launches by Site'
        )
    else:
        # Pie chart of success vs failure for selected site
        filtered_df = spacex_df[spacex_df['Launch Site'] == selected_site]
        fig = px.pie(
            filtered_df,
            names='class',  # 0 = fail, 1 = success
            title=f'Success vs Failure for site {selected_site}'
        )
    return fig

# TASK 4: Callback for updating scatter plot
@app.callback(
    Output('success-payload-scatter-chart', 'figure'),
    [
        Input('site-dropdown', 'value'),
        Input('payload-slider', 'value')
    ]
)
def update_scatter_chart(selected_site, payload_range):
    low, high = payload_range
    if selected_site == 'ALL':
        filtered_df = spacex_df[
            (spacex_df['Payload Mass (kg)'] >= low) &
            (spacex_df['Payload Mass (kg)'] <= high)
        ]
        fig = px.scatter(
            filtered_df,
            x='Payload Mass (kg)',
            y='class',
            color='Booster Version Category',
            title='Payload vs Outcome for All Sites'
        )
    else:
        filtered_df = spacex_df[
            (spacex_df['Launch Site'] == selected_site) &
            (spacex_df['Payload Mass (kg)'] >= low) &
            (spacex_df['Payload Mass (kg)'] <= high)
        ]
        fig = px.scatter(
            filtered_df,
            x='Payload Mass (kg)',
            y='class',
            color='Booster Version Category',
            title=f'Payload vs Outcome for site {selected_site}'
        )
    return fig

# Run the app
if __name__ == '__main__':
    app.run()
