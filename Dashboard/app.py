# IMPORT LIBRARIES
import warnings
warnings.filterwarnings("ignore")

import datetime as dt
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
pd.set_option('chained_assignment', None)

import plotly.express as px
import plotly.graph_objects as go

import dash_auth, dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# START APP
app = dash.Dash(__name__, external_stylesheets = [dbc.themes.LITERA],
                meta_tags = [{'name': 'viewport',
                'content': 'width=device-width, initial-scale = 1.0'}]
)

app.title = 'MATH 231.4 Dashboard'
server = app.server

# LOAD DATASETS
today = dt.date(2020,5,23).strftime('%Y%m%d')
client = pd.read_csv('dim_client.csv')
repayment = pd.read_csv('fact_repayment.csv')

def case_when(input):
    if len(input) == 1:
        return input[0]
    else:
        return np.where(input[0], input[1], case_when(input[2:]))

client = (client
    .assign(**{
        'company_type' : lambda df: df['company_type'].fillna("NA"),
        'company_size' : lambda df: pd.Categorical(case_when([
                df['company_size'] == "1-10", "1 - 10",
                df['company_size'] == "less than 10", "1 - 10",
                df['company_size'] == "21-30", "10 - 49",
                df['company_size'].isnull(), "NA",
                df['company_size']
            ]), ordered=True,
                categories = ['1 - 10', '10 - 49', '50 - 99',
                            '100 - 199', 'more than 200', 'NA']
        )
    })
)

repayment = (repayment
    .assign(**{
        'start_date' : lambda df: pd.to_datetime(df['start_date']),
        'original_due_date' : lambda df: pd.to_datetime(df['original_due_date']),
        'current_due_date' : lambda df: pd.to_datetime(df['current_due_date']),
        'is_due' : lambda df: df['current_due_date'] <= today,
        'is_late' : lambda df: df['current_due_date'] < df['paid_date'],
        'deleted_date' : lambda df: pd.to_datetime(df['deleted_date']),
        'paid_date' : lambda df: pd.to_datetime(df['paid_date']),
        'paid_principal' : lambda df: np.where(df['paid_principal'].isnull() | df['deleted_date'].notnull(), 0, df['paid_principal']),
        'paid_interest' : lambda df: np.where(df['paid_interest'].isnull() | df['deleted_date'].notnull(), 0, df['paid_interest']),
        'outstanding_principal' : lambda df: df['current_principal'] - df['paid_principal'],
        'outstanding_interest' : lambda df: df['current_interest'] - df['paid_interest'],
        'total_outstanding' : lambda df: df['outstanding_principal'] + df['outstanding_interest']
    })
)

data = (repayment
    .merge(client, on = 'client_uuid', how = "inner")
    .assign(**{
        "Account Status" : lambda df: case_when([
            df['is_due'] & (df['total_outstanding'] <= 0) & df['is_late'], "Late",
            df['is_due'] & (df['total_outstanding'] <= 0), "On time",
            df['is_due'] & (df['total_outstanding'] > 0), "Late",
            ~ df['is_due'] & df['total_outstanding'] <= 0, "Early",
            "On time"
        ]),
        'Start Month' : lambda df: df['start_date'].dt.strftime('%Y-%m-01')
    })
)

# CARD
controls = dbc.Card([
    html.H5("Select Inputs", style = {'textAlign' : 'center'}),
    dbc.Row([
        dbc.Label("Select Manager"),
        dbc.RadioItems(
            id = 'manager',
            options = [{'label':'Sales Manager (Metrics)', 'value':'Sales Manager Metrics'}, 
                    {'label':'Sales Manager (Clients)', 'value':'Sales Manager Clients'},
                    {'label':'Risk Manager (Status)', 'value':'Risk Manager Stat'},
                    {'label':'Risk Manager (Interest Rate)', 'value':'Risk Manager Int'}],
            value = 'Sales Manager Metrics'
        )
    ]),
    html.Br(),

    dbc.Row([
        dbc.Label("Select Graph"),
        dbc.Select(
            id = "graph-input"
        ),
    ]),
    html.Br(),

    dbc.Row([
        dbc.Label("Select Value"),
        dbc.Select(
            id = "value"
        ),
    ])
], body = True)

# LAYOUT
app.layout = dbc.Container([
    html.Br(),
    html.H3("MATH 231.4 Final Project Dashboard", style = {'textAlign': 'center'}),
    html.H6("Brian Godwin Lim, Terence Brian Tsai", style = {'textAlign': 'center'}),
    html.Hr(),
    dbc.Row([
        dbc.Col(controls, md = 4),
        dbc.Col(dcc.Graph(id = "graph"), md = 8)
    ], align = 'center')
], fluid = True)    

# UPDATE OPTIONS
@app.callback(
    Output('graph-input', 'options'),
    Output('graph-input', 'value'),
    Input('manager', 'value')
)
def update_options(manager):
    if manager == "Sales Manager Metrics":
        outputss = ["Total Origination", "Origination per Client", "Average Annual Interest Rate", "Average Duration"]
    elif manager == "Sales Manager Clients":
        outputss = ["Company Type", "Industry", "Company Size", "Times Rescheduled", "New and Existing"]
    elif manager == "Risk Manager Stat":
        outputss = ["Overall", "Company Type", "Industry", "Company Size", "Times Rescheduled"]
    elif manager == "Risk Manager Int":
        outputss = ["Overall", "Company Type", "Industry", "Company Size", "Times Rescheduled"]
    return [{"label": col, "value": col} for col in outputss], outputss[0]

# UPDATE VALUES
@app.callback(
    Output('value', 'options'),
    Output('value', 'value'),
    Input('manager', 'value'),
    Input('graph-input', 'value')
)
def update_values(manager, graph_input):    
    if manager == "Sales Manager Metrics":
        if graph_input in {"Average Annual Interest Rate", "Average Duration"}:
            outputss = ["Standard Average", "Weighted Average"]
        else:
            outputss = ["Amount", "Count"]
    
    elif manager == "Sales Manager Clients":
        outputss = ["Percent", "Actual"]

    elif manager == "Risk Manager Stat":
        outputss = ["Amount (Percent)", "Count (Percent)", "Amount (Actual)", "Count (Actual)"]
    
    elif manager == "Risk Manager Int":
        outputss = ["Standard Average", "Weighted Average"]

    return [{"label": col, "value": col} for col in outputss], outputss[0]

# UPDATE GRAPH
@app.callback(
    Output('graph', 'figure'),
    Input('graph-input', 'value'),
    Input('value', 'value'),
    Input('manager', 'value')
)
def update_graph(graph_input, value, manager):
    if manager == "Sales Manager Metrics":
        operation = 'sum' if value == "Amount" else 'size'

        if graph_input == "Total Origination":  
            df = (data
                .groupby('Start Month')
                .agg(**{
                    "Total Origination" : ('current_principal', operation)
                })
                .reset_index()
            )

            labelss = "Total Origination" if value == "Amount" else "Total Number of Loans"
            fig = px.bar(df, x = "Start Month", y = "Total Origination", 
                template = 'plotly_white', labels = {'Total Origination' : labelss})

        elif graph_input == "Origination per Client": 
            df = (data
                .groupby(['Start Month', 'client_uuid'])
                .agg(**{
                    "Total Origination" : ('current_principal', operation)
                })
                .groupby('Start Month')
                .agg(**{
                    "Average Origination" : ('Total Origination', 'mean')
                })
                .reset_index()
            )

            labelss = "Average Number of Loan" if value == "Count" else "Average Origination"
            fig = px.line(df, x = "Start Month", y = "Average Origination", 
                template = 'plotly_white', labels = {'Average Origination' : labelss})

        elif graph_input == "Average Annual Interest Rate":
            df = (data
                .assign(**{
                    "Duration" : lambda df: (df['current_due_date'] - df['start_date']).dt.days / 365,
                    "Weight" : lambda df: df['current_principal'] if value == "Weighted Average" else 1,
                    "Weighted Interest" : lambda df: np.where(df['Duration'] != 0, 
                        df['Weight'] * df['current_interest'] / (df['current_principal'] * df['Duration']), 0)
                })
                .groupby('Start Month')
                .agg(**{
                    "Total Weight" : ('Weight', 'sum'),
                    "Total Weighted Interest" : ('Weighted Interest', 'sum')
                })
                .assign(**{
                    "Average Annual Interest Rate" : lambda df: 100 * df['Total Weighted Interest'] / df['Total Weight']
                })
                .reset_index()
            )

            fig = px.line(df, x = "Start Month", y = "Average Annual Interest Rate", template = 'plotly_white')

        elif graph_input == "Average Duration":
            df = (data
                .assign(**{
                    "Weight" : lambda df: df['current_principal'] if value == "Weighted Average" else 1,
                    "Weighted Duration" : lambda df: df['Weight'] * (df['current_due_date'] - df['start_date']).dt.days
                })
                .groupby('Start Month')
                .agg(**{
                    "Total Weight" : ('Weight', 'sum'),
                    'Total Weighted Duration' : ('Weighted Duration', 'sum')
                })
                .assign(**{
                    'Average Duration (days)' : lambda df: df['Total Weighted Duration'] / df['Total Weight']
                })
                .reset_index()
            )
            
            fig = px.bar(df, x = "Start Month", y = "Average Duration (days)", template = "plotly_white")

    elif manager == "Sales Manager Clients":
        group = "_".join(graph_input.lower().split(" "))

        if graph_input == "New and Existing":
            df = (data
                .set_index('client_uuid')
                .assign(**{
                    "First": lambda df: df['Start Month'].groupby('client_uuid').min(),
                    "New": lambda df: np.where(df['First'] == df['Start Month'], 1, 0)
                })
                .groupby(["client_uuid", 'Start Month'])
                .agg(**{
                    "New": ('New', 'max')
                })
                .groupby('Start Month')
                .agg(**{
                    "New": ('New', 'sum'),
                    "Total": ('New', 'count')
                })
                .assign(**{
                    "Existing": lambda df: df['Total'] - df['New']
                })
                .drop("Total", axis = 1)
                .reset_index()
                .melt(id_vars = "Start Month", var_name = "Type of Client", value_name = "Number of Clients")
                .set_index("Start Month")
                .assign(**{
                    "Total Clients": lambda df: df['Number of Clients'].groupby('Start Month').sum(),
                    "Percent": lambda df: 100 * df['Number of Clients'] / df['Total Clients']
                })
                .reset_index()
            )
            yaxis = "Percent" if value == "Percent" else "Number of Clients"
            
            fig = px.bar(df, x = "Start Month", y = yaxis, 
                color = "Type of Client", template = "plotly_white")

        elif graph_input == "Times Rescheduled":
            df = (data
                .sort_values('Start Month')
                .groupby(["Start Month", "client_uuid"])
                .agg(**{
                    "times_rescheduled": ('times_rescheduled', 'max')
                })
                .reset_index()
                .groupby(['Start Month', group])
                .agg(**{
                    "Number of Clients" : ('client_uuid', 'nunique')
                })
                .assign(**{
                    "Percent" : lambda df: 100 * df['Number of Clients'] / df['Number of Clients'].groupby("Start Month").sum()
                })
                .sort_values(group)
                .reset_index()
            )
            yaxis = "Percent" if value == "Percent" else "Number of Clients"

            fig = px.bar(df, x = "Start Month", y = yaxis, color = group,
                template = "plotly_white", labels = {group : graph_input})

        else:
            df = (data
                .sort_values('Start Month')
                .groupby(['Start Month', group])
                .agg(**{
                    "Number of Clients" : ('client_uuid', 'nunique')
                })
                .assign(**{
                    "Percent" : lambda df: 100 * df['Number of Clients'] / df['Number of Clients'].groupby("Start Month").sum()
                })
                .sort_values(group)
                .reset_index()
            )
            yaxis = "Percent" if value == "Percent" else "Number of Clients"

            fig = px.bar(df, x = "Start Month", y = yaxis, color = group,
                template = "plotly_white", labels = {group : graph_input})

    elif manager == "Risk Manager Stat":

        if graph_input == "Overall":
            operation = 'sum' if "Amount" in value else 'size'
            keyword = "Amount" if "Amount" in value else "Count"
            yaxis = f"Total {keyword}" if "Percent" not in value else "Percent"

            df = (data
                .groupby(['Account Status'])
                .agg(**{
                    f"Total {keyword}" : ('current_principal', operation)
                })
                .assign(**{
                    "Percent" : lambda df: 100 * df[f"Total {keyword}"] / df[f"Total {keyword}"].sum()
                })
                .reset_index()
            )

            fig = px.bar(df, x = "Account Status", y = yaxis, color = "Account Status", 
                template = 'plotly_white', 
                color_discrete_map = {'Late' : "#EF553B", "On time" : "#00CC96"})

        else:
            group = "_".join(graph_input.lower().split(" "))
            operation = 'sum' if "Amount" in value else 'size'
            keyword = "Amount" if "Amount" in value else "Count"
            yaxis = f"Total {keyword}" if "Percent" not in value else "Percent"
            
            df = (data
                .groupby(['Account Status', group])
                .agg(**{
                    f"Total {keyword}" : ('current_principal', operation)
                })
                .assign(**{
                    "Percent" : lambda df: 100 * df[f"Total {keyword}"] / df.groupby(group)[f"Total {keyword}"].sum()
                })
                .reset_index()
                .sort_values(["Account Status", yaxis])
            )

            fig  = px.bar(df, x = group, y = yaxis, color = "Account Status", 
                template = 'plotly_white', labels = {group : graph_input},
                color_discrete_map = {'Late' : "#EF553B", "On time" : "#00CC96"})
            
            if group == "industry":
                fig.update_layout(xaxis = dict(tickfont = dict(size = 8)))
    
    elif manager == "Risk Manager Int":

        if graph_input == "Overall":
            df = (data
                .assign(**{
                    "Duration" : lambda df: (df['current_due_date'] - df['start_date']).dt.days / 365,
                    "Weight" : lambda df: df['current_principal'] if value == "Weighted Average" else 1,
                    "Weighted Interest" : lambda df: np.where(df['Duration'] != 0, 
                        df['Weight'] * df['current_interest'] / (df['current_principal'] * df['Duration']), 0)
                })
                .filter(['Weight', "Weighted Interest"])
                .agg('sum')
            )

            fig = go.Figure(go.Indicator(
                value = 100 * df["Weighted Interest"] / df["Weight"],
                number = {'suffix': "%"},
                title = "Average Annual Interest Rate"
            ))

        else:
            group = "_".join(graph_input.lower().split(" "))

            df = (data
                .assign(**{
                    "Duration" : lambda df: (df['current_due_date'] - df['start_date']).dt.days / 365,
                    "Weight" : lambda df: df['current_principal'] if value == "Weighted Average" else 1,
                    "Weighted Interest" : lambda df: np.where(df['Duration'] != 0, 
                        df['Weight'] * df['current_interest'] / (df['current_principal'] * df['Duration']), 0)
                })
                .groupby(group)
                .agg(**{
                    "Total Weight" : ('Weight', 'sum'),
                    "Total Weighted Interest" : ('Weighted Interest', 'sum')
                })
                .assign(**{
                    "Average Annual Interest Rate" : lambda df: 100 * df['Total Weighted Interest'] / df['Total Weight']
                })
                .reset_index()
                .sort_values("Average Annual Interest Rate")
            )

            fig  = px.bar(df, x = group, y = "Average Annual Interest Rate", 
                template = 'plotly_white', labels = {group : graph_input})
            
            if group == "industry":
                fig.update_layout(xaxis = dict(tickfont = dict(size = 8)))

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
