#Libraries/Extensions
from dash import Dash, dcc, html, Input, Output, callback
import dash_leaflet as dl
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests
from datetime import datetime
#---------------------------------------------------------
#everything about styling
custom_icon = dict(
    iconUrl='https://images.emojiterra.com/google/noto-emoji/unicode-15/color/512px/1f535.png',
    iconSize=[10, 10]
)
#---------------------------------------------------------
#constants
latitude = 52.516181 #Breitengrad
longitude = 13.376935 #Längegrad
#---------------------------------------------------------
#dataframe all boxes in germany
coordinates = {"id": [],
                  "latitude": [],
                  "longitude": []
                }
df_coordinates = pd.DataFrame(coordinates)

url = "https://api.opensensemap.org/boxes?near=10.323007,51.16501&maxDistance=450000&exposure=outdoor&title=PM10,PM2.5&format=geojson"
data = requests.get(url).json()
for idx in range(0, len(data["features"])):
    df2 = {"id": data["features"][idx]["properties"]["_id"], "latitude": data["features"][idx]["geometry"]["coordinates"][1], "longitude": data["features"][idx]["geometry"]["coordinates"][0]}
    df_coordinates = df_coordinates._append(df2, ignore_index = True)
#---------------------------------------------------------
#Analyse von ausgewählten Boxen/Dataframes
df_land_pm10 = pd.read_csv("werte_daten/werte_land_PM10.csv")
df_land_pm2_5 = pd.read_csv("werte_daten/werte_land_PM2_5.csv")
df_city_pm10 = pd.read_csv("werte_daten/werte_stadt_PM10.csv")
df_city_pm2_5 = pd.read_csv("werte_daten/werte_stadt_PM2_5.csv")
#---------------------------------------------------------
#funnel: Wie viele verfügbare Daten sind vorhanden
value_columns = [col for col in df_land_pm10.columns if col.startswith('value_')]
# count()-Funktion auf gefilterte Spalten anwenden
value_counts = df_land_pm10[value_columns].count()
# Ergebnisse als DataFrame konvertieren und sortieren
value_counts_df = pd.DataFrame(value_counts, columns=['Boxen']).reset_index()
value_counts_df.columns = ['Boxen', 'Count']
sorted_value_counts_df_land10 = value_counts_df.sort_values(by='Count', ascending=False)

value_columns = [col for col in df_city_pm10.columns if col.startswith('value_')]
# count()-Funktion auf gefilterte Spalten anwenden
value_counts = df_city_pm10[value_columns].count()
# Ergebnisse als DataFrame konvertieren und sortieren
value_counts_df = pd.DataFrame(value_counts, columns=['Boxen']).reset_index()
value_counts_df.columns = ['Boxen', 'Count']
sorted_value_counts_df_city10 = value_counts_df.sort_values(by='Count', ascending=False)
#---------------------------------------------------------
#static figures
#funnel: analyzing available Data (no Null-Values)
funnel_land_10 = px.funnel(sorted_value_counts_df_land10, x='Count', y='Boxen', title="Verfügbare Daten bei ländlichen SenseBoxen")
funnel_city_10 = px.funnel(sorted_value_counts_df_city10, x='Count', y='Boxen', title="Verfügbare Daten bei städischen SenseBoxen")
#---------------------------------------------------------
#dash-layout
app = Dash(__name__)

app.layout = html.Div([
    #-----------------------------------------------------
    #Titel und Live-Tracking-Part
    html.Div([
        html.H1("Luftqualität Deutschlands"),
        html.H2("Live-Tracking")
    ], style = {"text-align": "center"}),
    html.Div([
        html.Iframe(id="map", srcDoc= open("germany-boxes.html", "r").read(), width="100%", height="600")
    ]),
    html.Div([
        "Live-Tracking BoxID: ",
        dcc.Input(id="my-input", value="5c4eee5235809500190463cc")
    ], style={"text-align": "center"}),
    html.Div([
        dcc.Graph(id="live-graph"),
        dcc.Interval(
            id="interval-component",
            interval=1_000,
            n_intervals=0
        )
    ]),
    #------------------------------------------------------
    #Ausgewählte Boxen
    html.Div([
        html.H2("Analyse ausgewählter Boxen")
    ], style={"text-align": "center"}),
    html.Div([
        dcc.Dropdown(["Box1", "Box2", "Box3", "Box4", "Box5"], "Box1", id="landboxes"),
        dcc.Dropdown(["Box1", "Box2", "Box3", "Box4", "Box5"], "Box1", id="cityboxes"),
    ], style={"width": "40%"}),
    html.Div([
        dcc.Graph(id="available-data-land10", figure=funnel_land_10),
        dcc.Graph(id="available-data-city10", figure=funnel_city_10),
    ], style={"display": "flex"}),
    #------------------------------------------------------
])
df_interactive = pd.DataFrame(columns=["timestamp", "value"])
@callback(
    Output("live-graph", "figure"),
    Input("interval-component", "n_intervals"),
    Input("my-input", "value")
)
def update_graph(n,my_input_value):
    if my_input_value == "":
        return "    "
    url = f"https://api.opensensemap.org/boxes/{my_input_value}?format=json"
    data = requests.get(url).json()
    value = data["sensors"][1]["lastMeasurement"]["value"]
    timestamp = datetime.now()

    global df_interactive
    df_interactive = pd.concat([df_interactive, pd.DataFrame([{"timestamp": timestamp, "value": float(value)}])], ignore_index=True)
    trace = go.Scatter(x=df_interactive["timestamp"], y=df_interactive["value"], mode="lines+markers")
    layout = go.Layout(title="Real-time Sensor Data", xaxis=dict(title="Timestamp"), yaxis=dict(title="value"))
    fig = go.Figure(data=[trace], layout=layout)
    return fig

# @callback(
#     Output("available-data-land", "figure"),
#     Output("available-data-city", "figure"),
#     Input("landboxes", "value"),
#     Input("cityboxes", "value")
# )
# def update_graph_availability(landboxes_value, cityboxes_value):
#     if landboxes_value == ""
#--------------------------------------------------
if __name__=="__main__":
    app.run_server(debug=True)