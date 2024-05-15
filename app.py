#Libraries/Extensions
from dash import Dash, dcc, html, Input, Output, callback
import dash_leaflet as dl
import plotly.express as px
import pandas as pd
import requests
#---------------------------------------------------------
#everything about styling
attr_style = '&copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
url_style = 'https://tiles.stadiamaps.com/tiles/osm_bright/{z}/{x}/{y}{r}.png'
#---------------------------------------------------------
#constants
latitude = 52.516181 #Breitengrad
longitude = 13.376935 #Längegrad
#---------------------------------------------------------
#dataframe
coordinates = {"id": [],
                  "latitude": [],
                  "longitude": []
                }
df_coordinates = pd.DataFrame(coordinates)

url = "https://api.opensensemap.org/boxes?near=10.323007,51.16501&maxDistance=100000&exposure=outdoor&title=PM10,PM2.5&format=geojson"
data = requests.get(url).json()
for idx in range(0, len(data["features"])):
    df2 = {"id": data["features"][idx]["properties"]["_id"], "latitude": data["features"][idx]["geometry"]["coordinates"][1], "longitude": data["features"][idx]["geometry"]["coordinates"][0]}
    df_coordinates = df_coordinates._append(df2, ignore_index = True)
#---------------------------------------------------------
#dash-layout
app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1("Luftqualität Deutschlands")
    ], style = {"text-align": "center"}),
    html.Div([
        dl.Map([
            dl.TileLayer(url=url_style, attribution=attr_style),
            *[
                dl.Marker(
                    position=[row["latitude"], row["longitude"]], children=[dl.Popup(content=f"BoxID: {row['id']}")]
                )for index, row in df_coordinates.iterrows()
            ]
        ], center=[51.16501, 10.323007], zoom=7, style={"height": "65vh"}, id="boxes-in-germany")
    ])
])

# @callback(
# Output("interactive-map", "children"),
# Input()
# )
# def update_map():
#     updated_map = [
#         dl.TileLayer(url=url_style, attribution=attr_style),
#         *[
#             dl.Marker(
#                 position=[row["latitude"], row["longitude"]],
#                 children=[
#                     dl.Popup([
#                         html.P(f"BoxID: {row['id']}"),
#                     ])
#                 ]
#             )for index, row in df_coordinates.iterrows()
#         ]
#     ]
#     return updated_map
if __name__=="__main__":
    app.run_server(debug=True)