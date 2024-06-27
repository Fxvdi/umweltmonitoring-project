#Libraries/Extensions
from dash import Dash, dcc, html, Input, Output, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error
import json
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
list_dataframes = [df_land_pm10, df_land_pm2_5, df_city_pm10, df_city_pm2_5]
columns_to_process = ['value_1', 'value_2', 'value_3', 'value_4', 'value_5']
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
funnel_land_10 = px.funnel(sorted_value_counts_df_land10, x='Count', y='Boxen', title="Verfügbare Daten bei ländlichen SenseBoxen", template="plotly_dark")
funnel_land_10.update_layout(title_x=0.5)
funnel_city_10 = px.funnel(sorted_value_counts_df_city10, x='Count', y='Boxen', title="Verfügbare Daten bei städischen SenseBoxen", template="plotly_dark")
funnel_city_10.update_layout(title_x=0.5)
#---------------------------------------------------------
##bereinigen,interpolieren,gruppieren von 
grouped_dataframes = []
for idx in list_dataframes:
    idx["datum"] = pd.to_datetime(idx["createdAt"])
    idx[columns_to_process] = idx[columns_to_process].apply(pd.to_numeric)
    for col in columns_to_process:
        idx[f'{col}_int'] = idx[col].interpolate()
    grouped_dataframes.append(idx.groupby("datum").mean(numeric_only=True))
##Liniendiagram
###land-PM10
result_land_pm10 = grouped_dataframes[0][[f'{col}_int' for col in columns_to_process]]
fig1 = px.line(result_land_pm10, title="Luftqualität auf dem Land nach PM10", template="plotly_dark")
fig1.update_layout(title_x=0.5)
###city-PM10
result_city_pm10 = grouped_dataframes[2][[f'{col}_int' for col in columns_to_process]]
fig2 = px.line(result_city_pm10, title="Luftqualität in der Stadt nach PM10", template="plotly_dark")
fig2.update_layout(title_x=0.5)
###land-PM2.5
result_land_pm2_5 = grouped_dataframes[1][[f'{col}_int' for col in columns_to_process]]
fig3 = px.line(result_land_pm2_5, title="Luftqualität auf dem Land nach PM2.5", template="plotly_dark")
fig3.update_layout(title_x=0.5)
###city-PM2.5
result_city_pm2_5 = grouped_dataframes[3][[f'{col}_int' for col in columns_to_process]]
#---------------------------------------------------------
fig4 = px.line(result_city_pm2_5, title="Luftqualität in der Stadt nach PM2.5", template="plotly_dark")
fig4.update_layout(title_x=0.5)
## Durchschnitt von Boxen
avg_landpm10 = grouped_dataframes[0].groupby('datum')[columns_to_process].mean()
avg_landpm25 = grouped_dataframes[1].groupby('datum')[columns_to_process].mean()
avg_citypm10 = grouped_dataframes[2].groupby('datum')[columns_to_process].mean()
avg_citypm25 = grouped_dataframes[3].groupby('datum')[columns_to_process].mean()

avg_landpm10 = avg_landpm10.reset_index()
avg_landpm25 = avg_landpm25.reset_index()
avg_citypm10 = avg_citypm10.reset_index()
avg_citypm25 = avg_citypm25.reset_index()

avg_landpm10['average_value'] = avg_landpm10[columns_to_process].mean(axis=1)
avg_landpm25['average_value'] = avg_landpm25[columns_to_process].mean(axis=1)
avg_citypm10['average_value'] = avg_citypm10[columns_to_process].mean(axis=1)
avg_citypm25['average_value'] = avg_citypm25[columns_to_process].mean(axis=1)

avg_landpm10 = avg_landpm10.dropna(subset="average_value")
avg_landpm25 = avg_landpm25.dropna(subset="average_value")
avg_citypm10 = avg_citypm10.dropna(subset="average_value")
avg_citypm25 = avg_citypm25.dropna(subset="average_value")

dfs = [avg_landpm10, avg_landpm25, avg_citypm10, avg_citypm25] # Zur weiterverarbeitung und zum Iterieren durch alle Dataframes
###visualisierung: Linien-Diagramm
#---------------------------------------------------------
## Test: Mergen von Dataframes
combined_df = dfs[0][['datum', 'average_value']].rename(columns={'average_value': 'average_value_df1'})

for i, df in enumerate(dfs[1:], start=2):
    combined_df = combined_df.merge(df[['datum', 'average_value']].rename(columns={'average_value': f'average_value_df{i}'}), on='datum')

column_rename_dict = {
    "average_value_df1": "LandPM10",
    "average_value_df2": "LandPM2_5",
    "average_value_df3": "CityPM10",
    "average_value_df4": "CityPM2_5"
}

fig_test = px.line(combined_df, x="datum", y=combined_df.columns[1:])
fig_test.for_each_trace(lambda t: t.update(name=column_rename_dict[t.name]))
#---------------------------------------------------------
fig5 = px.line(avg_landpm10, x="datum", y="average_value", title="Durchschnittliche Luftqualität auf dem Land | PM10")
fig6 = px.line(avg_landpm25, x="datum", y="average_value", title="Durchschnittliche Luftqualität auf dem Land | PM2.5")
fig7 = px.line(avg_citypm10, x="datum", y="average_value", title="Durchschnittliche Luftqualität in der Stadt | PM2.5")
fig8 = px.line(avg_citypm25, x="datum", y="average_value", title="Durchschnittliche Luftqualität in der Stadt | PM2.5")
fig5.update_layout(title_x=0.5)
fig6.update_layout(title_x=0.5)
fig7.update_layout(title_x=0.5)
fig8.update_layout(title_x=0.5)
#---------------------------------------------------------
df_stats = pd.read_csv("werte_daten/stats_df.csv")
#---------------------------------------------------------
#dash-layout
app = Dash(__name__)

app.layout = html.Div([
        html.Link(
        rel='stylesheet',
        href='assets/style.css'  # Pfad zur CSS-Datei im assets-Ordner
    ),
    #-----------------------------------------------------
    #Titel und Live-Tracking-Part
    html.Div([
        html.H1("Luftqualität Deutschlands"),
        html.H2("Live-Tracking")
    ], style = {"text-align": "center"}),
    html.Div([
        html.Iframe(id="map", srcDoc= open("germany-boxes.html", "r").read(), width="95%", height="600")
    ], style = {"text-align": "center"}),
    html.Div([
        "Live-Tracking BoxID: ",
        dcc.Input(id="my-input", value="5c4eee5235809500190463cc"),
        html.Br(),
        "(Achtung! Inaktive Boxen konnten nicht rausgefiltert werden - Bei Fehler, andere Box auswählen)"
    ], style={"text-align": "center", "font-size": "1.2em", "margin-top" : "0.5em"}),
    html.Div([
        dcc.Graph(id="live-graph"),
        dcc.Interval(
            id="interval-component",
            interval=1_000,
            n_intervals=0
        )
    ]),
    #------------------------------------------------------
    #Analyse ausgewählter Boxen
    html.Div([
        html.H2("Analyse ausgewählter Boxen")
    ], style={"text-align": "center"}),
    html.Div([
        dcc.Graph(id="available-data-land10", figure=funnel_land_10),
        dcc.Graph(id="available-data-city10", figure=funnel_city_10),
    ], style={        
        "display": "grid", 
        "grid-template-columns": "1fr 1fr", 
        "justify-content": "center",
        "align-items": "center",
        "gap": "20px"}),
    html.Div([
        html.H3("Analyse: PM10")
    ], style={"text-align": "center"}),
    html.Div([
        dcc.Graph(id="Stationen-Land-PM10", figure=fig1),
        dcc.Graph(id="Stationen-City-PM10", figure=fig2)
    ], style={        
        "display": "grid", 
        "grid-template-columns": "1fr 1fr", 
        "justify-content": "center",
        "align-items": "center",
        "gap": "20px"}),
    html.Div([
        html.H3("Analyse: PM10")
    ], style={"text-align": "center"}),
    html.Div([
        dash_table.DataTable(
            id="table",
            columns=[{"name": i, "id": i} for i in df_stats.columns],
            data=df_stats.to_dict("records"),
            style_header={
                'backgroundColor': 'rgb(30, 30, 30)',
                'color': 'white',
                'textAlign': 'center'
            },
            style_data={
                'backgroundColor': 'rgb(50, 50, 50)',
                'color': 'white',
                'textAlign': 'center'
            },
        )
    ]),
    html.Div([
        html.H3("Analyse: PM2.5")
    ], style={"text-align": "center"}),
    html.Div([
        dcc.Graph(id="Stationen-Land-PM2.5", figure=fig3),
        dcc.Graph(id="Stationen-City-PM2.5", figure=fig4)
    ], style={        
        "display": "grid", 
        "grid-template-columns": "1fr 1fr", 
        "justify-content": "center",
        "align-items": "center",
        "gap": "20px"}),
    html.Div([
        html.H3("Durchschnitt aller Boxen nach Attribut")
    ], style={"text-align": "center"}),
    html.Div([
        dcc.Graph(id="Average-of-Boxes-PM10", figure=fig_test),
    ]),
    html.Div([
        html.H2("Vorhersage"),
        dcc.Dropdown(
            ["land_PM10", "land_PM2_5", "stadt_PM10", "stadt_PM2_5"],
            value="land_PM10",
            id="select",
            style={'color': 'black', "background-color": "#111111"}
        )
    ], style={"text-align": "center"}),
    html.Div([
        dcc.Graph(id="prediction-graph")
    ]),
    html.Div([
        html.H3("Mean Squared Error (MSE)"),
    ], style={"text-align": "center"}),
    html.Div([
        html.Table([
            html.Tr([html.Td("Arithmetisches Mittel: "), html.Td(id='arith')]),
            html.Tr([html.Td("Naive Methode: "), html.Td(id='fool')]),
            html.Tr([html.Td("Saisonal Naiv"), html.Td(id='seasonal-fool')]),
            html.Tr([html.Td("exponentielle Glättung"), html.Td(id='exp-smooth')]),
        ]),
    ], className="centered-table", style={"display": "flex", "justify-content": "center", "align-items": "center"}),
    html.Div([
        html.H3("LSTM-Modell"),
    ], style={"text-align": "center"}),
    html.Div([
        dcc.Graph(
            id="lstm-model",
        )
    ])
    #------------------------------------------------------
    # Callback/interaktives Livetracking
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
    value1 = data["sensors"][1]["lastMeasurement"]["value"]
    value0 = data["sensors"][0]["lastMeasurement"]["value"]
    timestamp = datetime.now()

    global df_interactive
    df_interactive = pd.concat([df_interactive, pd.DataFrame([{"timestamp": timestamp, "value1": float(value1), "value0": float(value0)}])], ignore_index=True)
    trace_pm2_5 = go.Scatter(x=df_interactive["timestamp"], y=df_interactive["value1"], mode="lines+markers", name="PM2,5")
    trace_pm10 = go.Scatter(x=df_interactive["timestamp"], y=df_interactive["value0"], mode="lines+markers", name="PM10")
    
    layout = go.Layout(
        title=f"Real-time Sensor Data {my_input_value}",
        xaxis=dict(title="Timestamp"), 
        yaxis=dict(title="value"),
        showlegend=True
        )
    fig = go.Figure(data=[trace_pm2_5, trace_pm10], layout=layout)
    fig.update_layout(title_x=0.5, template="plotly_dark")
    #fig = px.scatter(x=df_interactive["timestamp"], y=df_interactive["value1", "value0"], mode="lines+markers", title="Real-time Sensor Data")
    return fig
#---------------------------------------------------
# Callback: Selektion von Vorhergesagtem Attribut
@callback(
    Output("prediction-graph", "figure"),
    Output("arith", "children"),
    Output("fool", "children"),
    Output("seasonal-fool", "children"),
    Output("exp-smooth", "children"),
    Output("lstm-model", "figure"),
    Input("select", "value")
)
def update_prediction_graph(my_select_value):
    df_predict = pd.read_csv(f"werte_daten/werte_{my_select_value}.csv")
    # Datumsspalte in DateTime-Objekte konvertieren
    df_predict['createdAt'] = pd.to_datetime(df_predict['createdAt'])

    # Mittelwert der Spalten 'value_1', 'value_2' und 'value_3' berechnen
    df_predict[f'{my_select_value}'] = df_predict[['value_1', 'value_2', 'value_3', 'value_4', 'value_5']].mean(axis=1)

    # DataFrame auf ausgewählte Spalten reduzieren und createdAt als Index setzen
    selected_df1 = df_predict[['createdAt', f'{my_select_value}']].set_index('createdAt')
    # Duplikate im Index entfernen
    selected_df1 = selected_df1[~selected_df1.index.duplicated(keep='first')]
    # Remove rows with NaN values
    selected_df1 = selected_df1.dropna()
#----------------------------------------------------
#interne Methoden für die Prediction
    # 1. Arithmetisches Mittel
    def arithmetisches_mittel(train, test):
        mean_value = train.mean()
        predictions['Arithmetisches Mittel'] = mean_value
        forecasts['Arithmetisches Mittel'] = mean_value
        return np.full_like(test, fill_value=mean_value)

    # 2. Naive Methode
    def naive_methode(train, test):
        last_value = train.iloc[-1]
        predictions['Naive Methode'] = last_value
        last_value = test.iloc[-1]
        forecasts['Naive Mathode'] = last_value
        return np.full_like(test, fill_value=last_value)

    # 3. Saisonale naive Vorhersage (Vorhersage basierend auf dem letzten bekannten Wert derselben Saison)
    def saisonale_naive(train, test):
        # Die letzten bekannten Werte jeder Saison (Monat) aus dem Trainingsdatensatz erhalten
        seasonal_last_values = train.groupby(level='Monat').apply(lambda x: x.tail(1)).values
        predictions['Saisonale naive'] = np.tile(seasonal_last_values, n_test // 12 + 1)[:n_test]
        n_forecast = len(forecasts.index)
        seasonal_last_values_test = test.groupby(level='Monat').apply(lambda x: x.tail(1)).values
        # Für die Vorhersagen der Zukunft, beginnt es im Juni
        start_month = 5
        # Berechne den Index des Startmonats in der Liste
        start_index = (start_month - 1) % 12
        # Verschiebe die Liste so, dass sie mit dem Startmonat beginnt
        shifted_values = np.concatenate((seasonal_last_values_test[start_index:], seasonal_last_values_test[:start_index]))
        forecasts['Saisonale naive'] = np.tile(shifted_values, n_forecast // 12 + 1)[:n_forecast]
        return predictions['Saisonale naive'].values

    def exponentielle_glättung(train, test):
        prediction = pd.Series(index=test.index)
        prediction.iloc[0] = train.iloc[-1]  # Initialisierung des ersten Werts
        # den Alpha Wert festlegen
        alpha = 0.5
        for i in range(1, len(test)):
            prediction.iloc[i] = alpha*test.iloc[i - 1] + alpha*(1 - alpha)**2 * test.iloc[i - 2] + alpha*(1 - alpha)**3 * test.iloc[i - 3]
        predictions["Exponentielle Glättung"] = prediction
        forecast = pd.Series(index=forecasts.index)
        # hier gibt es andere Alphawerte, da es in meinen Augen nur Sinn macht, wenn es insgesamt 1 ist
        # Da die Werte sonst immer kleiner werden
        alpha1 = 0.58
        alpha2 = 0.29
        alpha3 = 0.13
        # es wird unterschieden, ob man noch den Index von den Testdaten benutzen muss oder nur die Forecast-Daten
        for i in range(0, len(forecast.index)):
            if i == 0:
                forecast.iloc[i] = alpha1*test.iloc[-1] + alpha2* test.iloc[-2] + alpha3* test.iloc[-3]
            elif i == 1:
                forecast.iloc[i] = alpha1*forecast.iloc[i - 1] + alpha2* test.iloc[-1] + alpha3* test.iloc[-2]
            elif i == 2:
                forecast.iloc[i] = alpha1*forecast.iloc[i - 1] + alpha2* forecast.iloc[i - 2] + alpha3* test.iloc[-1]
            else:
                forecast.iloc[i] = alpha1*forecast.iloc[i - 1] + alpha2* forecast.iloc[i - 2] + alpha3* forecast.iloc[i - 3]
        forecasts["Exponentielle Glättung"] = forecast
        return predictions['Exponentielle Glättung'].values
#----------------------------------------------------
    # Erstellen von Spalten für Jahr und Monat
    selected_df1['Jahr'] = selected_df1.index.year
    selected_df1['Monat'] = selected_df1.index.month

    # Gruppierung nach Jahr und Monat, Berechnung des Durchschnitts
    monthly_mean = selected_df1.groupby(['Jahr', 'Monat']).mean()

    # Aufteilung in Trainings- und Testdaten
    train = monthly_mean[monthly_mean.index < (2023, 6)][f'{my_select_value}']
    test = monthly_mean[monthly_mean.index >= (2023, 6)][f'{my_select_value}']

    # Anzahl der Testdatenpunkte
    n_test = len(test)

    # Platzhalter für Vorhersagen
    predictions = pd.DataFrame(index=test.index)

    # Erstellen des Index mit dem gewünschten Datumbereich für die Vorhersage
    forecast_index = pd.period_range(start='2024-06', end='2025-06', freq='M')

    # Erstellen des leeren DataFrames mit dem Index
    forecasts = pd.DataFrame(index=forecast_index)

    # die Funktionen werden nun ausgeführt und die Dataframes mit den berechneten Werten gefüllt
    arith_predictions = arithmetisches_mittel(train, test)
    naive_predictions = naive_methode(train, test)
    seasonal_naive_predictions = saisonale_naive(train, test)
    exp_smooth_predictions = exponentielle_glättung(train, test)

    # Reset index für Predictions DataFrame
    predictions_reset = predictions.reset_index()

    # Schmelzen des DataFrames für Plotly Express
    predictions_melted = predictions_reset.melt(id_vars=['Jahr', 'Monat'], 
                                                value_vars=['Arithmetisches Mittel', 'Naive Methode', 'Saisonale naive', 'Exponentielle Glättung'],
                                                var_name='Methode', 
                                                value_name=f'{my_select_value}')


    # das Datum wird in das richtige Format gebracht, für das bessere darstellen beider Dataframes
    predictions_melted['Datum'] = pd.to_datetime(predictions_melted['Jahr'].astype(str) + '-' + predictions_melted['Monat'].astype(str))

    # Liniengrafik erstellen
    fig_predict = px.line(predictions_melted, x='Datum', y=f'{my_select_value}', color='Methode',
                labels={f'{my_select_value}': f'{my_select_value}', 'Methode': 'Methode', 'Datum': 'Datum'},
                title=f'Vergleich der Vorhersagemethoden für Werte {my_select_value}')

    # Das Datum wird nun auch in das richtige Format gebracht
    test.index = test.index.map(lambda x: f'{x[0]}-{x[1]:02d}')

    # Hinzufügen der Testdaten
    fig_predict.add_scatter(x=test.index, y=test.values, mode='lines', name='Testdaten (2023)')

    # nun wird auch der Index des Forecasts zu dem gleichen Format
    forecasts.index = forecasts.index.to_timestamp()
    # Linien für den Forecast erstellen
    fig_predict.add_trace(go.Scatter(x=forecasts.index, y=forecasts['Arithmetisches Mittel'], mode='lines', name='Arithmetisches Mittel Forecast'))
    fig_predict.add_trace(go.Scatter(x=forecasts.index, y=forecasts['Naive Mathode'], mode='lines', name='Naive Methode Forecast'))
    fig_predict.add_trace(go.Scatter(x=forecasts.index, y=forecasts['Saisonale naive'], mode='lines', name='Saisonale naive Forecast'))
    fig_predict.add_trace(go.Scatter(x=forecasts.index, y=forecasts['Exponentielle Glättung'], mode='lines', name='Exponentielle Glättung Forecast'))

    fig_predict.update_layout(title_x=0.5, template="plotly_dark")

    mae_arith = mean_absolute_error(test, arith_predictions)
    mae_naive = mean_absolute_error(test, naive_predictions)
    mae_seasonal_naive = mean_absolute_error(test, seasonal_naive_predictions)
    mae_exp_smooth = mean_absolute_error(test, exp_smooth_predictions)

    if my_select_value == "land_PM10":
        with open("prediction_land_PM10.json", "r") as f:
            fig_json = f.read()
        fig_lstm = go.Figure(json.loads(fig_json))
    elif my_select_value == "land_PM2_5":
        with open("prediction_land_PM2_5.json", "r") as f:
            fig_json = f.read()
        fig_lstm = go.Figure(json.loads(fig_json))
    elif my_select_value == "stadt_PM10":
        with open("prediction_stadt_PM10.json", "r") as f:
            fig_json = f.read()
        fig_lstm = go.Figure(json.loads(fig_json))
    elif my_select_value == "stadt_PM2_5":
        with open("prediction_stadt_PM2_5.json", "r") as f:
            fig_json = f.read()
        fig_lstm = go.Figure(json.loads(fig_json))

    fig_lstm.update_layout(title_x=0.5, template="plotly_dark")
    # Grafik und Werte anzeigen
    return fig_predict, mae_arith, mae_naive, mae_seasonal_naive, mae_exp_smooth, fig_lstm
#--------------------------------------------------
if __name__=="__main__":
    app.run_server(debug=True)