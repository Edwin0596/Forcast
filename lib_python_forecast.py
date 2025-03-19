
# PAQUETERIA
import pandas as pd
import numpy as np
import urllib
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, PoissonRegressor # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import zscore
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import timedelta
from prophet import Prophet
import warnings
import ruptures as rpt
import os
import zipfile

warnings.filterwarnings("ignore")
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
#####################################################################  
# DICCIONARIO DE VARIABLES
dict_variables = {'tiempo': 'fecha', 'producto': 'id_item', 'cantidad':'unidades'}

###############################################################################################
##### MODELO ARIMA
# Funcion para modelo ARIMA
def forecast_arima(data, forecast_period):
    item_data = data.set_index('fecha')['unidades']

    if not isinstance(item_data.index, pd.DatetimeIndex):
        item_data.index = pd.to_datetime(item_data.index)
    
    item_data = item_data.asfreq('W-MON')

    if len(item_data) < 15:
        return None, None

    try:
        p_value = [0, 1, 2, 3, 4, 5, 6]
        d_value = [0, 1]
        q_value = [0, 1, 2, 3]

        best_rmse = np.inf
        best_order = None
        best_model = None

        for p in p_value:
            for d in d_value:
                for q in q_value:
                    order = (p, d, q)
                    try:
                        arima_model = ARIMA(item_data, order=order)
                        arima_fit = arima_model.fit()

                        rmse = np.sqrt(np.mean((arima_fit.resid ** 2)))

                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_order = order
                            best_model = arima_fit
                    except (np.linalg.LinAlgError, ValueError):
                        continue
                    #
                #
            # 
        #
        if best_model is not None:
            fitted_values = best_model.fittedvalues
            forecast_values = arima_fit.forecast(steps=forecast_period)
            print('Modelo ARIMA ajustado...\n')
            return fitted_values, forecast_values
        else:
            return None, None
    #   
    except Exception as e:
        return None, None
#

#####################################################################
##### MODELO DE REGRESION
# Funcion para modelo de regresion con efectos temporales
def forecast_regresion(data, forecast_period, max_date):
    item_data = data.copy()
    # fechas ordinales
    item_data['fecha_ordinal'] = item_data['fecha'].map(pd.Timestamp.toordinal)
    item_data['semana'] = item_data['fecha'].dt.isocalendar().week
    # variables de tiempo
    X = item_data[['fecha_ordinal', 'semana']].values
    y = item_data['unidades'].values
    # ajuste modelo
    lineal_model = LinearRegression()
    lineal_model.fit(X, y)
    # valores ajustados
    fitted_values = lineal_model.predict(X)
    # tiempo futuro
    future_dates = pd.date_range(start=max_date + timedelta(days=1), periods=forecast_period)
    future_dates_ordinal = future_dates.map(pd.Timestamp.toordinal).values
    future_wk = future_dates.isocalendar().week.values
    # X_prediccion
    X_prediccion = np.column_stack((future_dates_ordinal, future_wk))
    # prediccion
    forecast_values = lineal_model.predict(X_prediccion)
    #
    print('Modelo Regresion ajustados...\n')
    return fitted_values, forecast_values
#

#####################################################################
##### MODELO DE REGRESION POISSON
# Funcion para modelo de regresion de Poisson
def poisson_forecast(data):
    # parametros
    alphas = [0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    item_p = data.copy()
    item_p['unidades'] = item_p['unidades'].apply(lambda x: max(0, int(x)))
    # tiempo ordinal
    item_p['fecha_ordinal'] = item_p['fecha'].map(pd.Timestamp.toordinal)
    item_p['semana'] = item_p['fecha'].dt.isocalendar().week
    # separacion de datos entrenamiento y prueba
    train_data, test_data = train_test_split(item_p, test_size=0.3, random_state=42, shuffle=False)
    # parametros de seleccion
    best_alpha = None
    best_model = None
    best_rmse = np.inf

    for alpha in alphas:
        # modelo para alpha
        poisson_modelo = PoissonRegressor(alpha=alpha)
        # entrenar el modelo
        poisson_modelo.fit(train_data[['fecha_ordinal', 'semana']], train_data['unidades'])
        test_data['Cantidad_Pronosticada_Poisson'] = poisson_modelo.predict(test_data[['fecha_ordinal', 'semana']])
        rmse = np.sqrt(mean_squared_error(test_data['unidades'], test_data['Cantidad_Pronosticada_Poisson']))
        # seleccion de mejor modelo
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
            best_model = poisson_modelo
        #
    #
    print('Modelo Poisson ajustados...\n')
    return best_model

###############################################################################################
### MODELO DE RED NEURONAL RECURRENTE
# Red Neuronal Recurrente (RNN)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    #
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
#
# Dataset de series temporales
class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length
    #
    def __len__(self):
        return len(self.data) - self.sequence_length
    #
    def __getitem__(self, index):
        return (self.data[index:index+self.sequence_length], self.data[index+self.sequence_length])
#
# Pesos
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
#
def train_lstm_model(data, input_size, hidden_size, output_size, sequence_lenght):
    data = torch.FloatTensor(data).view(-1, 1)
    dataset = TimeSeriesDataset(data, sequence_lenght)
    dataloader = DataLoader(dataset, batch_size= 32, shuffle= True)
    #
    model = LSTMModel(input_size, hidden_size, output_size)
    device = torch.device("cpu")
    model.to(device)
    model.apply(init_weights)
    #
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    #
    for epoch in range(100):
        for seq, labels in dataloader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            y_pred = model(seq)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
        #
    #
    return model
#

###############################################################################################
### METODO DE AJUSTE PROPHET
def prophet_forecast(data, forecast_period):
    data_dia = data.rename(columns = {'fecha':'ds', 'unidades':'y'}).copy()
    # crear modelo
    modelo = Prophet()
    # entrenar modelo
    modelo.fit(data_dia)
    # dataframe de fechas de pronostico
    pronostico = modelo.make_future_dataframe(periods=forecast_period, freq='W-MON')
    # generar pronostico
    forecast = modelo.predict(pronostico)
    # recuperar nombres
    forecast.rename(columns={'ds':'fecha', 'yhat':'unidades'}, inplace=True)
    forecast = forecast[['unidades']]
    # ajustados
    fitted_values = forecast[:(forecast.shape[0]-forecast_period)]
    # pronostico
    fitted_forecast = forecast.tail(forecast_period)
    print('Modelo Prophet ajustados...\n')
    return fitted_values, fitted_forecast


###############################################################################################
### DETECCION ATIPICOS
def detectar_atipicos(data, unidades, umbral_z = 3):
    # validacion de existencia de la columna
    if unidades not in data.columns:
        raise ValueError(f"La columna '{unidades}' no existe en el set de datos")
    #
    try:
        model = ARIMA(data[unidades], order = (1, 1, 1))
        result = model.fit()
        # analisis de residuos
        residuos = result.resid
        # detectar atipicos usando z-score
        z_scores = zscore(residuos)
        outliers = np.abs(z_scores) > umbral_z
        out_int = [int(i) for i in outliers]
        # marcar atipicos
        data['atipico'] = out_int
        #
        return data
    except Exception as e:
        raise RuntimeError(f"Error al ajustar modelo\n")

#

###############################################################################################
### DETECTAR INTERVENCION
def detectar_intervencion(data, unidades, penalizacion = 10):
    # validacion de existencia de columna
    if unidades not in data.columns:
        raise ValueError(f"La columna '{unidades}' no existe en el set de datos")
    # deteccion de senal
    signal = np.array(data[unidades])
    # detectar puntos de cambio usando algoritmo PELT
    try:
        pelt = rpt.Pelt(model="rbf").fit(signal)
        punto_cambio = pelt.predict(pen = penalizacion)
    except Exception as e:
        raise RuntimeError(f"Error al detectar puntos de cambio\n")
    # marcar intervencion
    data['intervencion'] = 0
    for cp in punto_cambio:
        if cp < len(data):
            data.loc[cp, 'intervencion'] = 1
        #
    #
    return data
#

#####################################################################
def arc_to_zip(origen, destino):
    zip_file = destino

    with zipfile.ZipFile(zip_file, 'w') as zipf:
        # Recorrer los archivos en el directorio
        for file in os.listdir(origen):
            if file.endswith('.csv'):
                #incorporar archivo al zip
                file_path = os.path.join(origen, file)
                zipf.write(file_path, arcname = file)
                # eliminar el archivo
                os.remove(file_path)
            #
        #
    #
    print('\nArchivo comprimido con exito almacenado en la ruta:', destino)
#

#####################################################################
##### CONEXION A BASE DE DATOS

server = ''
database = ''
username = ''
password = ''
driver = '{ODBC Driver 17 for SQL Server}'

# Cadena de conexion
connection_string = (
    f'DRIVER={driver};'
    f'SERVER={server};'
    f'DATABASE={database};'
    f'UID={username};'
    f'PWD={password}'
)

# Codificar la cadena de conexión
params = urllib.parse.quote_plus(connection_string)

# Crear el motor de SQLAlchemy
engine = create_engine(f'mssql+pyodbc:///?odbc_connect={params}')