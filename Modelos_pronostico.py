#%% paqueteria
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime as dt
from datetime import timedelta
import torch
import os

import lib_python_forecast as utils
#
_inicio = dt.now()
print('\nESTIMACION INICIADA... | ' + str(dt.now()) + '\n')

# variables de configuracion
forecast_period = 13 # semanas equivalentes a 91 dias
cond_min_dias = 15
_ejecucion = dt.now().strftime("%Y%m%d")

# ruta de trabajo
ruta = os.path.dirname(os.path.abspath(__file__))
# ruta de datos
dir_datos = os.path.join(os.path.dirname(ruta),'Datos')
if not os.path.exists(dir_datos):
    os.makedirs(dir_datos)
# ruta de resultados
dir_resultados = os.path.join(os.path.dirname(ruta), 'Resultados')
# verificacion de directorio resultados
if not os.path.exists(dir_resultados):
    os.makedirs(dir_resultados)

print('\nRUTA DE TRABAJO: ' + ruta)
print('\nDIRECTORIO DE DATOS: ' + dir_datos)
print('\nDIRECTORIO DE RESULTADOS: ' + dir_resultados)
#
###############################################################################################
# variables de resultados y de procesos
forecast_result = pd.DataFrame()
torneo_result = pd.DataFrame()
# resultados de ajuste
arima_result = pd.DataFrame()
lineal_result = pd.DataFrame()
poisson_result = pd.DataFrame()
rnn_result = pd.DataFrame()
prophet_result = pd.DataFrame()
# resultados pronosticos
arima_forecast = pd.DataFrame()
lineal_forecast = pd.DataFrame()
poisson_forecast = pd.DataFrame()
rnn_forecast = pd.DataFrame()
prophet_forecast = pd.DataFrame()
# global para modelos
arima = pd.DataFrame()
lineal = pd.DataFrame()
poisson = pd.DataFrame()
rnn = pd.DataFrame()
prophet = pd.DataFrame()
# resultados globales
resultados_globales = pd.DataFrame()

print('\nCONFIGURACION INICIALES TERMINADA | ' + str(dt.now()) )
###############################################################################################
#% lectura de datos
print('\nLECTURA DE DATOS INICIO | ' + str(dt.now()) )
fin_mes = dt.today().date().replace(day=1)-timedelta(1)

query ='''
WITH ventas_semanales AS (
    SELECT 
        v.fecha,
        v.codProducto,
        'C' + CONVERT(VARCHAR, p.CodigoCategoria) AS categoria_id,
        SUM(v.unidades) AS Unidades,
        SUM(v.venta_usd) Venta,
        DATEADD(DAY, -(DATEPART(WEEKDAY, v.fecha) - 2), v.fecha) AS LunesSemana -- Calcula el primer lunes de la semana
    FROM 
        CMS_BI.cube.Ventas v
    JOIN 
        [CMS_BI].cube.Productos p
        ON v.CodProducto = p.CodProducto
    WHERE 
        p.Descripcion NOT IN ('INACTIVO', 'ERROR', 'SIN DATOS')
        AND p.Descripcion IS NOT NULL
        AND p.CodigoCategoria NOT IN (0, '')
        AND p.CodigoCategoria IS NOT NULL
        AND v.Venta_USD > 0	
        AND v.unidades > 0	
        AND v.Fecha BETWEEN '2021-01-01' and CONVERT(date, \''''+str(fin_mes)+'''\')
    GROUP BY 
        v.fecha,
        v.codProducto,
        'C' + CONVERT(VARCHAR, p.CodigoCategoria)
)
SELECT 
    MIN(LunesSemana) AS fecha,
    codProducto producto_id,
    categoria_id,
    SUM(Unidades) AS unidades,
    SUM(Venta) AS venta
FROM 
    ventas_semanales
GROUP BY 
    LunesSemana,
    codProducto,
    categoria_id
'''

#datos = pd.read_sql(query, utils.engine)
datos = pd.read_parquet('Datos/20250303_datos_wk.parquet')
#
datos = datos.rename(columns = {'codProducto':'producto_id', 'Semana':'fecha'} )
print('\nLECTURA DE DATOS FIN | ' + str(dt.now()))
productos = datos['producto_id'].unique()
#
print('\nPRODUCTOS OBTENIDOS | ' +str(productos.shape[0]) + ' | '+ str(dt.now()) )
print('\nREGISTROS OBTENIDOS >> '+str(datos.shape[0]) + ' | '+ str(dt.now()) )

###############################################################################################
# preparacion de datos
# formatos de datos
datos['fecha'] = pd.to_datetime(datos['fecha'])
datos['producto_id'] = datos['producto_id'].astype(str)
#
print('\nCONDICION PARA MODELADO DE DATOS | ' + str(dt.now()))
periodo_datos = datos['fecha'].max() - timedelta(180)
modelos = datos[datos['fecha'] > periodo_datos].groupby(['producto_id'], as_index = False).agg(dias_venta =('fecha', 'nunique'))
modelos['ajuste'] = np.where(modelos['dias_venta'] > cond_min_dias, 1, 0)
print(f"\nPRODUCTOS MODELABLES {modelos['ajuste'].sum()} de {modelos.shape[0]} EQUIVALENTE A {round(100*modelos['ajuste'].sum()/modelos.shape[0],1)}% DE LOS PRODUCTOS | {str(dt.now())}")

modelable = modelos[modelos['ajuste'] == 1]['producto_id'].values

###############################################################################################
# ajuste de modelos
print('\nINICIO DE MODELOS  | ' + str(dt.now()))
a = 0
#%

for prod in modelable:
    
    #%
    a += 1
    t_resultado = pd.DataFrame()
    print(f'\nINICIO PRODUCTO {prod} | {a} DE {modelable.shape[0]} | {str(dt.now())}')
    temp = datos[datos['producto_id'] == prod].copy()
    temp = temp.sort_values('fecha')
    #
    inicio = temp['fecha'].min()
    fechas = {'fecha': pd.date_range(start=inicio, end=fin_mes, freq='W-MON')}
    fechas = pd.DataFrame(fechas)
    #
    fechas['fecha_ordinal'] = fechas['fecha'].map(pd.Timestamp.toordinal)
    fechas['semana'] = fechas['fecha'].map(lambda x: x.isocalendar()[1])
    fechas = pd.merge(fechas, temp, how = 'left', right_on = 'fecha', left_on = 'fecha')
    fechas['producto_id'] = prod
    fechas['unidades'] = fechas['unidades'].fillna(0)
    #
    temp = fechas.copy()
    # deteccion de atipicos
    temp = utils.detectar_atipicos(temp, 'unidades', umbral_z = 3)
    # deteccion de eventos de intervencion
    temp = utils.detectar_intervencion(temp, 'unidades', penalizacion = 10)
    # tratamiento de atipicos
    lim_sup = temp[(temp['atipico'] == 0) & (temp['intervencion'] == 0)]['unidades'].quantile(0.975)
    temp['unidades'] = np.where(temp['unidades'] > lim_sup, lim_sup, temp['unidades'])
    #
    max_fecha = datos['fecha'].max()
    f_pronosticos = {'fecha': pd.date_range(start= max_fecha + timedelta(7), periods = forecast_period, freq='W-MON')}
    f_pronosticos = pd.DataFrame(f_pronosticos)
    f_pronosticos['fecha_ordinal'] = f_pronosticos['fecha'].map(pd.Timestamp.toordinal)
    f_pronosticos['semana'] = f_pronosticos['fecha'].map(lambda x: x.isocalendar()[1])    
    #%
    ################################################################################################
    # ARIMA
    print('MODELO ARIMA | ' + str(dt.now()) + ' \n')
    fitted, forecast = utils.forecast_arima(temp, forecast_period)
    if fitted is not None:
        # ajuste
        t_fit = pd.DataFrame()
        t_fit = pd.DataFrame({
            'fecha': temp['fecha'],
            'unidades':fitted.values
        })
        t_fit['producto_id'] = prod
        t_fit['unidades'] = np.where(t_fit['unidades'] < 0, 0, t_fit['unidades'])
        arima_result = pd.concat([arima_result, t_fit], ignore_index=True)
        # pronostico
        t_forecast = pd.DataFrame()
        t_forecast = pd.DataFrame({
            'fecha': f_pronosticos['fecha'],
            'unidades':forecast.values
        })
        t_forecast['producto_id'] = prod
        t_forecast['unidades'] = np.where(t_forecast['unidades'] < 1, 1, t_forecast['unidades'])
        arima_forecast = pd.concat([arima_forecast, t_forecast], ignore_index=True)
        #
        arima_result['fecha'] = pd.to_datetime(arima_result['fecha'])
        arima_forecast['fecha'] = pd.to_datetime(arima_forecast['fecha'])
        t_arima = pd.DataFrame()
        t_arima = pd.concat([arima_result, arima_forecast], ignore_index = True)
        arima = t_arima.copy()
        #%
    #

    ################################################################################################
    # LINEAL
    print('MODELO LINEAL | ' + str(dt.now()) + ' \n')
    fitted, forecast = utils.forecast_regresion(temp, forecast_period, fin_mes)
    if fitted is not None:
        # ajuste
        t_fit = pd.DataFrame()
        t_fit = pd.DataFrame({
            'fecha': temp['fecha'],
            'unidades':fitted
        })
        t_fit['producto_id'] = prod
        t_fit['unidades'] = np.where(t_fit['unidades'] < 1, 1, t_fit['unidades'])
        lineal_result = pd.concat([lineal_result, t_fit], ignore_index=True)
        # pronostico
        t_forecast = pd.DataFrame()
        t_forecast = pd.DataFrame({
            'fecha': f_pronosticos['fecha'],
            'unidades':forecast
        })
        t_forecast['producto_id'] = prod
        t_forecast['unidades'] = np.where(t_forecast['unidades'] < 1, 1, t_forecast['unidades'])
        lineal_forecast = pd.concat([lineal_forecast, t_forecast], ignore_index=True)
        #
        lineal_result['fecha'] = pd.to_datetime(lineal_result['fecha'])
        lineal_forecast['fecha'] = pd.to_datetime(lineal_forecast['fecha'])
        t_lineal = pd.DataFrame()
        t_lineal = pd.concat([lineal_result, lineal_forecast], ignore_index=True)
        lineal = t_lineal.copy()
    #
    
    ################################################################################################
    # POISSON
    print('MODELO POISSON | ' + str(dt.now()) + ' \n')
    # poisson_model = utils.poisson_forecast(temp)
    poisson_model = utils.poisson_forecast(temp)
    if poisson_model is not None:
        # ajuste
        fitted = poisson_model.predict(temp[['fecha_ordinal','semana']])
        t_fit = pd.DataFrame()
        t_fit = pd.DataFrame({
            'fecha': temp['fecha'],
            'unidades':fitted
        })
        t_fit['producto_id'] = prod
        t_fit['unidades'] = np.where(t_fit['unidades'] < 1, 1, t_fit['unidades'])
        poisson_result = pd.concat([poisson_result, t_fit], ignore_index=True)
        # pronostico
        forecast = poisson_model.predict(f_pronosticos[['fecha_ordinal','semana']])
        t_forecast = pd.DataFrame()
        t_forecast = pd.DataFrame({
            'fecha': f_pronosticos['fecha'],
            'unidades':forecast
        })
        t_forecast['producto_id'] = prod
        t_forecast['unidades'] = np.where(t_forecast['unidades'] < 1, 1, t_forecast['unidades'])
        poisson_forecast = pd.concat([poisson_forecast, t_forecast], ignore_index=True)
        #
        poisson_result['fecha'] = pd.to_datetime(poisson_result['fecha'])
        poisson_forecast['fecha'] = pd.to_datetime(poisson_forecast['fecha'])
        t_poisson = pd.DataFrame()
        t_poisson = pd.concat([poisson_result, poisson_forecast], ignore_index=True)
        poisson = t_poisson.copy()
    #

    ################################################################################################
    # RNN
    print('MODELO DE RED NEURONAL | ' + str(dt.now()) + ' \n')
    secuencia = 14
    item_data = temp['unidades'].values
    if item_data.shape[0] > 125:
        rnn_model = utils.train_lstm_model(item_data,input_size= 1, hidden_size= 10, output_size= 1, sequence_lenght= secuencia)

        if rnn_model is not None:
            fitted = []
            for i in range(secuencia, len(item_data)):
                input_seq = item_data[i-secuencia:i].reshape(1, -1, 1)
                prediccion = rnn_model(torch.FloatTensor(input_seq))
                fitted.append(prediccion.detach().numpy().flatten()[0])
            #
            # convertir predicciones a array
            fitted = np.array(fitted)
            # ajuste
            t_fit = pd.DataFrame()
            t_fit = pd.DataFrame({
                'fecha': temp['fecha'][secuencia:],
                'unidades': fitted
            })
            t_fit['producto_id'] = prod
            t_fit['unidades'].fillna(1, inplace=True)
            t_fit['unidades'] = np.where(t_fit['unidades'] < 1, 1, t_fit['unidades'])
            rnn_result = pd.concat([rnn_result, t_fit], ignore_index=True)
            # pronostico
            forecast = []
            for i in range(forecast_period):
                input_seq = item_data[-secuencia+i:].reshape(1, -1, 1)
                prediccion = rnn_model(torch.FloatTensor(input_seq))
                forecast.append(prediccion.detach().numpy().flatten()[0])
            #
            # convertir predicciones a array
            forecast = np.array(forecast)
            t_forecast = pd.DataFrame()
            t_forecast = pd.DataFrame({
                'fecha': f_pronosticos['fecha'],
                'unidades': forecast
            })
            t_forecast['producto_id'] = prod
            t_forecast['unidades'] = np.where(t_forecast['unidades'] < 1, 1, t_forecast['unidades'])
            rnn_forecast = pd.concat([rnn_forecast, t_forecast], ignore_index=True)
            #
            print('Modelo Red ajustados...\n')
            rnn_result['fecha'] = pd.to_datetime(rnn_result['fecha'])
            rnn_forecast['fecha'] = pd.to_datetime(rnn_forecast['fecha']) 
            t_rnn = pd.DataFrame()
            t_rnn = pd.concat([rnn_result, rnn_forecast], ignore_index=True)
            rnn = t_rnn.copy()
        #
    #

    ################################################################################################
    # PROPHET
    print('MODELO PROPHET | ' + str(dt.now()) + ' \n')
    fitted, forecast = utils.prophet_forecast(temp, forecast_period)
    if fitted is not None:
        # ajuste
        t_fit = pd.DataFrame()
        t_fit = pd.DataFrame({
            'fecha': temp['fecha'],
            'unidades': fitted.values.flatten()
        })
        t_fit['producto_id'] = prod
        t_fit['unidades'] = np.where(t_fit['unidades'] < 1, 1, t_fit['unidades'])
        prophet_result = pd.concat([prophet_result, t_fit], ignore_index=True)
        # pronostico
        t_forecast = pd.DataFrame()
        t_forecast = pd.DataFrame({
            'fecha': f_pronosticos['fecha'],
            'unidades': forecast.values.flatten()
        })
        t_forecast['producto_id'] = prod
        t_forecast['unidades'] = np.where(t_forecast['unidades'] < 1, 1, t_forecast['unidades'])
        prophet_forecast = pd.concat([prophet_forecast, t_forecast], ignore_index=True)
        #
        prophet_result['fecha'] = pd.to_datetime(prophet_result['fecha'])
        prophet_forecast['fecha'] = pd.to_datetime(prophet_forecast['fecha'])
        t_prophet = pd.DataFrame()
        t_prophet = pd.concat([prophet_result, prophet_forecast], ignore_index= True)
        prophet = t_prophet.copy()
    #
    print('FIN DE AJUSTE DE MODELOS | ' + str(dt.now()) + ' \n')
    #%

    ################################################################################################
    # integracion de resultados de ajustes
    print('INICIO DE TORNEO DE MODELOS | ' + str(dt.now()) + ' \n')
    #
    #% converitir producto_id a string
    temp['producto_id'] = temp['producto_id'].astype(str)
    arima_result['producto_id'] = arima_result['producto_id'].astype(str)
    lineal_result['producto_id'] = lineal_result['producto_id'].astype(str)
    poisson_result['producto_id'] = poisson_result['producto_id'].astype(str)
    rnn_result['producto_id'] = rnn_result['producto_id'].astype(str)
    prophet_result['producto_id'] = prophet_result['producto_id'].astype(str)
    #

    if t_resultado.empty:
        t_resultado = pd.merge(arima_result[['producto_id', 'fecha', 'unidades']].rename(columns={'unidades':'unidades_arima'}), 
                            lineal_result.rename(columns={'unidades':'unidades_lineal'}),  how='left', on=['fecha','producto_id'])
        t_resultado = pd.merge(t_resultado, poisson_result.rename(columns={'unidades':'unidades_poisson'}),  how='left', on=['fecha','producto_id'])
        t_resultado = pd.merge(t_resultado, rnn_result.rename(columns={'unidades':'unidades_rnn'}),  how='left', on=['fecha','producto_id'])
        t_resultado = pd.merge(t_resultado, prophet_result.rename(columns={'unidades':'unidades_prophet'}),  how='left', on=['fecha','producto_id'])
        t_resultado = pd.merge(t_resultado, temp[['fecha', 'producto_id', 'unidades', 'atipico', 'intervencion']], how='left', on=['fecha','producto_id'])
        #
        t_resultado['unidades_rnn'] = t_resultado['unidades_rnn'].fillna(1)
        t_resultado = t_resultado[t_resultado['producto_id'] == str(prod)].copy()
        print('despues de merging')
        print(t_resultado.info())

    # limites para torneo
    p10 = t_resultado['unidades'].quantile(0.1)
    p90 = t_resultado['unidades'].quantile(0.9)
    st_trunc = t_resultado[(t_resultado['unidades'] >= p10) & (t_resultado['unidades'] <= p90)]['unidades'].std()
    #
    t_resultado['lim_inf'] = t_resultado['unidades'] - 0.5*st_trunc
    t_resultado['lim_sup'] = t_resultado['unidades'] + 0.5*st_trunc
    t_resultado['lim_inf'] = np.where(t_resultado['lim_inf'] < 1, 1, t_resultado['lim_inf'])
    #
    t_resultado['arima_pts'] = np.where((t_resultado['unidades_arima'] >= t_resultado['lim_inf']) & (t_resultado['unidades_arima'] <= t_resultado['lim_sup']), 1, 0)
    t_resultado['lineal_pts'] = np.where((t_resultado['unidades_lineal'] >= t_resultado['lim_inf']) & (t_resultado['unidades_lineal'] <= t_resultado['lim_sup']), 1, 0)
    t_resultado['poisson_pts'] = np.where((t_resultado['unidades_poisson'] >= t_resultado['lim_inf']) & (t_resultado['unidades_poisson'] <= t_resultado['lim_sup']), 1, 0)
    t_resultado['rnn_pts'] = np.where((t_resultado['unidades_rnn'] >= t_resultado['lim_inf']) & (t_resultado['unidades_rnn'] <= t_resultado['lim_sup']), 1, 0)
    t_resultado['prophet_pts'] = np.where((t_resultado['unidades_prophet'] >= t_resultado['lim_inf']) & (t_resultado['unidades_prophet'] <= t_resultado['lim_sup']), 1, 0)
    #%

    pts_modelo = t_resultado.groupby('producto_id', as_index = False).agg(
        arima = ('arima_pts', 'sum'),
        lineal = ('lineal_pts', 'sum'),
        poisson = ('poisson_pts', 'sum'),
        rnn = ('rnn_pts', 'sum'),
        prophet = ('prophet_pts', 'sum')
    )
    #
    var_mod =[ 'arima', 'lineal', 'poisson', 'rnn', 'prophet']
    #
    max_col_index = pts_modelo[var_mod].apply(np.argmax, axis=1) + 1
    pts_modelo['modelo_ganador'] = max_col_index
    #
    print('FIN DE TORNEO DE MODELOS | ' + str(dt.now()) + ' \n')
    # %
    ################################################################################################
    # integracion de resultados de pronosticos
    arima_forecast['producto_id'] = arima_forecast['producto_id'].astype(str)
    lineal_forecast['producto_id'] = lineal_forecast['producto_id'].astype(str)
    poisson_forecast['producto_id'] = poisson_forecast['producto_id'].astype(str)
    rnn_forecast['producto_id'] = rnn_forecast['producto_id'].astype(str)
    prophet_forecast['producto_id'] = prophet_forecast['producto_id'].astype(str)

    #
    t_pronostico = pd.merge(arima_forecast[['producto_id', 'fecha', 'unidades']].rename(columns={'unidades':'unidades_arima'}), 
                            lineal_forecast.rename(columns={'unidades':'unidades_lineal'}),  how='left', on=['fecha','producto_id'])
    t_pronostico = pd.merge(t_pronostico, poisson_forecast.rename(columns={'unidades':'unidades_poisson'}),  how='left', on=['fecha','producto_id'])
    t_pronostico = pd.merge(t_pronostico, rnn_forecast.rename(columns={'unidades':'unidades_rnn'}),  how='left', on=['fecha','producto_id'])
    t_pronostico = pd.merge(t_pronostico, prophet_forecast.rename(columns={'unidades':'unidades_prophet'}),  how='left', on=['fecha','producto_id'])
    #
    ################################################################################################
    # integracion de resultados
    #
    unificacion = pd.concat([t_resultado[t_pronostico.columns], t_pronostico[t_pronostico['producto_id']==str(prod)]], ignore_index=True)
    #
    if max_col_index[0] == 1:
        u_temp = unificacion['unidades_arima']
    elif max_col_index[0] == 2:
        u_temp = unificacion['unidades_lineal']
    elif max_col_index[0] == 3:
        u_temp = unificacion['unidades_poisson']
    elif max_col_index[0] == 4:
        u_temp = unificacion['unidades_rnn']
    elif max_col_index[0] == 5:
        u_temp = unificacion['unidades_prophet']
    #
    to_forecast = pd.DataFrame({
        'unidades': u_temp
    }, index = unificacion.index)
    to_forecast['producto_id'] = prod
    to_forecast['fecha'] = unificacion['fecha']
    #
    # guardar resultado de torneo de modelos para el producto
    pts_modelo['ejecucion'] = _ejecucion
    pts_modelo['producto_id'] = pts_modelo['producto_id'].astype(str)
    torneo_result = pd.concat([torneo_result, pts_modelo], ignore_index=True)
    #
    to_forecast['ejecucion'] = _ejecucion
    to_forecast['producto_id'] = to_forecast['producto_id'].astype(str)
    to_forecast['fecha'] = pd.to_datetime(to_forecast['fecha'])
    forecast_result = pd.concat([forecast_result, to_forecast], ignore_index=True)
    #
    # guardar resultados globales
    t_resultado['ejecucion'] = _ejecucion
    t_resultado['producto_id'] = t_resultado['producto_id'].astype(str)
    #
    resultados_globales = pd.concat([resultados_globales, t_resultado], ignore_index=True)
    #
    print(f'\nFIN PRODUCTO {prod} | {a} DE {modelable.shape[0]} | {str(dt.now())}\n')
#%%

###############################################################################################
# guardar resultados
print(f'\nALMACENADO DE RESULTADOS | {str(dt.now())}')
#
print(f'\nGuardado de archivos csv | {str(dt.now())}')
# modelos individuales
print(f"Ruta de guardado de csv {dir_resultados}\n")
arima.to_csv(f"{dir_resultados}\\{_ejecucion}_arima.csv", index=False)
lineal.to_csv(f"{dir_resultados}\\{_ejecucion}_lineal.csv", index=False)
poisson.to_csv(f"{dir_resultados}\\{_ejecucion}_poisson.csv", index=False)
rnn.to_csv(f"{dir_resultados}\\{_ejecucion}_rnn.csv", index=False)
prophet.to_csv(f"{dir_resultados}\\{_ejecucion}_prophet.csv", index=False)
# resultados globales
try:
    resultados_globales['fecha'] = pd.to_datetime(resultados_globales['fecha'])
except:
    pass
resultados_globales.to_csv(f"{dir_resultados}\\{_ejecucion}_resultados_globales.csv", index=False)
resultados_globales[['producto_id','fecha','atipico','intervencion','ejecucion']].to_csv(f"{dir_resultados}\\{_ejecucion}_atipicos_intervencion.csv", index = False)
# torneo de modelos
try:
    torneo_result['fecha'] = pd.to_datetime(torneo_result['fecha'])
except:
    pass
torneo_result.to_csv(f"{dir_resultados}\\{_ejecucion}_torneo_modelos.csv", index=False)
# pronosticos
var_sal = ['producto_id', 'fecha', 'unidades', 'ejecucion']
forecast_result[var_sal].to_csv(f"{dir_resultados}\\{_ejecucion}_pronostico.csv", index=False)
# datos entrada
datos.to_csv(f"{dir_resultados}\\{_ejecucion}_datos_entrada.csv", index=False)

# guardar resultados csv
utils.arc_to_zip(dir_resultados, f"{dir_resultados}\\{_ejecucion}_resultados_ejecucion.zip")
#
print(f'\nGuardado en base de datos | {str(dt.now())}')
#forecast_result[var_sal].to_sql('tdw_forecast', con = utils.engine, schema = 'modelos', if_exists = 'append', index = False, chunksize= 5000)

#torneo_result.to_sql('tdw_resultado_torneo', con = utils.engine, schema = 'modelos', if_exists = 'append', index = False, chunksize= 5000)

###############################################################################################
_fin = dt.now()
print('\nESTIMACION TERMINADA... | ' + str(dt.now()))
print('\nTiempo de ejecucion: ' + str(_fin-_inicio) + '\n\n')
###############################################################################################










# %%
