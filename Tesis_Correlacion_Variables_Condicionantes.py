#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import pearsonr
# from mpl_toolkits.axes_grid1 import host_subplot
# import mpl_toolkits.axisartist as AA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib.cm as cm
import matplotlib.font_manager as fm
import math as m
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
import matplotlib.colors as colors
import netCDF4 as nc
from netCDF4 import Dataset
import os
from scipy.stats import pearsonr

#------------------------------------------------------------------------------
# Motivación codigo -----------------------------------------------------------

"Codigo para determinar en nivel de correlacion entre las variables  incidentes en la potencia de los paneles,"
"por lo cual se hace para el horizonte de tiempo de desde que se tengan datos del experimento de los paneles."
"Se hace con la correlación lineal de Pearson."

############################################################################################
## ----------------ACOTANDO LAS FECHAS POR DIA Y MES PARA TOMAR LOS DATOS---------------- ##
############################################################################################

fi_m = 3
fi_d = 23
ff_m = 12
ff_d = 20
Anio_datosGOES = 2019
Path_save = '/home/nacorreasa/Maestria/Datos_Tesis/Arrays/'
Significancia = False ##--->Para que grafique los valores P y elimine las relaciones poco significantes debe ser True
Solo_Meteo = True     ##--->Para que tome solo los df con info meteorologica debe ser True

##############################################################################
#-----------------------------------------------------------------------------
# Rutas para las fuentes -----------------------------------------------------
##############################################################################

prop = fm.FontProperties(fname='/home/nacorreasa/SIATA/Cod_Califi/AvenirLTStd-Heavy.otf' )
prop_1 = fm.FontProperties(fname='/home/nacorreasa/SIATA/Cod_Califi/AvenirLTStd-Book.otf')
prop_2 = fm.FontProperties(fname='/home/nacorreasa/SIATA/Cod_Califi/AvenirLTStd-Black.otf')

##############################################################################
## ----------------LECTURA DE LOS DATOS DE LOS EXPERIMENTOS---------------- ##
##############################################################################

df_P975 = pd.read_csv('/home/nacorreasa/Maestria/Datos_Tesis/Experimentos_Panel/Panel975.txt',  sep=',', index_col =0)
df_P350 = pd.read_csv('/home/nacorreasa/Maestria/Datos_Tesis/Experimentos_Panel/Panel350.txt',  sep=',', index_col =0)
df_P348 = pd.read_csv('/home/nacorreasa/Maestria/Datos_Tesis/Experimentos_Panel/Panel348.txt',  sep=',', index_col =0)


df_P975.index = pd.to_datetime(df_P975.index, format="%Y-%m-%d %H:%M:%S", errors='coerce')
df_P350.index = pd.to_datetime(df_P350.index, format="%Y-%m-%d %H:%M:%S", errors='coerce')
df_P348.index = pd.to_datetime(df_P348.index, format="%Y-%m-%d %H:%M:%S", errors='coerce')

## ----------------ACOTANDO LOS DATOS A CALORES VÁLIDOS---------------- ##

df_P975 = df_P975[df_P975['radiacion'] > 0]
df_P350 = df_P350[df_P350['radiacion'] > 0]
df_P348 = df_P348[df_P348['radiacion'] > 0]

df_P975 = df_P975[(df_P975['NI'] >= 0) & (df_P975['strength'] >= 0) & (df_P975['strength'] <=100)]
df_P350 = df_P350[(df_P350['NI'] >= 0) & (df_P350['strength'] >= 0) & (df_P350['strength'] <=100)]
df_P348 = df_P348[(df_P348['NI'] >= 0) & (df_P348['strength'] >= 0) & (df_P348['strength'] <=100)]

df_P975 = df_P975[(df_P975['strength'] >= 0)]
df_P350 = df_P350[(df_P350['NI'] >= 0) & (df_P350['strength'] >= 0)]
df_P348 = df_P348[(df_P348['NI'] >= 0) & (df_P348['strength'] >= 0)]

df_P975_h = df_P975.groupby(pd.Grouper(freq="H")).mean()
df_P350_h = df_P350.groupby(pd.Grouper(freq="H")).mean()
df_P348_h = df_P348.groupby(pd.Grouper(freq="H")).mean()

df_P975_h = df_P975_h[(df_P975_h.index >= pd.to_datetime('2019-'+str(fi_m)+ '-'
            +str(fi_d), errors='coerce')) & (df_P975_h.index
            <= pd.to_datetime('2019-'+str(ff_m)+ '-'+str(ff_d), errors='coerce') )]

df_P350_h = df_P350_h[(df_P350_h.index >= pd.to_datetime('2019-'+str(fi_m)+ '-'
            +str(fi_d), errors='coerce')) & (df_P350_h.index
            <= pd.to_datetime('2019-'+str(ff_m)+ '-'+str(ff_d), errors='coerce') )]

df_P348_h = df_P348_h[(df_P348_h.index >= pd.to_datetime('2019-'+str(fi_m)+ '-'
            +str(fi_d), errors='coerce')) & (df_P348_h.index
            <= pd.to_datetime('2019-'+str(ff_m)+ '-'+str(ff_d), errors='coerce') )]

df_P975_h = df_P975_h.drop(['NI', 'radiacion'], axis=1)
df_P350_h = df_P350_h.drop(['NI', 'radiacion'], axis=1)
df_P348_h = df_P348_h.drop(['NI', 'radiacion'], axis=1)
##########################################################################################
## ----------------LECTURA DE LOS DATOS DE LAS ANOMALIAS DE LA RADIACION--------------- ##
##########################################################################################

Anomal_df_975 = pd.read_csv('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/df_AnomalRad_pix975_2018_2019.csv',  sep=',')
Anomal_df_348 = pd.read_csv('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/df_AnomalRad_pix348_2018_2019.csv',  sep=',')
Anomal_df_350 = pd.read_csv('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/df_AnomalRad_pix350_2018_2019.csv',  sep=',')

Anomal_df_975['fecha_hora'] = pd.to_datetime(Anomal_df_975['fecha_hora'], format="%Y-%m-%d %H:%M", errors='coerce')
Anomal_df_975.index = Anomal_df_975['fecha_hora']
Anomal_df_975 = Anomal_df_975.drop(['fecha_hora'], axis=1)
Anomal_df_975 = Anomal_df_975.between_time('06:00', '18:00')                      ##--> Seleccionar solo los datos de horas del dia
Anomal_df_975_h = Anomal_df_975.groupby(pd.Grouper(freq="H")).mean()

Anomal_df_350['fecha_hora'] = pd.to_datetime(Anomal_df_350['fecha_hora'], format="%Y-%m-%d %H:%M", errors='coerce')
Anomal_df_350.index = Anomal_df_350['fecha_hora']
Anomal_df_350 = Anomal_df_350.drop(['fecha_hora'], axis=1)
Anomal_df_350 = Anomal_df_350.between_time('06:00', '18:00')                      ##--> Seleccionar solo los datos de horas del dia
Anomal_df_350_h = Anomal_df_350.groupby(pd.Grouper(freq="H")).mean()

Anomal_df_348['fecha_hora'] = pd.to_datetime(Anomal_df_348['fecha_hora'], format="%Y-%m-%d %H:%M", errors='coerce')
Anomal_df_348.index = Anomal_df_348['fecha_hora']
Anomal_df_348 = Anomal_df_348.drop(['fecha_hora'], axis=1)
Anomal_df_348 = Anomal_df_348.between_time('06:00', '18:00')                      ##--> Seleccionar solo los datos de horas del dia
Anomal_df_348_h = Anomal_df_348.groupby(pd.Grouper(freq="H")).mean()

Anomal_df_975_h = Anomal_df_975_h[(Anomal_df_975_h.index >= pd.to_datetime('2019-'+str(fi_m)+ '-'+str(fi_d))) & (Anomal_df_975_h.index <= pd.to_datetime('2019-'+str(ff_m)+ '-'+str(ff_d)))]
Anomal_df_350_h = Anomal_df_350_h[(Anomal_df_350_h.index >= pd.to_datetime('2019-'+str(fi_m)+ '-'+str(fi_d))) & (Anomal_df_350_h.index <= pd.to_datetime('2019-'+str(ff_m)+ '-'+str(ff_d)))]
Anomal_df_348_h = Anomal_df_348_h[(Anomal_df_348_h.index >= pd.to_datetime('2019-'+str(fi_m)+ '-'+str(fi_d))) & (Anomal_df_348_h.index <= pd.to_datetime('2019-'+str(ff_m)+ '-'+str(ff_d)))]

Anomal_df_348_h = Anomal_df_348_h.drop(['Radiacion_Med', 'radiacion',], axis=1)
Anomal_df_350_h = Anomal_df_350_h.drop(['Radiacion_Med', 'radiacion',], axis=1)
Anomal_df_975_h = Anomal_df_975_h.drop(['Radiacion_Med', 'radiacion',], axis=1)

Anomal_df_348_h = Anomal_df_348_h.loc[~Anomal_df_348_h.index.duplicated(keep='first')]
Anomal_df_350_h = Anomal_df_350_h.loc[~Anomal_df_350_h.index.duplicated(keep='first')]
Anomal_df_975_h = Anomal_df_975_h.loc[~Anomal_df_975_h.index.duplicated(keep='first')]

################################################################################
## ----------------LECTURA DE LOS DATOS DE LAS METEOROLÓGICAS---------------- ##
################################################################################

data_T_Torre = pd.read_csv('/home/nacorreasa/Maestria/Datos_Tesis/Meteorologicas/Entrega_TH_CuRad201.txt',  sep=',')
data_T_Conse = pd.read_csv('/home/nacorreasa/Maestria/Datos_Tesis/Meteorologicas/Entrega_TH_CuRad206.txt',  sep=',')
data_T_Joaqu = pd.read_csv('/home/nacorreasa/Maestria/Datos_Tesis/Meteorologicas/Entrega_TH_CuRad367.txt',  sep=',')

data_T_Torre.index = data_T_Torre['fecha_hora']
data_T_Torre = data_T_Torre.drop(['fecha_hora'], axis=1)
data_T_Torre.index = pd.to_datetime(data_T_Torre.index)
data_T_Torre = data_T_Torre.between_time('06:00', '18:00')                      ##--> Seleccionar solo los datos de horas del dia
data_T_Torre = data_T_Torre[data_T_Torre[u'calidad'] < 100]
data_T_Torre = data_T_Torre[data_T_Torre['T'] > 0]
data_T_Torre_h = data_T_Torre.groupby(pd.Grouper(freq="H")).mean()

data_T_Conse.index = data_T_Conse['fecha_hora']
data_T_Conse = data_T_Conse.drop(['fecha_hora'], axis=1)
data_T_Conse.index = pd.to_datetime(data_T_Conse.index)
data_T_Conse = data_T_Conse.between_time('06:00', '18:00')                      ##--> Seleccionar solo los datos de horas del dia
data_T_Conse = data_T_Conse[data_T_Conse[u'calidad'] < 100]
data_T_Conse = data_T_Conse[data_T_Conse['T'] > 0]
data_T_Conse_h = data_T_Conse.groupby(pd.Grouper(freq="H")).mean()

data_T_Joaqu.index = data_T_Joaqu['fecha_hora']
data_T_Joaqu = data_T_Joaqu.drop(['fecha_hora'], axis=1)
data_T_Joaqu.index = pd.to_datetime(data_T_Joaqu.index)
data_T_Joaqu = data_T_Joaqu.between_time('06:00', '18:00')                      ##--> Seleccionar solo los datos de horas del dia
data_T_Joaqu = data_T_Joaqu[data_T_Joaqu[u'calidad'] < 100]
data_T_Joaqu = data_T_Joaqu[data_T_Joaqu['T'] > 0]
data_T_Joaqu_h = data_T_Joaqu.groupby(pd.Grouper(freq="H")).mean()

data_T_Torre_h = data_T_Torre_h[(data_T_Torre_h.index >= pd.to_datetime('2019-'+str(fi_m)+ '-'
            +str(fi_d), errors='coerce')) & (data_T_Torre_h.index
            <= pd.to_datetime('2019-'+str(ff_m)+ '-'+str(ff_d), errors='coerce') )]

data_T_Conse_h = data_T_Conse_h[(data_T_Conse_h.index >= pd.to_datetime('2019-'+str(fi_m)+ '-'
            +str(fi_d), errors='coerce')) & (data_T_Conse_h.index
            <= pd.to_datetime('2019-'+str(ff_m)+ '-'+str(ff_d), errors='coerce') )]

data_T_Joaqu_h = data_T_Joaqu_h[(data_T_Joaqu_h.index >= pd.to_datetime('2019-'+str(fi_m)+ '-'
            +str(fi_d), errors='coerce')) & (data_T_Joaqu_h.index
            <= pd.to_datetime('2019-'+str(ff_m)+ '-'+str(ff_d), errors='coerce') )]

data_T_Torre_h = data_T_Torre_h.drop(['codigo', 'calidad', 'H'], axis=1)
data_T_Conse_h = data_T_Conse_h.drop(['codigo', 'calidad', 'H'], axis=1)
data_T_Joaqu_h = data_T_Joaqu_h.drop(['codigo', 'calidad', 'H'], axis=1)

#########################################################################
## ----------------LECTURA DE LOS DATOS DEL RADIOMETRO---------------- ##
#########################################################################

data_Radi_WV = pd.read_csv('/home/nacorreasa/Maestria/Datos_Tesis/Radiometro/Radiometro_Panel/Integrated_VaporDensity_values.csv',  sep=',')
data_Radi_WV.index = data_Radi_WV[u'Unnamed: 0']
data_Radi_WV.index = pd.to_datetime(data_Radi_WV.index, format="%Y-%m-%d %H:%M:%S", errors='coerce')
data_Radi_WV = data_Radi_WV.between_time('06:00', '17:00')                      ##--> Seleccionar solo los datos de horas del dia
data_Radi_WV_h = data_Radi_WV[u'Integrate'].groupby(pd.Grouper(freq="H")).mean()

data_Radi_WV_h = data_Radi_WV_h[(data_Radi_WV_h.index >= pd.to_datetime('2019-'+str(fi_m)+ '-'
            +str(fi_d),  errors='coerce')) & (data_Radi_WV_h.index
            <= pd.to_datetime('2019-'+str(ff_m)+ '-'+str(ff_d),  errors='coerce') )]


#################################################################################################
##-------------------LECTURA DE LOS DATOS DE CH2 GOES PARA CADA PIXEL--------------------------##
#################################################################################################

Rad_pixel_975 = np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/Array_Rad_pix975_EXP.npy')
Rad_pixel_350 = np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/Array_Rad_pix350_EXP.npy')
Rad_pixel_348 = np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/Array_Rad_pix348_EXP.npy')
fechas_horas = np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/Array_FechasHoras_EXP.npy')

df_fh  = pd.DataFrame()
df_fh ['fecha_hora'] = fechas_horas
df_fh['fecha_hora'] = pd.to_datetime(df_fh['fecha_hora'], format="%Y-%m-%d %H:%M", errors='coerce')
df_fh.index = df_fh['fecha_hora']
w = pd.date_range(df_fh.index.min(), df_fh.index.max()).difference(df_fh.index)
df_fh = df_fh[df_fh.index.hour != 5]
fechas_horas = df_fh['fecha_hora'].values


                   ## -- Selección del pixel de la TS
Rad_df_975 = pd.DataFrame()
Rad_df_975['Fecha_Hora'] = fechas_horas
Rad_df_975['Radiacias'] = Rad_pixel_975
Rad_df_975['Fecha_Hora'] = pd.to_datetime(Rad_df_975['Fecha_Hora'], format="%Y-%m-%d %H:%M", errors='coerce')
Rad_df_975.index = Rad_df_975['Fecha_Hora']
Rad_df_975 = Rad_df_975.drop(['Fecha_Hora'], axis=1)

                   ## -- Selección del pixel de la CI

Rad_df_350 = pd.DataFrame()
Rad_df_350['Fecha_Hora'] = fechas_horas
Rad_df_350['Radiacias'] = Rad_pixel_350
Rad_df_350['Fecha_Hora'] = pd.to_datetime(Rad_df_350['Fecha_Hora'], format="%Y-%m-%d %H:%M", errors='coerce')
Rad_df_350.index = Rad_df_350['Fecha_Hora']
Rad_df_350 = Rad_df_350.drop(['Fecha_Hora'], axis=1)


                   ## -- Selección del pixel de la JV

Rad_df_348 = pd.DataFrame()
Rad_df_348['Fecha_Hora'] = fechas_horas
Rad_df_348['Radiacias'] = Rad_pixel_348
Rad_df_348['Fecha_Hora'] = pd.to_datetime(Rad_df_348['Fecha_Hora'], format="%Y-%m-%d %H:%M", errors='coerce')
Rad_df_348.index = Rad_df_348['Fecha_Hora']
Rad_df_348 = Rad_df_348.drop(['Fecha_Hora'], axis=1)

Rad_df_975_h = Rad_df_975.groupby(pd.Grouper(freq="H")).mean()
Rad_df_350_h = Rad_df_350.groupby(pd.Grouper(freq="H")).mean()
Rad_df_348_h = Rad_df_348.groupby(pd.Grouper(freq="H")).mean()

if Anio_datosGOES == 2018:
    Rad_df_975_h.index = [Rad_df_975_h.index[i].replace(year=2019) for i in range(len(Rad_df_975_h.index))]
    Rad_df_350_h.index = [Rad_df_350_h.index[i].replace(year=2019) for i in range(len(Rad_df_350_h.index))]
    Rad_df_348_h.index = [Rad_df_348_h.index[i].replace(year=2019) for i in range(len(Rad_df_348_h.index))]

Rad_df_975_h = Rad_df_975_h[(Rad_df_975_h.index >= pd.to_datetime('2019-'+str(fi_m)+ '-'+str(fi_d))) & (Rad_df_975_h.index <= pd.to_datetime('2019-'+str(ff_m)+ '-'+str(ff_d)))]
Rad_df_350_h = Rad_df_350_h[(Rad_df_350_h.index >= pd.to_datetime('2019-'+str(fi_m)+ '-'+str(fi_d))) & (Rad_df_350_h.index <= pd.to_datetime('2019-'+str(ff_m)+ '-'+str(ff_d)))]
Rad_df_348_h = Rad_df_348_h[(Rad_df_348_h.index >= pd.to_datetime('2019-'+str(fi_m)+ '-'+str(fi_d))) & (Rad_df_348_h.index <= pd.to_datetime('2019-'+str(ff_m)+ '-'+str(ff_d)))]

Rad_df_348_h = Rad_df_348_h.loc[~Rad_df_348_h.index.duplicated(keep='first')]
Rad_df_350_h = Rad_df_350_h.loc[~Rad_df_350_h.index.duplicated(keep='first')]
Rad_df_975_h = Rad_df_975_h.loc[~Rad_df_975_h.index.duplicated(keep='first')]

################################################################################
## ----------------LECTURA DE LOS DATOS DE EFICIENCIA NOMINAL---------------- ##
################################################################################

"Obtenidos del programa Tesis_Eficiencia_Panel.py"

df_nomEfi_975 = pd.read_csv('/home/nacorreasa/Maestria/Datos_Tesis/Experimentos_Panel/Efi_tiempo_nominal_P975.csv',  sep=',')
df_nomEfi_350 = pd.read_csv('/home/nacorreasa/Maestria/Datos_Tesis/Experimentos_Panel/Efi_tiempo_nominal_P350.csv',  sep=',')
df_nomEfi_348 = pd.read_csv('/home/nacorreasa/Maestria/Datos_Tesis/Experimentos_Panel/Efi_tiempo_nominal_P348.csv',  sep=',')

df_nomEfi_975['fecha_hora'] = pd.to_datetime(df_nomEfi_975['fecha_hora'], format="%Y-%m-%d %H:%M", errors='coerce')
df_nomEfi_975.index =  df_nomEfi_975['fecha_hora']

df_nomEfi_350['fecha_hora'] = pd.to_datetime(df_nomEfi_350['fecha_hora'], format="%Y-%m-%d %H:%M", errors='coerce')
df_nomEfi_350.index =  df_nomEfi_350['fecha_hora']

df_nomEfi_348['fecha_hora'] = pd.to_datetime(df_nomEfi_348['fecha_hora'], format="%Y-%m-%d %H:%M", errors='coerce')
df_nomEfi_348.index =  df_nomEfi_348['fecha_hora']

df_nomEfi_975 = df_nomEfi_975[(df_nomEfi_975.index >= pd.to_datetime('2019-'+str(fi_m)+ '-'
            +str(fi_d),  errors='coerce')) & (df_nomEfi_975.index
            <= pd.to_datetime('2019-'+str(ff_m)+ '-'+str(ff_d),  errors='coerce') )]

df_nomEfi_350 = df_nomEfi_350[(df_nomEfi_350.index >= pd.to_datetime('2019-'+str(fi_m)+ '-'
            +str(fi_d),  errors='coerce')) & (df_nomEfi_350.index
            <= pd.to_datetime('2019-'+str(ff_m)+ '-'+str(ff_d),  errors='coerce') )]

df_nomEfi_348 = df_nomEfi_348[(df_nomEfi_348.index >= pd.to_datetime('2019-'+str(fi_m)+ '-'
            +str(fi_d),  errors='coerce')) & (df_nomEfi_348.index
            <= pd.to_datetime('2019-'+str(ff_m)+ '-'+str(ff_d),  errors='coerce') )]

##------------------------ACOTANDOLO A VALORES VÁLIDOS--------------------------##
df_nomEfi_975 = df_nomEfi_975[df_nomEfi_975.Efi <=100]
df_nomEfi_350 = df_nomEfi_350[df_nomEfi_350.Efi <=100]
df_nomEfi_348 = df_nomEfi_975[df_nomEfi_975.Efi <=100]

df_nomEfi_975 = df_nomEfi_975.drop(['fecha_hora'], axis=1)
df_nomEfi_348 = df_nomEfi_348.drop(['fecha_hora'], axis=1)
df_nomEfi_350 = df_nomEfi_350.drop(['fecha_hora'], axis=1)

################################################################################
## -----------------------LECTURA DE LOS DATOS DE PM 2.5--------------------- ##
################################################################################
"Se toman los datos de PM 2.5 como indicados de las particulas que pueden dispersar la radiación solar."
"Los puntos a considerar son : la estacion 25 de UN Agronomia, la 80 en Villa Hermosa y la 38 en el Consejo"
"de Itagüí"

data_PM_TS = pd.read_csv('/home/nacorreasa/Maestria/Datos_Tesis/PM_2con5/Total_Timeseries_PM25.csv',  sep=',')
data_PM_JV = pd.read_csv('/home/nacorreasa/Maestria/Datos_Tesis/PM_2con5/Total_Timeseries_PM80.csv',  sep=',')
data_PM_CI = pd.read_csv('/home/nacorreasa/Maestria/Datos_Tesis/PM_2con5/Total_Timeseries_PM38.csv',  sep=',')

df_PM_TS = data_PM_TS[['Unnamed: 0', 'codigoSerial', 'pm25', 'calidad_pm25']]
df_PM_JV = data_PM_JV[['Unnamed: 0', 'codigoSerial', 'pm25', 'calidad_pm25']]
df_PM_CI = data_PM_CI[['Unnamed: 0', 'codigoSerial', 'pm25', 'calidad_pm25']]

df_PM_TS.columns = ['fecha_hora', 'codigoSerial', 'pm25', 'calidad_pm25']
df_PM_CI.columns = ['fecha_hora', 'codigoSerial', 'pm25', 'calidad_pm25']
df_PM_JV.columns = ['fecha_hora', 'codigoSerial', 'pm25', 'calidad_pm25']

df_PM_TS = df_PM_TS[df_PM_TS['calidad_pm25'] ==1]
df_PM_JV = df_PM_JV[df_PM_JV['calidad_pm25'] ==1]
df_PM_CI = df_PM_CI[df_PM_CI['calidad_pm25'] ==1]

df_PM_TS['fecha_hora'] = pd.to_datetime(df_PM_TS['fecha_hora'], format="%Y-%m-%d %H:%M", errors='coerce')
df_PM_TS.index = df_PM_TS['fecha_hora']
df_PM_TS = df_PM_TS.drop(['fecha_hora', 'codigoSerial', 'calidad_pm25'], axis=1)

df_PM_CI['fecha_hora'] = pd.to_datetime(df_PM_CI['fecha_hora'], format="%Y-%m-%d %H:%M", errors='coerce')
df_PM_CI.index = df_PM_CI['fecha_hora']
df_PM_CI = df_PM_CI.drop(['fecha_hora', 'codigoSerial', 'calidad_pm25'], axis=1)

df_PM_JV['fecha_hora'] = pd.to_datetime(df_PM_JV['fecha_hora'], format="%Y-%m-%d %H:%M", errors='coerce')
df_PM_JV.index = df_PM_JV['fecha_hora']
df_PM_JV = df_PM_JV.drop(['fecha_hora', 'codigoSerial', 'calidad_pm25'], axis=1)

df_PM_JV =  df_PM_JV.loc[~df_PM_JV.index.duplicated(keep='first')]
df_PM_CI =  df_PM_CI.loc[~df_PM_CI.index.duplicated(keep='first')]
df_PM_TS =  df_PM_TS.loc[~df_PM_TS.index.duplicated(keep='first')]

df_PM_TS = df_PM_TS[(df_PM_TS.index >= pd.to_datetime('2019-'+str(fi_m)+ '-'
            +str(fi_d),  errors='coerce')) & (df_PM_TS.index
            <= pd.to_datetime('2019-'+str(ff_m)+ '-'+str(ff_d),  errors='coerce') )]

df_PM_JV = df_PM_JV[(df_PM_JV.index >= pd.to_datetime('2019-'+str(fi_m)+ '-'
            +str(fi_d),  errors='coerce')) & (df_PM_JV.index
            <= pd.to_datetime('2019-'+str(ff_m)+ '-'+str(ff_d),  errors='coerce') )]

df_PM_CI = df_PM_CI[(df_PM_CI.index >= pd.to_datetime('2019-'+str(fi_m)+ '-'
            +str(fi_d),  errors='coerce')) & (df_PM_CI.index
            <= pd.to_datetime('2019-'+str(ff_m)+ '-'+str(ff_d),  errors='coerce') )]

df_PM_TS = df_PM_TS.between_time('06:00', '17:59')
df_PM_CI = df_PM_CI.between_time('06:00', '17:59')
df_PM_JV = df_PM_JV.between_time('06:00', '17:59')

#################################################################################
## ----------------CORRELACION ENTRE LAS VARIABLES PRINCIPALES---------------- ##
#################################################################################

if Solo_Meteo == False:

    df_corr_975 =  pd.concat([Anomal_df_975_h, df_P975_h ,  data_T_Torre_h ,data_Radi_WV_h , Rad_df_975_h, df_nomEfi_975, df_PM_TS], axis=1)
    df_corr_350 =  pd.concat([Anomal_df_350_h, df_P350_h , data_T_Conse_h, data_Radi_WV_h , Rad_df_350_h, df_nomEfi_350, df_PM_CI], axis=1)
    df_corr_348 =  pd.concat([Anomal_df_348_h, df_P348_h ,  data_T_Joaqu_h, data_Radi_WV_h , Rad_df_348_h, df_nomEfi_348, df_PM_JV], axis=1)

    df_corr_975.columns = ['I Anomaly', 'Power', 'Temp', u'WV Den', 'FR', u'$\eta$', u'PM 2.5']
    df_corr_350.columns = ['I Anomaly', 'Power', 'Temp', u'WV Den', 'FR', u'$\eta$', u'PM 2.5']
    df_corr_348.columns = ['I Anomaly', 'Power', 'Temp', u'WV Den', 'FR', u'$\eta$', u'PM 2.5']

    corr_975 = df_corr_975.corr(method = 'pearson')
    corr_350 = df_corr_350.corr(method = 'pearson')
    corr_348 = df_corr_348.corr(method = 'pearson')

    nombre = 'ConMeteo'

elif Solo_Meteo == True:

    df_corr_975 =  pd.concat([Anomal_df_975_h,  data_T_Torre_h ,data_Radi_WV_h , Rad_df_975_h,  df_PM_TS], axis=1)
    df_corr_350 =  pd.concat([Anomal_df_350_h, data_T_Conse_h, data_Radi_WV_h , Rad_df_350_h,  df_PM_CI], axis=1)
    df_corr_348 =  pd.concat([Anomal_df_348_h,  data_T_Joaqu_h, data_Radi_WV_h , Rad_df_348_h,  df_PM_JV], axis=1)

    df_corr_975.columns = ['I Anomaly',  'Temp', u'WV Den', 'FR',  u'PM 2.5']
    df_corr_350.columns = ['I Anomaly', 'Temp', u'WV Den', 'FR',  u'PM 2.5']
    df_corr_348.columns = ['I Anomaly', 'Temp', u'WV Den', 'FR',  u'PM 2.5']

    corr_975 = df_corr_975.corr(method = 'pearson')
    corr_350 = df_corr_350.corr(method = 'pearson')
    corr_348 = df_corr_348.corr(method = 'pearson')

    nombre = 'SinMeteo'

################################################################################
##--------------------------VALOR P DE LA CORRELACIÓN-------------------------##
################################################################################

def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues

pval_975 = calculate_pvalues(df_corr_975)
pval_350 = calculate_pvalues(df_corr_350)
pval_348 = calculate_pvalues(df_corr_348)

pval_975.to_csv('/home/nacorreasa/Escritorio/pval_975.csv')
pval_350.to_csv('/home/nacorreasa/Escritorio/pval_350.csv')
pval_348.to_csv('/home/nacorreasa/Escritorio/pval_348.csv')

###################################################################################################
##----------------------------------------BARRA DE COLORES A USAR--------------------------------##
###################################################################################################

def newjet():
    """función para crear un nuevo color bar con cero en la mitad, modeificando el color bar jet"""
    jetcmap = cm.get_cmap("jet", 11)  #generate a jet map with 11 values
    jet_vals = jetcmap(np.arange(11)) #extract those values as an array
    jet_vals[5] = [1, 1, 1, 1]        #change the middle value
    newcmap = colors.LinearSegmentedColormap.from_list("newjet", jet_vals)
    return newcmap

my_cbar=newjet()

###################################################################################################
origin_cmap = cm.get_cmap("bwr", 14)
origin_vals = origin_cmap(np.arange(14))
withe = np.array([1,1,1,1])
for i in range(3):
    origin_vals = np.insert(origin_vals, 7,withe, axis=0)

colrs=origin_vals
levels=np.arange(-1,1.1,0.1)
name='coqueto'
cmap_new       = colors.LinearSegmentedColormap.from_list(name,colrs)
levels_nuevos  = np.linspace(np.min(levels),np.max(levels),255)
#levels_nuevos  = np.linspace(-0.75,np.max(levels),255)
norm_new       = colors.BoundaryNorm(boundaries=levels_nuevos, ncolors=256)

############################################################
## ----------------GRÁFICA DE CORRELACIÓN---------------- ##
############################################################

fig = plt.figure(figsize=[10, 5])

ax1 = fig.add_subplot(131)
cax1 = ax1.imshow(corr_350,interpolation = 'none', cmap=cmap_new, norm=norm_new, vmin=-1, vmax=1)
ticks = np.arange(0, len(df_corr_350.columns), 1)
ax1.set_xticks(ticks)
plt.xticks(rotation=90)
ax1.set_title('Correlations in West', fontsize = 10, fontproperties = prop_1)
ax1.set_yticks(ticks)
ax1.set_xticklabels(df_corr_350.columns, rotation = 45, fontsize = 7)
ax1.set_yticklabels(df_corr_350.columns, rotation = 45, fontsize = 7)

ax2 = fig.add_subplot(132)
cax2 = ax2.imshow(corr_975,interpolation = 'none', cmap=cmap_new, norm=norm_new, vmin=-1, vmax=1)
ticks = np.arange(0, len(df_corr_975.columns), 1)
ax2.set_xticks(ticks)
plt.xticks(rotation=90)
ax2.set_title('Correlations in West Center', fontsize = 10, fontproperties = prop_1)
ax2.set_yticks(ticks)
ax2.set_xticklabels(df_corr_975.columns, rotation = 45, fontsize = 7)
ax2.set_yticklabels(df_corr_975.columns, rotation = 45, fontsize = 7)

ax3 = fig.add_subplot(133)
cax3 = ax3.imshow(corr_348,interpolation = 'none', cmap=cmap_new, norm=norm_new, vmin=-1, vmax=1)
ticks = np.arange(0, len(df_corr_348.columns), 1)
ax3.set_xticks(ticks)
plt.xticks(rotation=90)
ax3.set_title('Correlations in East', fontsize = 10, fontproperties = prop_1)
ax3.set_yticks(ticks)
ax3.set_xticklabels(df_corr_348.columns, rotation = 45, fontsize = 7)
ax3.set_yticklabels(df_corr_348.columns, rotation = 45, fontsize = 7)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.30, 0.015, 0.40 ])
cbar = fig.colorbar(cax3, label = u"Correlation coefficient", cax=cbar_ax)
cbar.set_ticks([-1, -0.75, -0.5,-0.25,  0.0,0.25, 0.5, 0.75, 1])
cbar.set_ticklabels([str(i) for i in [-1, -0.75, -0.5,-0.25,  0.0,0.25, 0.5, 0.75, 1]])
plt.subplots_adjust(wspace=0.3)
plt.savefig('/home/nacorreasa/Escritorio/Figuras/Corr_imshowP_'+nombre+'.pdf', format='pdf')
plt.close('all')
os.system('scp /home/nacorreasa/Escritorio/Figuras/Corr_imshowP_'+nombre+'.pdf nacorreasa@192.168.1.74:/var/www/nacorreasa/Graficas_Resultados/Estudio')

#################################################################################################
##------------------REEMPLAZANDO POR CERO LAS CORRELACIONES NO SIGNIFICATIVAS------------------##
#################################################################################################
if Significancia == True:
    p_975 = np.array(pval_975)
    p_348 = np.array(pval_348)
    p_350 = np.array(pval_350)

    alpha = 0.05
    w_348 = np.where(p_348>alpha)
    w_350 = np.where(p_350>alpha)
    w_975 = np.where(p_975>alpha)

    if len(w_975[0]) >0:
        for i in range(len(w_975[0])):
            a = w_975[0][i]
            b = w_975[1][i]
            corr_975.iloc[a, b] = 0.0
    else:
        pass

    if len(w_350[0]) >0:
        for i in range(len(w_350[0])):
            a = w_350[0][i]
            b = w_350[1][i]
            corr_350.iloc[a, b] = 0.0
    else:
        pass

    if len(w_348[0]) >0:
        for i in range(len(w_348[0])):
            a = w_348[0][i]
            b = w_348[1][i]
            corr_348.iloc[a, b] = 0.0
    else:
        pass

    ############################################################
    ## -------------------GRÁFICA DE VALOR P----------------- ##
    ############################################################

    fig = plt.figure(figsize=[10, 8])
    ax1 = fig.add_subplot(131)
    cax1 = ax1.imshow(np.array(pval_975), cmap='viridis')
    ticks = np.arange(0, len(pval_975.columns), 1)
    ax1.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax1.set_title('Valor P en TS', fontsize = 10, fontproperties = prop_1)
    ax1.set_yticks(ticks)
    ax1.set_xticklabels(pval_975.columns, rotation = 45, fontsize = 7)
    ax1.set_yticklabels(pval_975.columns, rotation = 45, fontsize = 7)
    for i in range(len(pval_975)):
        for j in range(len(pval_975)):
            text = ax1.text(j, i, round(float(np.array(pval_975)[i, j]), 3),
                           ha="center", va="center", color="w")

    ax2 = fig.add_subplot(132)
    cax2 = ax2.imshow(pval_350, cmap='viridis')
    ticks = np.arange(0, len(pval_350.columns), 1)
    ax2.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax2.set_title('Valor P en CI', fontsize = 10, fontproperties = prop_1)
    ax2.set_yticks(ticks)
    ax2.set_xticklabels(pval_350.columns, rotation = 45, fontsize = 7)
    ax2.set_yticklabels(pval_350.columns, rotation = 45, fontsize = 7)
    for i in range(len(pval_350)):
        for j in range(len(pval_350)):
            text = ax2.text(j, i, round(float(np.array(pval_350)[i, j]), 3),
                           ha="center", va="center", color="w")

    ax3 = fig.add_subplot(133)
    cax3 = ax3.imshow(pval_348, cmap='viridis')
    ticks = np.arange(0, len(pval_348.columns), 1)
    ax3.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax3.set_title('Valor P en CI', fontsize = 10, fontproperties = prop_1)
    ax3.set_yticks(ticks)
    ax3.set_xticklabels(pval_348.columns, rotation = 45, fontsize = 7)
    ax3.set_yticklabels(pval_348.columns, rotation = 45, fontsize = 7)
    for i in range(len(pval_348)):
        for j in range(len(pval_348)):
            text = ax3.text(j, i, round(float(np.array(pval_348)[i, j]), 3),
                           ha="center", va="center", color="w")


    fig.subplots_adjust(right=0.8)
    plt.subplots_adjust(wspace=0.3)
    plt.savefig('/home/nacorreasa/Escritorio/Figuras/Valor_P_Correlacion.png')
    plt.close('all')
    os.system('scp /home/nacorreasa/Escritorio/Figuras/Valor_P_Correlacion.png nacorreasa@192.168.1.74:/var/www/nacorreasa/Graficas_Resultados/Estudio')



















################################################################################
## --------------------------------DATOS NORMALZADOS------------------------- ##
################################################################################

df_corr_975_norm = (df_corr_975 - df_corr_975.mean()) / (df_corr_975.max() - df_corr_975.min())
df_corr_350_norm = (df_corr_350 - df_corr_350.mean()) / (df_corr_350.max() - df_corr_350.min())
df_corr_348_norm = (df_corr_348 - df_corr_348.mean()) / (df_corr_348.max() - df_corr_348.min())

fig = plt.figure(figsize=[10, 8])
plt.rc('axes', edgecolor='gray')
ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(df_corr_975_norm.index, df_corr_975_norm.Rad.values, color = 'r', label ='Rad')
ax1.plot(df_corr_975_norm.index, df_corr_975_norm.Pot.values, color = 'blue', label= 'Pot')
ax1.plot(df_corr_975_norm.index, df_corr_975_norm.Temp.values, color = 'orange', label= 'Temp')
ax1.plot(df_corr_975_norm.index, df_corr_975_norm.HR.values, color = 'g', label ='HR')
ax1.plot(df_corr_975_norm.index, df_corr_975_norm['Den WV'].values, color = 'black', label= 'Den WV')
ax1.plot(df_corr_975_norm.index, df_corr_975_norm.FR.values, color = 'c', label = 'FR')
ax1.plot(df_corr_975_norm.index, df_corr_975_norm['$\eta$'].values, color = 'pink', label = '$\eta$')
ax1.set_ylabel(u"Variables en TS", fontsize=14, fontproperties=prop_1)
ax1.set_title(u"Comparacion entre variables relacionadas en el tiempo", fontsize=17,  fontweight = "bold",  fontproperties = prop)
ax1.set_ylim(np.nanmin(df_corr_975_norm.values), np.nanmax(df_corr_975_norm.values) * 1.2)
ax1.set_xlim(df_corr_975_norm.index[0], df_corr_975_norm.index[-1])
ax1.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.7)
ax1.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter("%H:%M"))
ax1.xaxis.set_minor_locator(tck.MaxNLocator(nbins=5))
ax1.tick_params(axis='x', which='minor')
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
ax1.xaxis.set_major_locator(tck.MaxNLocator(nbins=5))
ax1.tick_params(axis='x', which='major', pad=15)
ax1.legend()

ax2 = fig.add_subplot(3, 1, 2)
ax2.plot(df_corr_350_norm.index, df_corr_350_norm.Rad.values, color = 'r', label ='Rad')
ax2.plot(df_corr_350_norm.index, df_corr_350_norm.Pot.values, color = 'blue', label= 'Pot')
ax2.plot(df_corr_350_norm.index, df_corr_350_norm.Temp.values, color = 'orange', label= 'Temp')
ax2.plot(df_corr_350_norm.index, df_corr_350_norm.HR.values, color = 'g', label ='HR')
ax2.plot(df_corr_350_norm.index, df_corr_350_norm['Den WV'].values, color = 'black', label= 'Den WV')
ax2.plot(df_corr_350_norm.index, df_corr_350_norm.FR.values, color = 'c', label = 'FR')
ax2.plot(df_corr_350_norm.index, df_corr_350_norm['$\eta$'].values, color = 'pink', label = '$\eta$')
ax2.set_ylabel(u"Variables en CI", fontsize=14, fontproperties=prop_1)
ax2.set_ylim(np.nanmin(df_corr_350_norm.values), np.nanmax(df_corr_350_norm.values) * 1.2)
ax2.set_xlim(df_corr_350_norm.index[0], df_corr_350_norm.index[-1])
ax2.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.7)
ax2.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter("%H:%M"))
ax2.xaxis.set_minor_locator(tck.MaxNLocator(nbins=5))
ax2.tick_params(axis='x', which='minor')
ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
ax2.xaxis.set_major_locator(tck.MaxNLocator(nbins=5))
ax2.tick_params(axis='x', which='major', pad=15)
ax2.legend()

ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(df_corr_348_norm.index, df_corr_348_norm.Rad.values, color = 'r', label ='Rad')
ax3.plot(df_corr_348_norm.index, df_corr_348_norm.Pot.values, color = 'blue', label= 'Pot')
ax3.plot(df_corr_348_norm.index, df_corr_348_norm.Temp.values, color = 'orange', label= 'Temp')
ax3.plot(df_corr_348_norm.index, df_corr_348_norm.HR.values, color = 'g', label ='HR')
ax3.plot(df_corr_348_norm.index, df_corr_348_norm['Den WV'].values, color = 'black', label= 'Den WV')
ax3.plot(df_corr_348_norm.index, df_corr_348_norm.FR.values, color = 'c', label = 'FR')
ax3.plot(df_corr_348_norm.index, df_corr_348_norm['$\eta$'].values, color = 'pink', label = '$\eta$')
ax3.set_ylabel(u"Variables en JV", fontsize=14, fontproperties=prop_1)
ax3.set_ylim(np.nanmin(df_corr_348_norm.values), np.nanmax(df_corr_348_norm.values) * 1.2)
ax3.set_xlim(df_corr_348_norm.index[0], df_corr_348_norm.index[-1])
ax3.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.7)
ax3.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter("%H:%M"))
ax3.xaxis.set_minor_locator(tck.MaxNLocator(nbins=5))
ax3.tick_params(axis='x', which='minor')
ax3.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
ax3.xaxis.set_major_locator(tck.MaxNLocator(nbins=5))
ax3.tick_params(axis='x', which='major', pad=15)
ax3.legend()

plt.savefig('/home/nacorreasa/Escritorio/Figuras/Corr_variables_plot.png')
#plt.show()
plt.close('all')
os.system('scp /home/nacorreasa/Escritorio/Figuras/Corr_variables_plot.png nacorreasa@192.168.1.74:/var/www/nacorreasa/Graficas_Resultados/Estudio')

################################################################################
## ----------------ANOMALÍAS DE LA RADIACIÓN Y LA TEMPERATURA---------------- ##
################################################################################

new_idx = np.arange(6, 18, 1)

df_All_P975 = df_All_P975.between_time('06:00', '17:00')              ##--> Seleccionar solo los datos de horas del dia
df_All_P350 = df_All_P350.between_time('06:00', '17:00')              ##--> Seleccionar solo los datos de horas del dia
df_All_P348 = df_All_P348.between_time('06:00', '17:00')              ##--> Seleccionar solo los datos de horas del dia


df_All_P975_Mean  = df_All_P975.groupby(by=[df_All_P975.index.hour]).mean()
df_All_P975_Mean  = df_All_P975_Mean.reindex(new_idx)
df_All_P975_Std  = df_All_P975.groupby(by=[df_All_P975.index.hour]).std()
df_All_P975_Std  = df_All_P975_Std.reindex(new_idx)
df_All_P975_Mean  = df_All_P975_Mean[['radiacion', 'T']]
df_All_P975_Std  = df_All_P975_Std[['radiacion', 'T']]
df_anomal_P975 = df_All_P975[['radiacion', 'T']]
#df_anomal_P975 = df_All_P975['T']
df_anomal_P975 = pd.DataFrame(df_anomal_P975)
df_anomal_P975 = df_anomal_P975[df_anomal_P975['T']>0]
df_anomal_P975['hour'] = df_anomal_P975.index.hour

Mean_T_P975 = []
Std_T_P975 = []
Mean_Rad_P975 = []
Std_Rad_P975 = []
for i in range(len(df_anomal_P975)):
    for j in range(len(df_All_P975_Mean)):
        if df_anomal_P975['hour'][i] == df_All_P975_Mean.index[j]:
            Mean_T_P975.append(df_All_P975_Mean['T'].values[j])
            Std_T_P975.append(df_All_P975_Std['T'].values[j])
            Mean_Rad_P975.append(df_All_P975_Mean['radiacion'].values[j])
            Std_Rad_P975.append(df_All_P975_Std['radiacion'].values[j])
        else:
            pass

df_anomal_P975['Mean_T'] = Mean_T_P975
df_anomal_P975['Std_T'] = Std_T_P975
df_anomal_P975['Anomal_T'] = (df_anomal_P975['T']-df_anomal_P975['Mean_T'])/df_anomal_P975['Std_T']

df_anomal_P975['Mean_Rad'] = Mean_Rad_P975
df_anomal_P975['Std_Rad'] = Std_Rad_P975
df_anomal_P975['Anomal_Rad'] = (df_anomal_P975['radiacion']-df_anomal_P975['Mean_Rad'])/df_anomal_P975['Std_Rad']


df_All_P350_Mean  = df_All_P350.groupby(by=[df_All_P350.index.hour]).mean()
df_All_P350_Mean  = df_All_P350_Mean.reindex(new_idx)
df_All_P350_Std  = df_All_P350.groupby(by=[df_All_P350.index.hour]).std()
df_All_P350_Std  = df_All_P350_Std.reindex(new_idx)
df_All_P350_Mean  = df_All_P350_Mean[['radiacion', 'T']]
df_All_P350_Std  = df_All_P350_Std[['radiacion', 'T']]
df_anomal_P350 = df_All_P350[['radiacion', 'T']]
#df_anomal_P350 = df_All_P350['T']
df_anomal_P350 = pd.DataFrame(df_anomal_P350)
df_anomal_P350 = df_anomal_P350[df_anomal_P350['T']>0]
df_anomal_P350['hour'] = df_anomal_P350.index.hour

Mean_T_P350 = []
Std_T_P350 = []
Mean_Rad_P350 = []
Std_Rad_P350 = []
for i in range(len(df_anomal_P350)):
    for j in range(len(df_All_P350_Mean)):
        if df_anomal_P350['hour'][i] == df_All_P350_Mean.index[j]:
            Mean_T_P350.append(df_All_P350_Mean['T'].values[j])
            Std_T_P350.append(df_All_P350_Std['T'].values[j])
            Mean_Rad_P350.append(df_All_P350_Mean['radiacion'].values[j])
            Std_Rad_P350.append(df_All_P350_Std['radiacion'].values[j])
        else:
            pass

df_anomal_P350['Mean_T'] = Mean_T_P350
df_anomal_P350['Std_T'] = Std_T_P350
df_anomal_P350['Anomal_T'] = (df_anomal_P350['T']-df_anomal_P350['Mean_T'])/df_anomal_P350['Std_T']

df_anomal_P350['Mean_Rad'] = Mean_Rad_P350
df_anomal_P350['Std_Rad'] = Std_Rad_P350
df_anomal_P350['Anomal_Rad'] = (df_anomal_P350['radiacion']-df_anomal_P350['Mean_Rad'])/df_anomal_P350['Std_Rad']


df_All_P348_Mean  = df_All_P348.groupby(by=[df_All_P348.index.hour]).mean()
df_All_P348_Mean  = df_All_P348_Mean.reindex(new_idx)
df_All_P348_Std  = df_All_P348.groupby(by=[df_All_P348.index.hour]).std()
df_All_P348_Std  = df_All_P348_Std.reindex(new_idx)
df_All_P348_Mean  = df_All_P348_Mean[['radiacion', 'T']]
df_All_P348_Std  = df_All_P348_Std[['radiacion', 'T']]
df_anomal_P348 = df_All_P348[['radiacion', 'T']]
#df_anomal_P348 = df_All_P348['T']
df_anomal_P348 = pd.DataFrame(df_anomal_P348)
df_anomal_P348 = df_anomal_P348[df_anomal_P348['T']>0]
df_anomal_P348['hour'] = df_anomal_P348.index.hour

Mean_T_P348 = []
Std_T_P348 = []
Mean_Rad_P348 = []
Std_Rad_P348 = []
for i in range(len(df_anomal_P348)):
    for j in range(len(df_All_P348_Mean)):
        if df_anomal_P348['hour'][i] == df_All_P348_Mean.index[j]:
            Mean_T_P348.append(df_All_P348_Mean['T'].values[j])
            Std_T_P348.append(df_All_P348_Std['T'].values[j])
            Mean_Rad_P348.append(df_All_P348_Mean['radiacion'].values[j])
            Std_Rad_P348.append(df_All_P348_Std['radiacion'].values[j])
        else:
            pass

df_anomal_P348['Mean_T'] = Mean_T_P348
df_anomal_P348['Std_T'] = Std_T_P348
df_anomal_P348['Anomal_T'] = (df_anomal_P348['T']-df_anomal_P348['Mean_T'])/df_anomal_P348['Std_T']

df_anomal_P348['Mean_Rad'] = Mean_Rad_P348
df_anomal_P348['Std_Rad'] = Std_Rad_P348
df_anomal_P348['Anomal_Rad'] = (df_anomal_P348['radiacion']-df_anomal_P348['Mean_Rad'])/df_anomal_P348['Std_Rad']

df_anomal_P975.index = pd.to_datetime(df_anomal_P975.index, format="%Y-%m-%d %H:%M:%S", errors='coerce')
df_anomal_P350.index = pd.to_datetime(df_anomal_P350.index, format="%Y-%m-%d %H:%M:%S", errors='coerce')
df_anomal_P348.index = pd.to_datetime(df_anomal_P348.index, format="%Y-%m-%d %H:%M:%S", errors='coerce')

## ----------------CD ANOMALÍAS DE LA RADIACIÓN Y LA TEMPERATURA---------------- ##

new_idx = np.arange(6, 18, 1)

df_anomal_P975_CD  = df_anomal_P975.groupby(by=[df_anomal_P975.index.hour]).mean()
df_anomal_P975_CD  = df_anomal_P975_CD.reindex(new_idx)

df_anomal_P350_CD  = df_anomal_P350.groupby(by=[df_anomal_P350.index.hour]).mean()
df_anomal_P350_CD  = df_anomal_P350_CD.reindex(new_idx)

df_anomal_P348_CD  = df_anomal_P348.groupby(by=[df_anomal_P348.index.hour]).mean()
df_anomal_P348_CD  = df_anomal_P348_CD.reindex(new_idx)

## ----------------CD DE LA RADIACIÓN Y LA TEMPERATURA---------------- ##

df_All_P975_CD  = df_All_P975.groupby(by=[df_All_P975.index.hour]).mean()
df_All_P975_CD  = df_All_P975_CD.reindex(new_idx)

df_All_P350_CD  = df_All_P350.groupby(by=[df_All_P350.index.hour]).mean()
df_All_P350_CD  = df_All_P350_CD.reindex(new_idx)

df_All_P348_CD  = df_All_P348.groupby(by=[df_All_P348.index.hour]).mean()
df_All_P348_CD  = df_All_P348_CD.reindex(new_idx)


## ----------------GRÁFICA DE LAS ANOMALÍAS DE LA RADIACIÓN Y LA TEMPERATURA---------------- ##

def two_scales(ax1, time, data1, data2, c1, c2, subplot_title):
    ax2 = ax1.twinx()
    ax1.plot(time, data1, color=c1)
    ax1.set_xlabel('Tiempo', fontproperties = prop_1)
    ax1.set_ylabel(r"Radiacion $[W/m^{2}]$", fontproperties = prop_1)
    ax2.plot(time, data2, color=c2)
    ax2.set_ylabel('Temperatura [°C]', fontproperties = prop_1)
    ax1.set_title(subplot_title, fontproperties = prop_2)
    ax1.set_xticklabels(time, rotation = 45, fontsize = 7)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    return ax1, ax2

t975 = df_anomal_P975.index
Rad975 = df_anomal_P975['Anomal_Rad']
Temp975 = df_anomal_P975['Anomal_T']

t350 = df_anomal_P350.index
Rad350 = df_anomal_P350['Anomal_Rad']
Temp350 = df_anomal_P350['Anomal_T']

t348 = df_anomal_P348.index
Rad348 = df_anomal_P348['Anomal_Rad']
Temp348 = df_anomal_P348['Anomal_T']

# Create axes
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,4))
ax1, ax1a = two_scales(ax1, t975, Rad975, Temp975, 'gold', 'limegreen', u'Anomalías TS')
ax2, ax2a = two_scales(ax2, t350, Rad350, Temp350, 'gold', 'limegreen', u'Anomalías CI')
ax3, ax3a = two_scales(ax3, t348, Rad348, Temp348, 'gold', 'limegreen', u'Anomalías JV')

# Change color of each axis
def color_y_axis(ax, color):
    """Color your axes."""
    for t in ax.get_yticklabels():
        t.set_color(color)

color_y_axis(ax1, 'gold')
color_y_axis(ax1a, 'limegreen')
color_y_axis(ax2, 'gold')
color_y_axis(ax2a, 'limegreen')
color_y_axis(ax3, 'gold')
color_y_axis(ax3a, 'limegreen')

plt.tight_layout()
plt.savefig('/home/nacorreasa/Escritorio/Figuras/Serie_Anomalias.png')
plt.show()

## ----------------GRÁFICA CD DE LAS ANOMALÍAS DE LA RADIACIÓN Y LA TEMPERATURA---------------- ##

def two_scales_CD(ax1, time, data1, data2, c1, c2, subplot_title):
    x_pos = np.arange(len(time))

    ax2 = ax1.twinx()
    ax1.plot(x_pos, data1, color=c1)
    #ax1.bar(x_pos, data1, color=c1, align='center', alpha=0.5)
    ax1.set_xlabel(u'Horas del dia', fontproperties = prop_1)
    ax1.set_ylabel(r"Radiacion $[W/m^{2}]$", fontproperties = prop_1, colo)
    ax2.plot(x_pos, data2, color=c2)
    #ax2.bar(x_pos, data2, color=c2, align='center', alpha=0.5)
    ax2.set_ylabel('Temperatura [°C]', fontproperties = prop_1)
    ax1.set_title(subplot_title, fontproperties = prop_2)
    ax1.set_xticks(np.arange(0, 12, 1))
    ax1.set_xticklabels(time.values, rotation = 20)
    #ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    return ax1, ax2

t975_CD = df_anomal_P975_CD.index
Rad975_CD = df_anomal_P975_CD['Anomal_Rad']
Temp975_CD = df_anomal_P975_CD['Anomal_T']

t350_CD = df_anomal_P350_CD.index
Rad350_CD = df_anomal_P350_CD['Anomal_Rad']
Temp350_CD = df_anomal_P350_CD['Anomal_T']

t348_CD = df_anomal_P348_CD.index
Rad348_CD = df_anomal_P348_CD['Anomal_Rad']
Temp348_CD = df_anomal_P348_CD['Anomal_T']

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,4))
ax1, ax1a = two_scales_CD(ax1, t975_CD, Rad975_CD , Temp975_CD , 'gold', 'limegreen', u'CD Anomalías TS')
ax2, ax2a = two_scales_CD(ax2, t350_CD, Rad350_CD , Temp350_CD , 'gold', 'limegreen', u'CD Anomalías CI')
ax3, ax3a = two_scales_CD(ax3, t348_CD, Rad348_CD , Temp348_CD , 'gold', 'limegreen', u'CD Anomalías JV')

def color_y_axis(ax, color):
    """Color your axes."""
    for t in ax.get_yticklabels():
        t.set_color(color)

color_y_axis(ax1, 'gold')
color_y_axis(ax1a, 'limegreen')
color_y_axis(ax2, 'gold')
color_y_axis(ax2a, 'limegreen')
color_y_axis(ax3, 'gold')
color_y_axis(ax3a, 'limegreen')

plt.tight_layout()
plt.savefig('/home/nacorreasa/Escritorio/Figuras/CD_Anomalias.png')
plt.show()

## ----------------GRÁFICA CD  DE LA RADIACIÓN Y LA TEMPERATURA---------------- ##

def two_scales_CD(ax1, time, data1, data2, c1, c2, subplot_title):
    x_pos = np.arange(len(time))

    ax2 = ax1.twinx()
    ax1.bar(x_pos, data1, color=c1, align='center', alpha=0.5)
    ax1.set_xlabel(u'Horas del dia', fontproperties = prop_1)
    ax1.set_ylabel(r"Radiacion $[W/m^{2}]$", fontproperties = prop_1, color = c1)
    ax2.bar(x_pos+0.75, data2, color=c2, align='center', alpha=0.5)
    ax2.set_ylabel('Temperatura [°C]', fontproperties = prop_1, color = c2)
    ax1.set_title(subplot_title, fontproperties = prop_2)
    ax1.set_xticks(np.arange(0, 12, 1))
    ax1.set_xticklabels(time.values, rotation = 20)
    #ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    return ax1, ax2

t975_CD = df_All_P975_CD.index
Rad975_CD = df_All_P975_CD['radiacion']
Temp975_CD = df_All_P975_CD['T']

t350_CD = df_All_P350_CD.index
Rad350_CD = df_All_P350_CD['radiacion']
Temp350_CD = df_All_P350_CD['T']

t348_CD = df_All_P350_CD.index
Rad348_CD = df_All_P350_CD['radiacion']
Temp348_CD = df_All_P350_CD['T']

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,4))
ax1, ax1a = two_scales_CD(ax1, t975_CD, Rad975_CD , Temp975_CD , 'gold', 'limegreen', u'CD Rad_Temp TS')
ax2, ax2a = two_scales_CD(ax2, t350_CD, Rad350_CD , Temp350_CD , 'gold', 'limegreen', u'CD Rad_Temp  CI')
ax3, ax3a = two_scales_CD(ax3, t348_CD, Rad348_CD , Temp348_CD , 'gold', 'limegreen', u'CD Rad_Temp  JV')

def color_y_axis(ax, color):
    """Color your axes."""
    for t in ax.get_yticklabels():
        t.set_color(color)

color_y_axis(ax1, 'gold')
color_y_axis(ax1a, 'limegreen')
color_y_axis(ax2, 'gold')
color_y_axis(ax2a, 'limegreen')
color_y_axis(ax3, 'gold')
color_y_axis(ax3a, 'limegreen')

plt.tight_layout()
plt.savefig('/home/nacorreasa/Escritorio/Figuras/CD_Rad_Temp.png')
plt.show()
