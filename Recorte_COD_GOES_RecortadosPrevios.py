#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import pearsonr
import matplotlib.font_manager as fm
import netCDF4 as nc
from netCDF4 import Dataset
import os
from pyproj import Proj
import ast

#------------------------------------------------------------------------------
# Motivación codigo -----------------------------------------------------------
"""
Codigo para el reconociemto de las variables de Porfundidad optica de las nubes de lat_GOES.
Los datos son leidos desde el disco duro por si volumen.
"""


path_directory = "/media/nacorreasa/NATHALIA/Productos/CODDF/"

path_save = '/home/nacorreasa/Maestria/Datos_Tesis/GOES/GOES_COD/'
resolucion = '2 km'
fec_ini = '2020-01-01'
fec_fin = '2020-01-12'
Modo = '6'

# =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
# =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
                                        # No cambiar nada en adelante
# =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
# =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*


##-----------------------------FUNCIONES PARA BUSCAR LOS ARCHIVOS---------------------------------##
import datetime as dt
fi  = dt.datetime.strptime(fec_ini, '%Y-%m-%d')
ff  = dt.datetime.strptime(fec_fin, '%Y-%m-%d')

def retorna_fecha_hora_info(ruta):
    #try:
    #    fecha = ruta[ruta.find('c2019')+1:ruta.find('c2019')+12]
    #except:
    fecha = ruta[ruta.find('c2020')+1:ruta.find('c2020')+12]
    fecha = dt.datetime.strptime(fecha, '%Y%j%H%M')
    fecha_utc = fecha.strftime('%Y/%m/%d/ %H:%M')
    fecha_local = fecha - dt.timedelta(hours=5)
    fecha_local = fecha_local.strftime('%Y/%m/%d %H:%M')
    fecha_path = fecha.strftime('%Y%m%d')
    fecha = fecha.strftime('%Y%m%d%H%M')
    return fecha_utc, fecha_path, fecha , fecha_local


def daterange(start_date, end_date):                                  ## PARA LA SELECCION DE LAS CARPETAS DE INTERES
    delta = timedelta(days=1)
    while start_date <= end_date:
        yield start_date
        start_date += delta

Folders = []
for single_date in daterange(fi, ff):
    Folders.append(single_date.date().strftime('%Y%m%d') )


def Listador(directorio, inicio, final):
    lf = []
    lista = os.listdir(directorio)
    for i in lista:
        if i.startswith(inicio) and i.endswith(final):
            lf.append(i)
    return lf

archivos = []
missing_folders = []
for folder in Folders:
    if os.path.exists(path_directory+folder) == True:
        dias =  Listador(path_directory+folder, 'CG_ABI-L2-CODDF-M'+Modo+'_G16', '.nc')
        dias.sort()
        for hora in dias:
            archivos.append(folder+'/'+hora)
    elif os.path.exists(path_directory+folder) == False:
        print ('hay q crear la carpeta: ' + folder)
        missing_folders.append(folder)
        #os.system('mkdir ' + path_directory+folder+'/')
        pass

archivos.sort()
##-----------------------------------FUNCION PARA CREAR LAS CARPETAS FALTNATES------------------------------------#
for folder in Folders:
    os.system('mkdir -p ' + path_save + folder )
##-----------------------------------FUNCIONES PARA RECORTAR------------------------------------#

def find_nearest(array, value):                              ## PARA ENCONTRAR LA CASILLA MAS CERCANA
    idx = (abs(array-value)).argmin()
    return array[idx], idx

def Posiciones(archivo, limites_zona):                        ## RELACIONAR ESAS CASILLAS CON LAS LATITUDES REQUERIDAS
    archivo = nc.Dataset(archivo)
    h = archivo.variables['goes_imager_projection'].perspective_point_height
    x = archivo.variables['x'][:]
    y = archivo.variables['y'][:]
    lon_0 = archivo.variables['goes_imager_projection'].longitude_of_projection_origin
    lat_0 = archivo.variables['goes_imager_projection'].latitude_of_projection_origin
    sat_sweep = archivo.variables['goes_imager_projection'].sweep_angle_axis
    archivo.close()
    x = x * h
    y = y * h
    p = Proj(proj='geos', h=h, lon_0=lon_0, swee=sat_sweep)
    xx, yy = np.meshgrid(x, y)
    print (xx, yy)

    lons, lats = p(xx[:, :], yy[:, :], inverse=True)
    lons = np.ma.array(lons)
    lons[lons == 1.00000000e+30] = np.ma.masked
    lats = np.ma.array(lats)
    lats[lats == 1.00000000e+30] = np.ma.masked

    mid = int(np.shape(lons)[0]/2)
    lon_O = find_nearest(lons[mid, :], limites_zona[0])[1]
    lon_E = find_nearest(lons[mid, :], limites_zona[1])[1]
    lat_N = find_nearest(lats[:, mid], limites_zona[2])[1]
    lat_S = find_nearest(lats[:, mid], limites_zona[3])[1]
    return lons, lats, lon_O, lon_E, lat_N, lat_S

def Obtiene_Lon_Lat_COD(ruta, dicc):                       ## PARA OBTENER LA INFORMACION ACOTADA A LA REGION DE INTERES
    limites_zona = dicc
    archivo = nc.Dataset(ruta)


    h = archivo.variables['goes_imager_projection'].perspective_point_height
    x = archivo.variables['x'][limites_zona[0]:limites_zona[1]]
    y = archivo.variables['y'][limites_zona[2]:limites_zona[3]]
    lon_0 = archivo.variables['goes_imager_projection'].longitude_of_projection_origin
    lat_0 = archivo.variables['goes_imager_projection'].latitude_of_projection_origin
    sat_sweep = archivo.variables['goes_imager_projection'].sweep_angle_axis
    #d = (archivo.variables['earth_sun_distance_anomaly_in_AU'][:])**2
    #esun = archivo.variables['esun'][:]

    COD = archivo.variables["COD"][limites_zona[2]:limites_zona[3], limites_zona[0]:limites_zona[1]]
    archivo.close()
    x = x * h
    y = y * h
    p = Proj(proj='geos', h=h, lon_0=lon_0, swee=sat_sweep)
    xx, yy = np.meshgrid(x,y)
    lons, lats = p(xx, yy, inverse=True)
    lons = np.ma.array(lons)
    lons[lons == 1.00000000e+30] = np.ma.masked
    lats = np.ma.array(lats)
    lats[lats == 1.00000000e+30] = np.ma.masked

    return lons, lats, COD

def LimitesZonas(resolucion, zonas, path_CH):                                   ## PARA OBTENER LA LISTA CON LOS LIMITES DE LA ZONA

    if resolucion == 0.5:
        for i in zonas.keys():
            lons, lats, lon_O, lon_E, lat_N, lat_S = Posiciones(path_CH, zonas[i])
            print (u"Posiciones dada una resolución espacial de 0.5 km")
            print ("Para la zona " + i + '\n'
                   u"La posición para la longitud occidental es : " + str(lon_O-1) + '\n' +
                   u"La posición para la longitud oriental es : " + str(lon_E) + '\n' +
                   u"La posición para la latitud norte es : " + str(lat_N-1) + '\n' +
                   u"La posición para la latitud sur es : " + str(lat_S) +'\n' +
                   u"inclúyalas así en el diccionario de posiciones:"  + '\n'
                   u"[" + str(lon_O-1) + ',' + str(lon_E) + ',' + str(lat_N-1) + ',' + str(lat_S) + "]")
        DiccionarioPos =   u"[" + str(lon_O-1) + ',' + str(lon_E) + ',' + str(lat_N-1) + ',' + str(lat_S) + "]"
        DiccionarioPos = ast.literal_eval(DiccionarioPos)
    elif resolucion == 1:
        for i in zonas.keys():
            lons, lats, lon_O, lon_E, lat_N, lat_S = Posiciones(path_CH, zonas[i])
            print (u"Posiciones dada una resolución espacial de 1 km")
            print ("Para la zona " + i + '\n'
                                         u"La posición para la longitud occidental es : " + str(lon_O - 1) + '\n' +
                   u"La posición para la longitud oriental es : " + str(lon_E) + '\n' +
                   u"La posición para la latitud norte es : " + str(lat_N - 1) + '\n' +
                   u"La posición para la latitud sur es : " + str(lat_S) + '\n' +
                   u"inclúyalas así en el diccionario de posiciones:" + '\n'
                                                                        u"[" + str(lon_O - 1) + ',' + str(
                lon_E) + ',' + str(lat_N - 1) + ',' + str(lat_S) + "]")
        DiccionarioPos =   u"[" + str(lon_O-1) + ',' + str(lon_E) + ',' + str(lat_N-1) + ',' + str(lat_S) + "]"
        DiccionarioPos = ast.literal_eval(DiccionarioPos)
    elif resolucion == 2:
        for i in zonas.keys():
            lons, lats, lon_O, lon_E, lat_N, lat_S = Posiciones(path_CH, zonas[i])
            print (u"Posiciones dada una resolución espacial de 2 km")
            print ("Para la zona " + i + '\n'
                   u"La posición para la longitud occidental es : " + str(lon_O-1) + '\n' +
                   u"La posición para la longitud oriental es : " + str(lon_E) + '\n' +
                   u"La posición para la latitud norte es : " + str(lat_N-1) + '\n' +
                   u"La posición para la latitud sur es : " + str(lat_S) +'\n' +
                   u"inclúyalas así en el diccionario de posiciones:"  + '\n'
                   u"[" + str(lon_O-1) + ',' + str(lon_E) + ',' + str(lat_N-1) + ',' + str(lat_S) + "]")
        DiccionarioPos =   u"[" + str(lon_O-1) + ',' + str(lon_E) + ',' + str(lat_N-1) + ',' + str(lat_S) + "]"
        DiccionarioPos = ast.literal_eval(DiccionarioPos)
    return DiccionarioPos


##-----------DELIMITAR LA ZONA  EN COORDENADAS CARTESIANAS Y CREAR  VARIABLES-----------#

## Coordenadas cartecianas de la zona de interés
latu = 6.5900
latd = 5.9300
lono = -75.85
lone = -75.07

zonas = {'Valle_de_Aburra': [-75.85, -75.07, 6.5900,  5.9300]}

## Ejecutar las funciones para tener la banda acotada a la región de interés y sus latitudes y longitudes

limzon = LimitesZonas(2, zonas, path_directory+archivos[0])
if resolucion == '1 km':
    limzon = [5328, 5416, 4698, 4771]
elif resolucion == '0.5 km':
    limzon = [10658, 10832, 9397, 9542]
elif resolucion == '2 km':
    limzon = [770, 815, 717, 754]

lons, lats, rad = Obtiene_Lon_Lat_COD(path_directory+ archivos[0], limzon)


## Iniciar variables
ntime    = 1
nlat     = np.shape(lats)[0]
nlon     = np.shape(lons)[1]
nrad     = np.shape(rad)

def Obtener_fecha_hora_de_Archvos (path_directory, archivos):
    local_hour = []
    local_date_str = []
    for i in range(len(archivos)):
        fecha_utc, fecha_path, fecha, fecha_local = retorna_fecha_hora_info(path_directory + archivos[i])
        local_hour.append(dt.datetime.strptime(fecha_local, '%Y/%m/%d %H:%M'))
        local_date_str.append(fecha_local)
    return local_date_str, local_hour

def retorna_scalefactor_y_addoffset_latlon(dataset,bits):         ## Nor maliza los valores de las variables
    import numpy as np
    max_o = np.max(dataset)
    min_o = np.min(dataset)
    scale_factor = (max_o - min_o) / (2**bits)
    add_offset = min_o
    return scale_factor, add_offset

fechas, time = Obtener_fecha_hora_de_Archvos (path_directory, archivos)

cdftime = 'hours since 2018-01-01 00:00:0.0'
date    = nc.date2num(time, units = cdftime)


# =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
# =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
                                        # Crear cada netCDF
# =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
# =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

for i  in range(len(archivos)):
    ## Varibales temporales
    lons_temp, lats_temp, COD_temp = Obtiene_Lon_Lat_COD(path_directory+archivos[i], limzon)

    COD = np.zeros((1, np.shape(COD_temp)[0], np.shape(COD_temp)[1]))
    COD[:, :, :] = COD_temp[:, :]

    ## Crear el nuevo archivo nc
    nw = Dataset(path_save + archivos[i], 'w', format='NETCDF4')

    ## Definir dimensiones locativas
    ncdim_lat = nw.createDimension('nlat', nlat)
    ncdim_lon = nw.createDimension('nlon', nlon)
    ncdim_time = nw.createDimension('ntime', ntime)

    # Crear variables locativas
    ncvar_lat = nw.createVariable('lat', 'f8', ('nlat', 'nlon'), zlib=True, complevel=9)
    ncvar_lon = nw.createVariable('lon', 'f8', ('nlat', 'nlon'), zlib=True, complevel=9)
    ncvar_time = nw.createVariable('time', 'f8', ('ntime',))

    ncvar_COD = nw.createVariable('COD', 'f8', ('ntime', 'nlat', 'nlon'), zlib=True, complevel=9)

    print ('netCDF variables created')

    ## Agregar unidades a las variables
    ncvar_lat.units = 'Degrees north'
    ncvar_lon.units = 'Degrees east'
    ncvar_time.units = 'Hours since 2018-01-01'

    ncvar_COD.units = 'dimensionless'

    try:
        ## Factores de escala
        ncvar_lon.scale_factor, ncvar_lon.add_offset = retorna_scalefactor_y_addoffset_latlon(lons, 11)
        ncvar_lat.scale_factor, ncvar_lat.add_offset = retorna_scalefactor_y_addoffset_latlon(lats, 11)
        ncvar_lat.scale_factor, ncvar_lat.add_offset = retorna_scalefactor_y_addoffset_latlon(COD, 11)
    except ValueError:
        pass

    ## Agregar nombres largos
    ncvar_lat.longname = 'Array of latitude values at the center of the grid box'
    ncvar_lon.longname = 'Array of longitude values at the center of the grid box'
    ncvar_time.longname = 'Hours since 2018-01-01'
    ncvar_COD.longname = 'COD'

    nw.title = archivos[i][12:]


    nw.spatial_resolution = resolucion

    nw.metadatos = "https://www.goes-r.gov/multimedia/dataAndImageryImagesGoes-16.html"

    # Agregar los datos al archivo
    ncvar_time[:] = date[i]
    ncvar_lat[:, :] = lats.data
    ncvar_lon[:, :] = lons.data

    print ('******************************************')
    print ('    writing variables in netCDF file ')
    print ('******************************************')
    ncvar_COD[:, :, :] = COD

    # Si no cierra el archivo es como dejar la BD abierta... se lo tira!
    nw.close()

print ('Hemos terminado')
