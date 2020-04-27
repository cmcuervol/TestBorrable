#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import random as rm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import matplotlib.colors as colors
import matplotlib

#-----------------------------------------------------------------------------
# Rutas para las fuentes -----------------------------------------------------

prop   = fm.FontProperties(fname='/home/nacorreasa/SIATA/Cod_Califi/AvenirLTStd-Heavy.otf' )
prop_1 = fm.FontProperties(fname='/home/nacorreasa/SIATA/Cod_Califi/AvenirLTStd-Book.otf')
prop_2 = fm.FontProperties(fname='/home/nacorreasa/SIATA/Cod_Califi/AvenirLTStd-Black.otf')

######################################################################################
## ----------------------LECTURA DE LOS ARRAY CONDICIONALES----------------------- ##
######################################################################################
Prob_Condicional_morning_JJA= np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/Array_Mark_Trans_MorningJJAmanianaCon.npy')
Prob_Condicional_noon_JJA   = np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/Array_Mark_Trans_NoonJJAnoonCon.npy')
Prob_Condicional_tarde_JJA  = np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/Array_Mark_Trans_TardeJJAtardeCon.npy')

Prob_Condicional_morning_SON= np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/Array_Mark_Trans_MorningSONmanianaCon.npy')
Prob_Condicional_noon_SON   = np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/Array_Mark_Trans_NoonSONnoonCon.npy')
Prob_Condicional_tarde_SON  = np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/Array_Mark_Trans_TardeSONtardeCon.npy')

Prob_Condicional_morning_DEF= np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/Array_Mark_Trans_MorningDEFmanianaCon.npy')
Prob_Condicional_noon_DEF   = np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/Array_Mark_Trans_NoonDEFnoonCon.npy')
Prob_Condicional_tarde_DEF  = np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/Array_Mark_Trans_TardeDEFtardeCon.npy')

Prob_Condicional_morning_MAM= np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/Array_Mark_Trans_MorningMAMmanianaCon.npy')
Prob_Condicional_noon_MAM   = np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/Array_Mark_Trans_NoonMAMnoonCon.npy')
Prob_Condicional_tarde_MAM  = np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/Array_Mark_Trans_TardeMAMtardeCon.npy')



######################################################################################
## ----------SIMULACION DE LAS REFLECTANCIAS CON LAS CADENAS DE MARKOV------------- ##
######################################################################################
import random as rm
def Markov_forecast(initial_state, steps, transitionMatrix):
    """
    Do Markov chain forecasting
    INPUTS
    initial_state    : The initial state
    steps            : number of predictions
    transitionMatrix : Array of states conditional probability

    OUTPUTS
    states : array of forecasting states
    """
    states = np.zeros((steps), dtype=int)
    n = transitionMatrix.shape[0]
    s = np.arange(1,n+1)

    states[0] = initial_state
    for i in range(steps-1):
        states[i+1] = np.random.choice(s,p=transitionMatrix[states[i]-1])

    return states

EI_List    = [1, 2, 3, 4, 5]
steps_time = 10  ##--> 2 y media Horas, el array final es de estos pasos

Simu_E1_JJA_Morning  = Markov_forecast(EI_List[0], steps_time, Prob_Condicional_morning_JJA/100)
Simu_E1_JJA_Noon     = Markov_forecast(EI_List[0], steps_time, Prob_Condicional_noon_JJA   /100)
Simu_E1_JJA_Tarde    = Markov_forecast(EI_List[0], steps_time, Prob_Condicional_tarde_JJA  /100)

Simu_E1_MAM_Morning  = Markov_forecast(EI_List[0], steps_time, Prob_Condicional_morning_MAM/100)
Simu_E1_MAM_Noon     = Markov_forecast(EI_List[0], steps_time, Prob_Condicional_noon_MAM   /100)
Simu_E1_MAM_Tarde    = Markov_forecast(EI_List[0], steps_time, Prob_Condicional_tarde_MAM  /100)

Simu_E1_SON_Morning  = Markov_forecast(EI_List[0], steps_time, Prob_Condicional_morning_SON/100)
Simu_E1_SON_Noon     = Markov_forecast(EI_List[0], steps_time, Prob_Condicional_noon_SON   /100)
Simu_E1_SON_Tarde    = Markov_forecast(EI_List[0], steps_time, Prob_Condicional_tarde_SON  /100)

Simu_E1_DEF_Morning  = Markov_forecast(EI_List[0], steps_time, Prob_Condicional_morning_DEF/100)
Simu_E1_DEF_Noon     = Markov_forecast(EI_List[0], steps_time, Prob_Condicional_noon_DEF   /100)
Simu_E1_DEF_Tarde    = Markov_forecast(EI_List[0], steps_time, Prob_Condicional_tarde_DEF  /100)

###------------------------GRAFICA DE LAS SIMULACIONES---------------------------##

Maniana_time = pd.date_range("06:00", "08:30", freq="15min").time[1:]
Noon_time    = pd.date_range("10:00", "12:30", freq="15min").time[1:]
Tarde_time   = pd.date_range("15:00", "17:30", freq="15min").time[1:]
Maniana_time = [now.strftime("%H:%M") for now in Maniana_time]
Noon_time    = [now.strftime("%H:%M") for now in Noon_time]
Tarde_time   = [now.strftime("%H:%M") for now in Tarde_time]


fig = plt.figure(figsize=[14, 6])
plt.rc('axes', edgecolor='gray')
ax1 = fig.add_subplot(1,3,1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.plot(np.arange(0,10,1),  Simu_E1_JJA_Morning, color='#b2df8a', label = 'JJA')
ax1.plot(np.arange(0,10,1),  Simu_E1_DEF_Morning, color='#33a02c', label = 'DEF')
ax1.plot(np.arange(0,10,1),  Simu_E1_MAM_Morning, color='#a6cee3', label = 'MAM')
ax1.plot(np.arange(0,10,1),  Simu_E1_SON_Morning, color='#1f78b4', label = 'SON')
ax1.scatter(np.arange(0,10,1),  Simu_E1_JJA_Morning, marker='.', color = '#b2df8a', s=30)
ax1.scatter(np.arange(0,10,1),  Simu_E1_DEF_Morning, marker='.', color = '#33a02c', s=30)
ax1.scatter(np.arange(0,10,1),  Simu_E1_MAM_Morning, marker='.', color = '#a6cee3', s=30)
ax1.scatter(np.arange(0,10,1),  Simu_E1_SON_Morning, marker='.', color = '#1f78b4', s=30)
ax1.set_title(u'Simulación de irradiancia 2h 15 min \n  adelante en la mañana', fontproperties=prop, fontsize = 14)
ax1.set_xticks(np.arange(0,len(Maniana_time), 1), minor=False)
ax1.set_xticklabels(np.array(Maniana_time), minor=False, rotation = 23)
ax1.set_ylabel(u'Estado', fontproperties=prop_1, fontsize = 13)
ax1.set_xlabel(u'Tiempo', fontproperties=prop_1, fontsize = 13)
ax1.set_ylim(0.5, 5.5)
ax1.set_yticks(np.arange(1,6,1), minor=False)
ax1.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.7)
#ax1.legend()

ax2 = fig.add_subplot(1,3,2)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.plot(np.arange(0,10,1),  Simu_E1_JJA_Noon, color='#b2df8a', label = 'JJA')
ax2.plot(np.arange(0,10,1),  Simu_E1_DEF_Noon, color='#33a02c', label = 'DEF')
ax2.plot(np.arange(0,10,1),  Simu_E1_MAM_Noon, color='#a6cee3', label = 'MAM')
ax2.plot(np.arange(0,10,1),  Simu_E1_SON_Noon, color='#1f78b4', label = 'SON')
ax2.scatter(np.arange(0,10,1),  Simu_E1_JJA_Noon, marker='.', color = '#b2df8a', s=30)
ax2.scatter(np.arange(0,10,1),  Simu_E1_DEF_Noon, marker='.', color = '#33a02c', s=30)
ax2.scatter(np.arange(0,10,1),  Simu_E1_MAM_Noon, marker='.', color = '#a6cee3', s=30)
ax2.scatter(np.arange(0,10,1),  Simu_E1_SON_Noon, marker='.', color = '#1f78b4', s=30)
ax2.set_title(u'Simulación de irradiancia 2h 15 min \n  adelante al medio dia', fontproperties=prop, fontsize = 14)
ax2.set_xticks(np.arange(0,len(Noon_time), 1), minor=False)
ax2.set_xticklabels(np.array(Noon_time), minor=False, rotation = 23)
ax2.set_ylabel(u'Estado', fontproperties=prop_1, fontsize = 13)
ax2.set_xlabel(u'Tiempo', fontproperties=prop_1, fontsize = 13)
ax2.set_ylim(0.5, 5.5)
ax2.set_yticks(np.arange(1,6,1), minor=False)
ax2.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.7)
#ax2.legend()

ax3 = fig.add_subplot(1,3,3)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.plot(np.arange(0,10,1),  Simu_E1_JJA_Tarde, color='#b2df8a', label = 'JJA')
ax3.plot(np.arange(0,10,1),  Simu_E1_DEF_Tarde, color='#33a02c', label = 'DEF')
ax3.plot(np.arange(0,10,1),  Simu_E1_MAM_Tarde, color='#a6cee3', label = 'MAM')
ax3.plot(np.arange(0,10,1),  Simu_E1_SON_Tarde, color='#1f78b4', label = 'SON')
ax3.scatter(np.arange(0,10,1),  Simu_E1_JJA_Tarde, marker='.', color = '#b2df8a', s=30)
ax3.scatter(np.arange(0,10,1),  Simu_E1_DEF_Tarde, marker='.', color = '#33a02c', s=30)
ax3.scatter(np.arange(0,10,1),  Simu_E1_MAM_Tarde, marker='.', color = '#a6cee3', s=30)
ax3.scatter(np.arange(0,10,1),  Simu_E1_SON_Tarde, marker='.', color = '#1f78b4', s=30)
ax3.set_title(u'Simulación de irradiancia 2h 15 min \n  adelante en la tarde', fontproperties=prop, fontsize = 14)
ax3.set_xticks(np.arange(0,len(Tarde_time), 1), minor=False)
ax3.set_xticklabels(np.array(Tarde_time), minor=False, rotation = 23)
ax3.set_ylabel(u'Estado', fontproperties=prop_1, fontsize = 13)
ax3.set_xlabel(u'Tiempo', fontproperties=prop_1, fontsize = 13)
ax3.set_ylim(0.5, 5.5)
ax3.set_yticks(np.arange(1,6,1), minor=False)
ax3.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.7)
ax3.legend()

plt.savefig('/home/nacorreasa/Escritorio/Figuras/Simulation_States_Radiacion.pdf', format = 'pdf')
plt.close('all')
os.system('scp /home/nacorreasa/Escritorio/Figuras/Simulation_States_Radiacion.pdf nacorreasa@192.168.1.74:/var/www/nacorreasa/Graficas_Resultados/Estudio')

################################################################################
##--------------------LECTURA DE LOS DATOS ORIGINALES ------------------------##
################################################################################
df = pd.read_csv('/home/nacorreasa/Maestria/Datos_Tesis/Experimentos_Panel/Panel975.txt',  sep=',', index_col =0)
df = df[df['radiacion'] > 0]
df = df[(df['NI'] >= 0) & (df['strength'] >= 0)& (df['strength'] <= 80)]
#df = df[df['calidad']<100]
df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M", errors='coerce')
df       = df.between_time('06:00', '17:00')

Histo_pot, Bins_pot = np.histogram(df['strength'][np.isfinite(df['strength'])] ,
                                   bins= df['strength'].quantile([.1, .5, .7, .75, .8, .85,.9, 0.95, 0.98, 0.99, 1]).values ,
                                   density=True)

###############################################################################################
##--PRONOSTICO DE LA DISTRIBUCIÓN.POTENCIA  POR LA RELACIÓN DE LOS ESTADOS DE LOS BORDES --- ##
###############################################################################################

def FDP_Simulations(Simulacion, df, field, trimestre, etapa, ajuste):
    """
    It obtains the bivariate histogram of probability of a variable,
    based on the states siutation. It takes the bins down and above
    of solar radiation. It has 20 bins and 10 steps of time

    INPUTS
    Simulacion : Array with the simulated states
    df         : DataFrame with the original values for computting the histogram
    field      : Str of field name of interes of the DataFrame with the data
    trimestre  : 0 DEF, 1 MAM, 2 JJA, 3 SON
    etapa      : 0 Morning, 1 Noon, 2 Afernoon
    ajuste     : 0 lineal, 1 cuadratico, 2 cubico

    OUTPUTS
    Histos     : 2D array with FDP for each simulation time
    """
    Coeficientes  = np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/CoeficientesAjuste.npy')
    Coef = Coeficientes[trimestre, etapa, ajuste]
    Ref_Bins_dwn = [  1,  271,  542,  813, 1084]
    Ref_Bins_abv = [  271,  542,  813, 1084, 1355]

    Histos=[]
    for j in range(len(Simulacion)):
        for i in range(len(Ref_Bins_dwn)):
            if  i+1 == Simulacion[j]:
                Pot_dwn =   (Ref_Bins_dwn[i]**Coef[2])*(np.e**Coef[3])
                Pot_abv =   (Ref_Bins_abv[i]**Coef[2])*(np.e**Coef[3])

                print(i+1)
                if Pot_abv > 0 and Pot_abv <1:
                    Pot_abv = 1
                if Pot_dwn <0 :
                    Pot_dwn = 0
                if Pot_abv > 80 :
                    Pot_abv = 80
                else:
                    pass
                df_temp = df[(df[field].values >= Pot_dwn) & (df[field].values <= Pot_abv)]
                Histo_temp, Bins_temp = np.histogram(df_temp[field][np.isfinite(df_temp[field])] , bins=Bins_pot,  density=True)
                Histos.append(Histo_temp)
            else:
                pass
    Histos = np.array(Histos).T
    return Histos



# def FDP_Simulations(Simulacion, df, field):
#     """
#     It obtains the bivariate histogram of probability of a variable,
#     based on the states siutation. It takes the bins down and above
#     of solar radiation. It has 20 bins and 10 steps of time
#
#     INPUTS
#     Simulacion : Array with the simulated states
#     df         : DataFrame with the original values for computting the histogram
#     field      : Str of field name of interes of the DataFrame with the data
#
#     OUTPUTS
#     Histos     : 2D array with FDP for each simulation time
#     """
#     Ref_Bins_dwn = [  1,  271,  542,  813, 1084]
#     Ref_Bins_abv = [  271,  542,  813, 1084, 1355]
#     Histos=[]
#     for j in range(len(Simulacion)):
#         for i in range(len(Ref_Bins_dwn)):
#             if  i+1 == Simulacion[j]:
#                 """
#                 A continuación la ecuación de la relación, hay q cambiarla
#                 """
#                 Pot_dwn = 0.063*(Ref_Bins_dwn[i])-11.295
#                 Pot_abv = 0.063*(Ref_Bins_abv[i])-11.295
#                 print(i+1)
#                 if Pot_dwn <0:
#                     Pot_dwn = 0
#                 else:
#                     pass
#                 df_temp = df[(df[field] >= Pot_dwn) & (df[field] <= Pot_abv)]
#                 Histo_temp, Bins_temp = np.histogram(df_temp[field][np.isfinite(df_temp[field])] , bins=Bins_pot,  density=True)
#                 Histos.append(Histo_temp)
#             else:
#                 pass
#     Histos = np.array(Histos).T
#     return Histos

Histo_JJA_Morning = FDP_Simulations(Simu_E1_JJA_Morning, df, 'strength', 2, 0, 0)
Histo_JJA_Noon    = FDP_Simulations(Simu_E1_JJA_Noon,    df, 'strength', 2, 1, 0)
Histo_JJA_Tarde   = FDP_Simulations(Simu_E1_JJA_Tarde,   df, 'strength', 2, 2, 0)

Histo_DEF_Morning = FDP_Simulations(Simu_E1_DEF_Morning, df, 'strength', 0, 0, 0)
Histo_DEF_Noon    = FDP_Simulations(Simu_E1_DEF_Noon,    df, 'strength', 0, 1, 0)
Histo_DEF_Tarde   = FDP_Simulations(Simu_E1_DEF_Tarde,   df, 'strength', 0, 2, 0)

Histo_SON_Morning = FDP_Simulations(Simu_E1_SON_Morning, df, 'strength', 3, 0, 0)
Histo_SON_Noon    = FDP_Simulations(Simu_E1_SON_Noon,    df, 'strength', 3, 1, 0)
Histo_SON_Tarde   = FDP_Simulations(Simu_E1_SON_Tarde,   df, 'strength', 3, 2, 0)

Histo_MAM_Morning = FDP_Simulations(Simu_E1_MAM_Morning, df, 'strength', 1, 0, 0)
Histo_MAM_Noon    = FDP_Simulations(Simu_E1_MAM_Noon,    df, 'strength', 1, 1, 0)
Histo_MAM_Tarde   = FDP_Simulations(Simu_E1_MAM_Tarde,   df, 'strength', 1, 2, 0)

################################################################################
## -----------------------------------GRAFICA-------------------------------- ##
################################################################################


class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

cmap = matplotlib.cm.hot_r
# cmap = matplotlib.cm.Spectral_r


data     = np.array([Histo_DEF_Morning, Histo_MAM_Morning, Histo_JJA_Morning, Histo_SON_Morning, Histo_DEF_Noon, Histo_MAM_Noon, Histo_JJA_Noon, Histo_SON_Noon,  Histo_DEF_Tarde, Histo_MAM_Tarde, Histo_JJA_Tarde, Histo_SON_Tarde])
titles   = [ u'DEF Mañana', u'MAM Mañana', u'JJA Mañana',u'SON Mañana', 'DEF Medio dia', 'MAM Medio dia', 'JJA Medio dia','SON Medio dia', 'DEF Tarde', 'MAM Tarde', 'JJA Tarde','SON Tarde']
x_arrays = [Maniana_time, Maniana_time, Maniana_time, Maniana_time, Noon_time ,Noon_time, Noon_time, Noon_time, Tarde_time, Tarde_time, Tarde_time, Tarde_time ]

plt.close('all')
fig = plt.figure(figsize=(13,10))
for i in range(0, 12):
    ax = fig.add_subplot(3, 4, i+1)
    # mapa = ax.imshow(data[i][::-1], interpolation = 'hamming', cmap=cmap,
    mapa = ax.imshow(data[i][::-1], interpolation = None, cmap=cmap,
                     # clim=(data.min()), vmin=Histo_pot.min(), vmax=Histo_pot.max())
                     clim=(data.min()), vmin=data.min(), vmax=data.max())
    ax.set_yticks(range(0,data[i].shape[0]), minor=False)
    ax.set_yticklabels(Bins_pot[1:][::-1], minor=False)
    ax.set_xticks(range(0,data[i].shape[1]), minor=False)
    ax.set_xticklabels(np.array(x_arrays[i]), minor=False, rotation = 31)
    ax.set_ylabel(u'Potencia $[W]$', fontproperties = prop_1,  fontsize=12)
    # ax.set_xlabel('Tiempo', fontproperties = prop_1, fontsize=12)
    ax.set_title(titles[i], fontproperties = prop, fontsize =15)


cbar_ax = fig.add_axes([0.11, 0.06, 0.78, 0.008])
cbar = fig.colorbar(mapa, cax=cbar_ax, orientation='horizontal', format="%.2f")
cbar.set_label(u"Probabilidad", fontsize=13, fontproperties=prop)

plt.subplots_adjust(left=0.125, bottom=0.085, right=0.9, top=0.95, wspace=0.4, hspace=0.003)

plt.savefig('/home/nacorreasa/Escritorio/Figuras/MarkovSimulation_Radiacion.pdf', format='pdf', transparent=True)
plt.close('all')
os.system('scp /home/nacorreasa/Escritorio/Figuras/MarkovSimulation_Radiacion.pdf nacorreasa@192.168.1.74:/var/www/nacorreasa/Graficas_Resultados/Estudio')

# OTRO CODE
#-----------------------------------------------------------------------------
# Rutas para las fuentes -----------------------------------------------------

prop = fm.FontProperties(fname='/home/nacorreasa/SIATA/Cod_Califi/AvenirLTStd-Heavy.otf' )
prop_1 = fm.FontProperties(fname='/home/nacorreasa/SIATA/Cod_Califi/AvenirLTStd-Book.otf')
prop_2 = fm.FontProperties(fname='/home/nacorreasa/SIATA/Cod_Califi/AvenirLTStd-Black.otf')

##########################################################################################
## ----------------LECTURA DE LOS DATOS DE LOS EXPERIMENTOS Y RADIACION---------------- ##
##########################################################################################

df_P975 = pd.read_csv('/home/nacorreasa/Maestria/Datos_Tesis/Experimentos_Panel/Panel975.txt',  sep=',', index_col =0)
df_P350 = pd.read_csv('/home/nacorreasa/Maestria/Datos_Tesis/Experimentos_Panel/Panel350.txt',  sep=',', index_col =0)
df_P348 = pd.read_csv('/home/nacorreasa/Maestria/Datos_Tesis/Experimentos_Panel/Panel348.txt',  sep=',', index_col =0)

df_P975['Fecha_hora'] = df_P975.index
df_P350['Fecha_hora'] = df_P350.index
df_P348['Fecha_hora'] = df_P348.index

df_P975.index = pd.to_datetime(df_P975.index, format="%Y-%m-%d %H:%M:%S", errors='coerce')
df_P350.index = pd.to_datetime(df_P350.index, format="%Y-%m-%d %H:%M:%S", errors='coerce')
df_P348.index = pd.to_datetime(df_P348.index, format="%Y-%m-%d %H:%M:%S", errors='coerce')

## ----------------ACOTANDO LOS DATOS A VALORES VÁLIDOS---------------- ##
df_P975 = df_P975[(df_P975['NI'] > 0) & (df_P975['strength'] > 0)]
df_P350 = df_P350[(df_P350['NI'] > 0) & (df_P350['strength'] > 0)]
df_P348 = df_P348[(df_P348['NI'] > 0) & (df_P348['strength'] > 0)]


df_P975 = df_P975[df_P975['radiacion'] > 0]
df_P350 = df_P350[df_P350['radiacion'] > 0]
df_P348 = df_P348[df_P348['radiacion'] > 0]

df_P975 = df_P975[df_P975['strength'] <=80]
df_P350 = df_P350[df_P350['strength'] <=100]
df_P348 = df_P348[df_P348['strength'] <=100]



df_P975_h = df_P975.groupby(pd.Grouper(freq="H")).mean()
df_P350_h = df_P350.groupby(pd.Grouper(freq="H")).mean()
df_P348_h = df_P348.groupby(pd.Grouper(freq="H")).mean()

df_P975_h.drop(df_P975_h[(df_P975_h['strength']>8) & (df_P975_h['radiacion']<100)].index, axis=0, inplace=True)
df_P350_h.drop(df_P350_h[(df_P350_h['strength']>8) & (df_P350_h['radiacion']<100)].index, axis=0, inplace=True)
df_P348_h.drop(df_P348_h[(df_P348_h['strength']>8) & (df_P348_h['radiacion']<100)].index, axis=0, inplace=True)

df_P975_h.drop(df_P975_h[(df_P975_h['strength']>20) & (df_P975_h['radiacion']<350)].index, axis=0, inplace=True)
df_P350_h.drop(df_P350_h[(df_P350_h['strength']>20) & (df_P350_h['radiacion']<350)].index, axis=0, inplace=True)
df_P348_h.drop(df_P348_h[(df_P348_h['strength']>20) & (df_P348_h['radiacion']<350)].index, axis=0, inplace=True)


Punto = 'Centro-Oeste'
df   = df_P975_h
df   = df.between_time('06:00', '17:00')

y_lablels = np.unique(df.index.date)
x_lablels = np.unique(np.array(df.index.hour))

data = np.zeros((len(y_lablels), len(x_lablels ) ), dtype=float)
idx = [6,7,8,9,10,11,12,13,14,15,16,17]
df_idx = pd.DataFrame(index= idx)
for i in range(len(y_lablels)):
    df_t         = pd.DataFrame(df.strength[df.index.date == y_lablels[i]])
    df_t['Hora'] = np.array(df_t.index.hour)
    df_t.index   = df_t['Hora']
    df_fin       = pd.concat([df_idx, df_t], axis=1, sort=False)
    data[i,:]   = df_fin.strength.values
    print(i)

cmap = matplotlib.cm.Spectral_r
plt.close('all')
# fig = plt.figure(figsize=(8,10))
fig = plt.figure()
ax = fig.add_subplot(111)
norml = colors.Normalize(vmin=np.nanmin(df.strength.values) , vmax=np.nanmax(df.strength.values))
mapa = ax.imshow(data, interpolation = 'none', cmap=cmap,norm = norml)
cbar = fig.colorbar(mapa, ax=ax, orientation='vertical', format="%.2f")
cbar.set_label(u"Potencia [W]", fontsize=12, fontproperties=prop)
ticks = np.arange(0,data.shape[0], 10)
labels = [y_lablels[i] for i in ticks]
ax.set_aspect(aspect=0.05)
ax.set_yticks(ticks, minor=False)
ax.set_yticklabels(labels , minor=False)
ax.set_xticks(range(0,data.shape[1]), minor=False)
ax.set_xticklabels(x_lablels, minor=False)
# ax.set_ylabel(u'Potencia $[W]$', fontproperties = prop_1,  fontsize=12)
ax.set_xlabel('Hora', fontproperties = prop_1, fontsize=12)
ax.set_title('Registros horarios de potencia en '+ Punto, fontproperties = prop, fontsize =15)

# cbar_ax = fig.add_axes([0.11, 0.05, 0.78, 0.008])
# cbar = fig.colorbar(mapa, cax=cbar_ax, orientation='horizontal', format="%.2f")
# cbar.set_label(u"Potencia [W]", fontsize=12, fontproperties=prop)

# plt.subplots_adjust(left=0.125, bottom=0.085, right=0.9, top=0.95, wspace=0.4, hspace=0.003)

plt.savefig('/home/nacorreasa/Escritorio/Figuras/Data_Potencia.pdf', format='pdf', transparent=True)
plt.close('all')
os.system('scp /home/nacorreasa/Escritorio/Figuras/Data_Potencia.pdf nacorreasa@192.168.1.74:/var/www/nacorreasa/Graficas_Resultados/Estudio')


def Hitograma_estados_ini(df_15min, trimestre, etapa ):

    if trimestre == 'JJA':
        mes = [6, 7, 8]
    elif trimestre == 'SON':
        mes = [9, 10, 11]
    elif trimestre == 'DEF':
        mes = [12, 1, 2]
    elif trimestre == 'MAM':
        mes = [3, 4, 5]

    df_meses = df_15min[(df_15min.index.month  == mes[0])|(df_15min.index.month  == mes[1])|(df_15min.index.month  == mes[2])]

    if etapa == 'maniana':
        df_meses = df_meses[(df_meses.index.hour == 6)|(df_meses.index.hour == 7)|(df_meses.index.hour == 8)|(df_meses.index.hour == 9)]
    elif etapa == 'noon':
        df_meses = df_meses[(df_meses.index.hour == 10)|(df_meses.index.hour == 11)|(df_meses.index.hour == 12)|(df_meses.index.hour == 13)|(df_meses.index.hour == 14)]
    elif etapa == 'tarde':
        df_meses = df_meses[(df_meses.index.hour == 15)|(df_meses.index.hour == 16)|(df_meses.index.hour == 17)]


    Histo_mes, Bins_mes = np.histogram(df_meses['Estado_ini'][np.isfinite(df_meses['Estado_ini'])] , bins=[0, 1, 2, 3, 4, 5], density = False)
    return Histo_mes, Bins_mes

Histo_DEF_noon, Bins_DEF_noon       = Hitograma_estados_ini(df_15min, 'DEF', 'noon')
Histo_DEF_maniana, Bins_DEF_maniana = Hitograma_estados_ini(df_15min, 'DEF', 'maniana')
Histo_DEF_tarde, Bins_DEF_tarde     = Hitograma_estados_ini(df_15min, 'DEF', 'tarde')

Histo_SON_noon, Bins_SON_noon       = Hitograma_estados_ini(df_15min, 'SON', 'noon')
Histo_SON_maniana, Bins_SON_maniana = Hitograma_estados_ini(df_15min, 'SON', 'maniana')
Histo_SON_tarde, Bins_SON_tarde     = Hitograma_estados_ini(df_15min, 'SON', 'tarde')

Histo_MAM_noon, Bins_MAM_noon       = Hitograma_estados_ini(df_15min, 'MAM', 'noon')
Histo_MAM_maniana, Bins_MAM_maniana = Hitograma_estados_ini(df_15min, 'MAM', 'maniana')
Histo_MAM_tarde, Bins_MAM_tarde     = Hitograma_estados_ini(df_15min, 'MAM', 'tarde')

Histo_JJA_noon, Bins_JJA_noon       = Hitograma_estados_ini(df_15min, 'JJA', 'noon')
Histo_JJA_maniana, Bins_JJA_maniana = Hitograma_estados_ini(df_15min, 'JJA', 'maniana')
Histo_JJA_tarde, Bins_JJA_tarde     = Hitograma_estados_ini(df_15min, 'JJA', 'tarde')





#!/usr/bin/env python
# -- coding: utf-8 --
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import datetime
import os

#------------------------------------------------------------------------------
# Motivación codigo -----------------------------------------------------------
"""
Programa para obtener las cadenas de markov por etapa dentro del dia para la radiacion.
"""

#-----------------------------------------------------------------------------
# Rutas para las fuentes -----------------------------------------------------

prop   = fm.FontProperties(fname='/home/nacorreasa/SIATA/Cod_Califi/AvenirLTStd-Heavy.otf' )
prop_1 = fm.FontProperties(fname='/home/nacorreasa/SIATA/Cod_Califi/AvenirLTStd-Book.otf')
prop_2 = fm.FontProperties(fname='/home/nacorreasa/SIATA/Cod_Califi/AvenirLTStd-Black.otf')

##############################################################################
## ----------------LECTURA DE LOS DATOS DE LOS EXPERIMENTOS---------------- ##
##############################################################################


df_P975 = pd.read_table('/home/nacorreasa/Maestria/Datos_Tesis/Piranometro/60012018_2019.txt', parse_dates=[2])
df_P975 = df_P975.set_index(["fecha_hora"])
df_P975.index = df_P975.index.tz_localize('UTC').tz_convert('America/Bogota')
df_P975.index = df_P975.index.tz_localize(None)
df_P975 = df_P975[df_P975['radiacion'] >=0]
df_P975 = df_P975.between_time('06:00', '17:00')

df_15min = df_P975.groupby(pd.Grouper(freq="15min")).mean()
df_15min = df_15min[df_15min.radiacion>0]

####################################################################################
## ---------------DEFICINICIÓN DE ESTADOS CON LA FDP DE LOS ESTADOS-------------- ##
####################################################################################
Histograma, Bins = np.histogram(df_15min['radiacion'][np.isfinite(df_15min['radiacion'])] , bins=5, density = False)

fig = plt.figure(figsize=[10, 7])
plt.rc('axes', edgecolor='gray')
ax1 = fig.add_subplot(111)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.hist(df_15min['radiacion'][np.isfinite(df_15min['radiacion'])], bins=5, density = False, color = '#a6cee3')
ax1.set_xlabel(u'Irradiancia $[W/m^{2}]$', fontproperties = prop_1, fontsize=15)
ax1.set_ylabel(r"Frecuencia", fontproperties = prop_1, fontsize=15)
ax1.set_title(u'Histograma de frecuencias de irradiancia', loc = 'center', fontsize=13)
ax1.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.7)
ax1.set_xticks(Bins, minor=False)
ax1.set_xticklabels(Bins.astype(int), minor=False, rotation = 28)
plt.savefig('/home/nacorreasa/Escritorio/Figuras/HistogramaDefinicionEstados_Radiacion.png')
plt.close('all')
os.system('scp /home/nacorreasa/Escritorio/Figuras/HistogramaDefinicionEstados_Radiacion.png nacorreasa@192.168.1.74:/var/www/nacorreasa/Graficas_Resultados/Estudio')



####################################################################################
## -----------------------GENERACION DE LA LISTA DE ESTADOS---------------------- ##
####################################################################################

Estado_ini = [df_15min [(df_15min.radiacion.values >= round(Bins[i],2)) & (df_15min.radiacion.values <=round(Bins[i+1],2))] for i in range(5)]

df_15min['Estado_ini'] = np.zeros(len(df_15min))

for i in range(len(df_15min.index)):
    for e in range(1,6):
        if df_15min.index[i] in Estado_ini[e-1].index:
            print(e)
            df_15min['Estado_ini'].iloc[i] = e

####################################################################################
## ----------------HISTOGRAMA DE LOS ESTADOS INICIALES POR CADA ETAPA------------ ##
####################################################################################

def Hitograma_estados_ini(df_15min, trimestre, etapa ):

    if trimestre == 'JJA':
        mes = [6, 7, 8]
    elif trimestre == 'SON':
        mes = [9, 10, 11]
    elif trimestre == 'DEF':
        mes = [12, 1, 2]
    elif trimestre == 'MAM':
        mes = [3, 4, 5]

    df_meses = df_15min[(df_15min.index.month  == mes[0])|(df_15min.index.month  == mes[1])|(df_15min.index.month  == mes[2])]

    if etapa == 'maniana':
        df_meses = df_meses[(df_meses.index.hour == 6)|(df_meses.index.hour == 7)|(df_meses.index.hour == 8)|(df_meses.index.hour == 9)]
    elif etapa == 'noon':
        df_meses = df_meses[(df_meses.index.hour == 10)|(df_meses.index.hour == 11)|(df_meses.index.hour == 12)|(df_meses.index.hour == 13)|(df_meses.index.hour == 14)]
    elif etapa == 'tarde':
        df_meses = df_meses[(df_meses.index.hour == 15)|(df_meses.index.hour == 16)|(df_meses.index.hour == 17)]


    Histo_mes, Bins_mes = np.histogram(df_meses['Estado_ini'][np.isfinite(df_meses['Estado_ini'])] , bins=[0, 1, 2, 3, 4, 5], density = False)
    return Histo_mes, Bins_mes

Histo_DEF_noon, Bins_DEF_noon       = Hitograma_estados_ini(df_15min, 'DEF', 'noon')
Histo_DEF_maniana, Bins_DEF_maniana = Hitograma_estados_ini(df_15min, 'DEF', 'maniana')
Histo_DEF_tarde, Bins_DEF_tarde     = Hitograma_estados_ini(df_15min, 'DEF', 'tarde')

Histo_SON_noon, Bins_SON_noon       = Hitograma_estados_ini(df_15min, 'SON', 'noon')
Histo_SON_maniana, Bins_SON_maniana = Hitograma_estados_ini(df_15min, 'SON', 'maniana')
Histo_SON_tarde, Bins_SON_tarde     = Hitograma_estados_ini(df_15min, 'SON', 'tarde')

Histo_MAM_noon, Bins_MAM_noon       = Hitograma_estados_ini(df_15min, 'MAM', 'noon')
Histo_MAM_maniana, Bins_MAM_maniana = Hitograma_estados_ini(df_15min, 'MAM', 'maniana')
Histo_MAM_tarde, Bins_MAM_tarde     = Hitograma_estados_ini(df_15min, 'MAM', 'tarde')

Histo_JJA_noon, Bins_JJA_noon       = Hitograma_estados_ini(df_15min, 'JJA', 'noon')
Histo_JJA_maniana, Bins_JJA_maniana = Hitograma_estados_ini(df_15min, 'JJA', 'maniana')
Histo_JJA_tarde, Bins_JJA_tarde     = Hitograma_estados_ini(df_15min, 'JJA', 'tarde')
