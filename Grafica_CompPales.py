#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import pearsonr
# from mpl_toolkits.axes_grid1 import host_subplot
# import mpl_toolkits.axisartist as AA
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib.cm as cm
import matplotlib.font_manager as fm
import math as m
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
import matplotlib.colors as colors
##############################################################################
#-----------------------------------------------------------------------------
# Rutas para las fuentes -----------------------------------------------------
##############################################################################

prop = fm.FontProperties(fname='/home/nacorreasa/SIATA/Cod_Califi/AvenirLTStd-Heavy.otf' )
prop_1 = fm.FontProperties(fname='/home/nacorreasa/SIATA/Cod_Califi/AvenirLTStd-Book.otf')
prop_2 = fm.FontProperties(fname='/home/nacorreasa/SIATA/Cod_Califi/AvenirLTStd-Black.otf')

#####################################################################
## ----------------GRÁFICA DE LOS VECTORES PROPIAS---------------- ##
#####################################################################
vector_propio_348_1 = np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/VectorProp_1_348.npy')
vector_propio_348_2 = np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/VectorProp_2_348.npy')
vector_propio_348_3 = np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/VectorProp_3_348.npy')

vector_propio_350_1 = np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/VectorProp_1_350.npy')
vector_propio_350_2 = np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/VectorProp_2_350.npy')
vector_propio_350_3 = np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/VectorProp_3_350.npy')

vector_propio_975_1 = np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/VectorProp_1_975.npy')
vector_propio_975_2 = np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/VectorProp_2_975.npy')
vector_propio_975_3 = np.load('/home/nacorreasa/Maestria/Datos_Tesis/Arrays/VectorProp_3_975.npy')


labels = np.array(['Irradiancia', 'Pot', 'Temp', 'Den VA', 'FR', '$\eta$', 'PM 2.5'])
n=vector_propio_975_1.shape[0]

fig = plt.figure(figsize=[18, 6])
plt.rc('axes', edgecolor='gray')
ax1 = fig.add_subplot(1, 3, 1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
for i in range(n):
    ax1.arrow(0, 0, vector_propio_350_1[i], vector_propio_350_2[i], head_width=0.05, head_length=0.1, color = '#ee6622', alpha = 0.5)
    ax1.text(vector_propio_350_1[i]* 1.15, vector_propio_350_2[i] * 1.15, s = labels[i],  ha='center', va='center')
ax1.set_xlim(-1,1)
ax1.set_ylim(-1,1)
ax1.set_xlabel("Componente 1", fontproperties = prop_1)
ax1.set_ylabel("Componente 2", fontproperties = prop_1)
ax1.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.7)
ax1.set_title(u'Direcciones de variabilidad de \n las anomalías en el Oeste', fontsize = 10, fontproperties = prop)

ax2 = fig.add_subplot(1, 3, 2)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
for i in range(n):
    ax2.arrow(0, 0, vector_propio_975_1[i], vector_propio_975_2[i], head_width=0.05, head_length=0.1, color = '#ee6622', alpha = 0.5)
    ax2.text(vector_propio_975_1[i]* 1.15, vector_propio_975_2[i] * 1.15, s = labels[i],  ha='center', va='center')
ax2.set_xlim(-1,1)
ax2.set_ylim(-1,1)
ax2.set_xlabel("Componente 1", fontproperties = prop_1)
ax2.set_ylabel("Componente 2", fontproperties = prop_1)
ax2.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.7)
ax2.set_title(u'Direcciones de variabilidad de \n las anomalías en el Centro-Oeste', fontsize = 10, fontproperties = prop)

ax3 = fig.add_subplot(1, 3, 3)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
for i in range(n):
    ax3.arrow(0, 0, vector_propio_348_1[i], vector_propio_348_2[i], head_width=0.05, head_length=0.1, color = '#ee6622', alpha = 0.5)
    ax3.text(vector_propio_348_1[i]* 1.15, vector_propio_348_2[i] * 1.15, s = labels[i],  ha='center', va='center')
ax3.set_xlim(-1,1)
ax3.set_ylim(-1,1)
ax3.set_xlabel("Componente 1", fontproperties = prop_1)
ax3.set_ylabel("Componente 2", fontproperties = prop_1)
ax3.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.7)
ax3.set_title(u'Direcciones de variabilidad de \n las anomalías en el Este', fontsize = 10, fontproperties = prop)

plt.show()
