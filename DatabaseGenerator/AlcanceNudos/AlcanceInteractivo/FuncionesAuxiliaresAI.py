# Version 22 Febrero 2023
import numpy as np
import random
import sys
import Informes

def LeeParametros(Nombre_Archivo):
	fichero = open(Nombre_Archivo)
	lineas = fichero.readlines()
	Periodos = lineas[2]
	Fase = lineas[3]
	AlcanceInicial = lineas[4]
	Intervalos = lineas[5]
	p = ComentarioArreglo(Periodos, ' Periodos:')
	f = ComentarioArreglo(Fase, ' Fases:')
	a = ComentarioArreglo(AlcanceInicial,  ' Alcance:')
	i = ComentarioArreglo(Intervalos, 'element vertex ')
	return p, f, a, i

def LeeAlcances(RutaAlcances):
	Alcances = []
	fichero = open(RutaAlcances)
	lineas = fichero.readlines()
	lineas = lineas[5:]
	for l in lineas:
		Alcances.append(float(l))
	return Alcances

def LeeAumentos(RutaAumentos):
	Aumentos = []
	fichero = open(RutaAumentos)
	lineas = fichero.readlines()
	lineas = lineas[5:]
	for l in lineas:
		if l == 'False':
			Aumentos.append(False)
		else:
			Aumentos.append(True)
	return Aumentos

def ComentarioArreglo(comentario, tipo):
	if tipo == 'element vertex ':
		comentario = comentario.replace(tipo, '')
		arreglo_comentario = comentario.split()
		arreglo = [int(elemento) for elemento in arreglo_comentario]
	else:
		comentario = comentario.replace('comment'+tipo, '')
		comentario = comentario.replace('[', '')
		comentario = comentario.replace(']', '')
		comentario = comentario.replace(',', '')
		arreglo_comentario = comentario.split()
		arreglo = [float(elemento) for elemento in arreglo_comentario]
	if len(arreglo) == 1:
		return arreglo[0]
	return arreglo


def EscribeArchivoNudo(Nudo, Periodo, Fase, intervalos, salida, iteraciones):
	vertices =[]
	for octagonos in Nudo:
		for t in octagonos:
			vertices.append(t)
	caras = []
	aristas = []
	for i in range(intervalos):
		a = 16*i
		# caras.append([a, a+1, a+9, a+8])
		# caras.append([a+1, a+2, a+10, a+9])
		# caras.append([a+2, a+3, a+11, a+10])
		# caras.append([a+3, a+4, a+12, a+11])
		# caras.append([a+4, a+5, a+13, a+12])
		# caras.append([a+5, a+6, a+14, a+13])
		# caras.append([a+6, a+7, a+15, a+14])
		# caras.append([a+7, a, a+8, a+15])
		caras.append([a, a+1, a+16])
		caras.append([a+1, a+17, a+16])
		caras.append([a+1, a+2, a+17])
		caras.append([a+2, a+18, a+17])
		caras.append([a+2, a+3, a+18])
		caras.append([a+3, a+19, a+18])
		caras.append([a+3, a+4, a+19])
		caras.append([a+4, a+20, a+19])

		caras.append([a+4, a+5, a+20])
		caras.append([a+5, a+21, a+20])
		caras.append([a+5, a+6, a+21])
		caras.append([a+6, a+22, a+21])
		caras.append([a+6, a+7, a+22])
		caras.append([a+7, a+23, a+22])
		caras.append([a+7, a+8, a+23])
		caras.append([a+8, a+24, a+23])

		caras.append([a+8, a+9, a+24])
		caras.append([a+9, a+25, a+24])
		caras.append([a+9, a+10, a+25])
		caras.append([a+10, a+26, a+25])
		caras.append([a+10, a+11, a+26])
		caras.append([a+11, a+27, a+26])
		caras.append([a+11, a+12, a+27])
		caras.append([a+12, a+28, a+27])

		caras.append([a+12, a+13, a+28])
		caras.append([a+13, a+29, a+28])
		caras.append([a+13, a+14, a+29])
		caras.append([a+14, a+30, a+29])
		caras.append([a+14, a+15, a+30])
		caras.append([a+15, a+31, a+30])
		caras.append([a+15, a, a+31])
		caras.append([a+0, a+16, a+31])

	nombre_x = "cos(" + str(Periodo[0]) + "t+" +str(Fase[0]) + ')'
	nombre_y = "cos(" + str(Periodo[1]) + "t+" +str(Fase[1]) + ')'
	nombre_z = "cos(" + str(Periodo[2]) + "t+" +str(Fase[2]) + ')'
	info_nudo = nombre_x+nombre_y+nombre_z
	info_rutina = '('+ str(iteraciones)+ '-' +str(intervalos)  +')'
	nombre_nudo = salida+info_nudo+info_rutina
	Informes.ArchivoPLY(nombre_nudo, vertices, caras, aristas, iteraciones, Periodo, Fase)