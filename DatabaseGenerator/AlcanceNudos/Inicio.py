# Version 4 de marzo 2023
# incluir path de carpetas 
import sys
sys.path.append('AlcanceInicial') #BuscaAlcanceMinimo
sys.path.append('AlcanceInteractivo') #ConstruyeNudo, FuncionesAuxiliaresAI
sys.path.append('AlcanceInicial\Ternas') #GuardaDatos

#librerias necesarias para este modulo
import time
import os
import random
import numpy as np
import BuscaAlcanceMinimo
import ConstruyeNudo
import GuardaDatos
from FuncionesAuxiliares import LeeTernas
from FuncionesAuxiliaresAI import LeeAlcances


# #################################################
# # Nudos de Lissajous
# # HACE TANTOS NUDOS COMO QUIERO
# # 1 nudo por default
# # Puntos por default en  la parametrizacion: 500 
# # Ternas.txt
# #################################################

  

def Rutina(Indices = [100], intervalos = 3000, iteraciones = 5, Fases = [[0,0,0]], umbral=0.001):
	start = time.process_time()
	A,B,C,D,E = [],[],[],[],[]
	for indice in Indices:
		for fase in Fases:
			print('---------------------------------')
			print('Indice en el archivo Ternas.txt: ', indice)

			RutaNudo, RutaAlcances, RutaAumentos, sirve = BuscaAlcanceMinimo.HazNudos(indice = indice,
																				intervalos = intervalos,
																				iteraciones = iteraciones,
																				Fase = fase,
																				umbral = umbral)
			# if sirve:
			Alcs = LeeAlcances(RutaAlcances)
			A.append(LeeTernas([indice])[0])
			B.append(fase)
			C.append(min(Alcs))
			D.append(max(Alcs))
			E.append([])
			# E.append([(1/3*np.pi, 2/3*np.pi, 200), (-1/3*np.pi, 2/3*np.pi, 200)])
			RutaSalida ='Nudos/Inflados/'
			# ConstruyeNudo.AlcanceInteractivo(RutaNudo, 
			# 							RutaAlcances, 
			# 							RutaAumentos,
			# 							RutaSalida, 
			# 							iteraciones)
	GuardaDatos.GuardaDatos(A,B,C,D,E)
	end = time.process_time()
	print('Ejecución en:', end - start, 'segundos')



def main():
    Rutina(Indices = [3], #range(29,30),
    	intervalos = 500,
    	iteraciones = 10,
    	Fases = [[0,0,np.pi/3]],
    	umbral = 1e-16)

if __name__ == "__main__":
    main()
