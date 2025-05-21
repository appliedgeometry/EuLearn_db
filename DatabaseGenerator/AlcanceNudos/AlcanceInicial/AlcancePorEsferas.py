# Version 14 de Febrero
# Alcance por esferas del nudo simple

import numpy as np

#Acota alcance
def acotaAlcancePorEsferas(Nudo, intervalos, iteraciones):
	#Alcance inicial
	factor = 0.3
	Pesos = []
	AlcancesPuntuales = []
	Aumentos = []
	indices = []
	dists = []
	mins = []
	autointerseccion = False
	i=0
	print('Buscando alcances iniciales:')
	print('Iteración 1')
	for i in range(intervalos):
		#los vecinos

		h = ( i - 1 )%intervalos
		j = ( i + 1 )%intervalos

		#Mide distancia a sus vecinos más proximos
		distancia1 = np.linalg.norm( Nudo[i] - Nudo[h] ) 
		distancia2 = np.linalg.norm( Nudo[i] - Nudo[j] )
		minimo = min(distancia1, distancia2)
		
		#Busca saturacion
		M = { h, i, j }
		peso = 0
		distancias = []

		for indice in range(intervalos):
			if indice not in M:
				distanciaIndice = np.linalg.norm( Nudo[i] - Nudo[indice] )
				if  distanciaIndice < minimo:
					peso = peso + 1
					distancias.append(distanciaIndice)

		if peso == 0:
			alcance_i = factor*minimo
			Aumentos.append(False)
		else:
			alcance_i = factor*min(distancias)
			Aumentos.append(True)
			indices.append(i)
			dists.append(round(factor*min(distancias),4))
			mins.append(round(factor*minimo,4))

			#Si no sirve el nudo por autointersección, acá termina
			if alcance_i < 1e-16:
				autointerseccion = True
				AlcancesPuntuales.append(alcance_i)
				print('Distancia cero =', round(alcance_i,4), 'en el índice', i)
				# return AlcancesPuntuales, Aumentos, autointerseccion

		AlcancesPuntuales.append(alcance_i)	
		Pesos.append(peso)

	#Si no sirve el nudo por saturación, acá termina
	if True in Aumentos:
		print('Saturado')#. Índices =', indices) #, 'dists =', dists, 'mins =', mins)
		# return AlcancesPuntuales, Aumentos, autointerseccion

	A2 = AlcancesPuntuales.copy()
	################	
	################

	for j in range(2, iteraciones + 1):
		print('Iteración', j)
		for i in range(intervalos):
			# Va a calcular la distancia permitida
			# M es el radio que va a excluir
			M={i}
			for a in range(j):
				M.update( [(i-a)%intervalos, (i+a)%intervalos ])

			D = []
			for m in M:
				if m != i:
					#esto puede mejorar
					distancia_i_m = np.linalg.norm( Nudo[i] - Nudo[m] )
					D.append( distancia_i_m )

			maximo = max(D)
			###############

			peso = 0
			distancias = []

			#Distancias a todo el nudo
			for indice in range(intervalos):
				if (indice not in M) and (Aumentos[indice] != True):
					#esto puede mejorar
					distanciaIndice = np.linalg.norm( Nudo[i] - Nudo[indice] )
					if  distanciaIndice <= maximo:
						peso = peso + 1
						distancias.append(distanciaIndice)

			if peso == 0:
				alcance_i = factor*maximo
				Aumentos[indice] = False
			else:
				d = min(distancias)
				alcance_i = factor*d
				Aumentos[indice] = True
			AlcancesPuntuales[i] = alcance_i
			Pesos = peso
	if autointerseccion == True:
		print("No debería pasar por autointersección -----")

	return AlcancesPuntuales, Aumentos, autointerseccion
