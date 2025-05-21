import numpy as np

def CarasPuntoJ(Poligono_j, Poligono_k):
	Caras = []
	
	cara1 = [Poligono_j[0], Poligono_j[1], Poligono_k[1], Poligono_k[0]]
	Caras.append(cara1)

	cara2 = [Poligono_j[1], Poligono_j[2], Poligono_k[2], Poligono_k[1]]
	Caras.append(cara2)

	cara3 = [Poligono_j[2], Poligono_j[3], Poligono_k[3], Poligono_k[2]]
	Caras.append(cara3)

	cara4 = [Poligono_j[3], Poligono_j[4], Poligono_k[4], Poligono_k[3]]
	Caras.append(cara4)

	cara5 = [Poligono_j[4], Poligono_j[5], Poligono_k[5], Poligono_k[4]]
	Caras.append(cara5)

	cara6 = [Poligono_j[5], Poligono_j[6], Poligono_k[6], Poligono_k[5]]
	Caras.append(cara6)

	cara7 = [Poligono_j[6], Poligono_j[7], Poligono_k[7], Poligono_k[6]]
	Caras.append(cara7)

	cara8 = [Poligono_j[7], Poligono_j[8], Poligono_k[8], Poligono_k[7]]
	Caras.append(cara8)

	return Caras
