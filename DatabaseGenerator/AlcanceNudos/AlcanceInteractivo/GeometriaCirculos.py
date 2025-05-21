import numpy as np
import BasicosGeometria

#Base del plano = [a,b,c] [d,e,f]
def Circulo(Centro, Base, radio, particion):
#Regresa circunferencia (arreglo de puntos) alrededor de un punto
	Circunferencia = []
	t = np.linspace(0, (2)*(np.pi), particion, endpoint=False)
	x = Centro[0] + radio*np.cos(t)*Base[0][0] + radio*np.sin(t)*Base[1][0]
	y = Centro[1] + radio*np.cos(t)*Base[0][1] + radio*np.sin(t)*Base[1][1]
	z = Centro[2] + radio*np.cos(t)*Base[0][2] + radio*np.sin(t)*Base[1][2]
	
	for i in range(particion):
		Circunferencia.append( [x[i], y[i], z[i] ] )
	return Circunferencia


def Circulo_2(Centro, Base, punto2):
	particion = 20
	radio = BasicosGeometria.distancia_dos_puntos(Centro, punto2)
	Circunferencia = []
	t = np.linspace(0, (2)*(np.pi), particion, endpoint=False)
	x = Centro[0] + radio*np.cos(t)*Base[0][0] + radio*np.sin(t)*Base[1][0]
	y = Centro[1] + radio*np.cos(t)*Base[0][1] + radio*np.sin(t)*Base[1][1]
	z = Centro[2] + radio*np.cos(t)*Base[0][2] + radio*np.sin(t)*Base[1][2]
	for i in range(particion):
		Circunferencia.append( [x[i], y[i], z[i] ] )
	return Circunferencia

#####################################################################

#Poligonos
#Circulo1 =  4 puntos [p1, p2, p3, p4]
#Circulo2 =  4 puntos [q1, q2, q3, q4]
def Interseccion_Dos_Poligonos(Circulo_1, Circulo_2):
	Interseccion_Boolean = False
	i=0
	while Interseccion_Boolean == False and i<4:
		Recta_Poligono = EcuacionParametricaRecta(Circulo_2[i], Circulo_2[(i+1)%4])
		Interseccion_Boolean = Interseccion_Poligono_Recta(Circulo_1, Recta_Poligono)
		i=i+1
	return Interseccion_Boolean

def Interseccion_Poligono_Recta(Circulo, Recta):
	Interseccion_Boolean = False
	i=0
	while Interseccion_Boolean == False and i<4:
		Recta_Poligono = EcuacionParametricaRecta(Circulo[i], Circulo[(i+1)%4])
		Interseccion_Boolean = Interseccion_Dos_Rectas(Recta, Recta_Poligono)
		i=i+1
	return Interseccion_Boolean

#def Interseccion_Circulo_Recta(Circulo, Recta)
#recta1 = [ [p1,p2,p3], [v1,v2,v3] ]
#recta2 = [ [q1,q2,q3], [w1,w2,w3] ]
def Interseccion_Dos_Rectas(Recta_1, Recta_2):
#Regresa la interseccion de las dos rectas
	p1=Recta_1[0][0]
	p2=Recta_1[0][1]
	p3=Recta_1[0][2]
	v1=Recta_1[1][0]
	v2=Recta_1[1][1]
	v3=Recta_1[1][2]

	q1=Recta_2[0][0]
	q2=Recta_2[0][1]
	q3=Recta_2[0][2]
	w1=Recta_2[1][0]
	w2=Recta_2[1][1]
	w3=Recta_2[1][2]

	a = np.array([[v1, -w1],[v2, -w2] ])
	b = np.array([ q1-p1,q2-p2 ])
	#Verifica si son paralelas
	R1 = [ v1, v2, v3 ]
	R2 = [ w1, w2, w3 ]

	if R1==[0,0,0] or R2==[0,0,0]:
		print('Algo salio mal 123')

	v_1 = hazMonico(R1)
	v_2 = hazMonico(R2)

	if (v_1==v_2).all():
		print('Paralelas')
		return True
	try:
		x = np.linalg.solve(a,b)
		# print(x)
		if 0<=x[0]<=1 and 0<=x[1]<=1:
			return True
		else:
			return False
	except np.linalg.LinAlgError:
		return False


def EcuacionParametricaRecta(Punto1, Punto2):
	P1=np.array(Punto1)
	P2=np.array(Punto2)
	V = np.array(P2-P1)
	R = np.array([Punto1, V])
	Recta = [R[0], R[1]]
	return Recta

def hazMonico(vector):
	if vector==[0,0,0]:
		return False
	else:
		if vector[0]!=0:
			monico = vector/vector[0]
			return monico
		elif vector[1]!=0:
			monico = vector/vector[1]
			return monico
		else:
			monico = vector/vector[2]
			return monico


# Recta_1 = np.array([ [1,0,0], [1,0,0] ]) 
# Recta_2 = np.array([ [1,0,0], [0,1,0] ]) 
# Punto1=np.array([2,1,1])
# Punto2=np.array([1,1,1])
# Recta3=EcuacionParametricaRecta(Punto1, Punto2)

# Circulo1 = np.array([ [0,0,0],[0,3,0],[0,3,3],[0,0,3] ])
# Circulo2 = np.array([ [0,0,0],[1,0,0],[1,1,0],[1,0,0] ])

# a = Interseccion_Dos_Poligonos(Circulo1, Circulo2)
# if a == True:
# 	print('Se intersectan')
# else:
# 	print('No Se intersectan')


# Circulo1 = np.array([ [1,0,0],[1,1,0],[2,1,0],[2,0,0] ])
# Circulo2 = np.array([ [0,2,0],[0,5,0],[0,5,5],[0,2,5] ])

# a = Interseccion_Dos_Poligonos(Circulo1, Circulo2)
# if a == True:
# 	print('Se intersectan')
# else:
# 	print('No Se intersectan')


# Recta_1 = np.array([ [1,0,0], [1,0,0] ]) 
# Recta_2 = np.array([ [2,0,0], [1,0,0] ]) 
# a = Interseccion_Dos_Rectas(Recta_1, Recta_2)
# print(a)



