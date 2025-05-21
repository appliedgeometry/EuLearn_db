import numpy as np
import GeometriaCirculos
import BasicosGeometria
import DefinicionCaras

#quiero evaluar la interseccion del poligono-j con los demás poligonos
def InterseccionJ(Nudo_Inicial, Nudo8, alcance, j ):
    Caras_j
    poligonoj=Nudo8[j]
    intervalos = len(Nudo_Inicial)
    h = ( i - 1 )%intervalos
    j = ( i + 1 )%intervalos
    M = { h, i, j }
    #Inicializo suponiendo que es falso
    interseccion = False
    #i va sobre a recorrer el nudo
    i = 0
    #Falso o antes de que recorra todo el nudo
    while Interseccion == False and i<len(Nudo8):
        #Va a pasar por todos los indices del nudo
        if i!=j:
            #Por cada i hace un poligono
            Poligono2 = Nudo8[i]
            #Regresa false si no hay interseccion
            #Regresa true si encuentra interseccion
            Interseccion = interseccionPoligonos(origen, alcance, EcuacionPlano, Poligono2)
            # if Interseccion ==  True:
            #     print('--------')
            #     print(Poligono1)
            #     print(Poligono2)
        i = i+1
    return Interseccion





#Empieza suponiendo que no hay interseccion
#El punto fijo es j
def InterseccionNudoConPoligonoJ(Nudo_Inicial, origen, alcance, Nudo_4vertices, j):
    #origen = Nudo_Inicial[j][0]
    #Inicializa variable interseccion
    Interseccion = False
    
    #Hace poligono j
    Poligono1 = Nudo_Inicial[j]
    EcuacionPlano = EcuacionGeneralPlano(Nudo_Inicial[j][1], Nudo_Inicial[j][0], Nudo_Inicial[j][2])

    #i va a ir moviendo los demás puntos de nudo
    i = 0
    #Falso o antes de que recorra todo el nudo
    while Interseccion == False and i<len(Nudo_4vertices):
        # print('....')
        # print('j,i', j, i, sep="-")
        #Va a pasar por todos los indices del nudo
        if i!=j:
            #Por cada i hace un poligono
            Poligono2 = Nudo_4vertices[i]
            #Regresa false si no hay interseccion
            #Regresa true si encuentra interseccion
            Interseccion = interseccionPoligonos(origen, alcance, EcuacionPlano, Poligono2)
            # if Interseccion ==  True:
            #     print('--------')
            #     print(Poligono1)
            #     print(Poligono2)
        i = i+1
    return Interseccion


#Regresa False si no hay intersección
#Regresa True si hay intersección
#Prueba con las 4 rectas del poligono2
def interseccionPoligonos(origen, alcance, EcuacionPlano, Poligono2):
    #Empieza suponiendo que no hay interseccion  
    a = [[0,1], [1,2], [2,3], [3,0]]
    i=0
    Interseccion = False
    while Interseccion == False and i<=3:
        Puntos = [ Poligono2[ a[i][0] ], Poligono2[ a[i][1] ] ]
        Interseccion = interseccionPoligonoRecta(origen, alcance, EcuacionPlano, Puntos)
        i = i+1
    return Interseccion


#Poligono = np.array ( [[p1], [p2], [p3],[p4] ] )
#interseccion de un poligono con una recta
def interseccionPoligonoRecta(origen, alcance, EcuacionPlano, Puntos):
    #A, B, C, D = EcuacionGeneralPlano(Poligono[0], Poligono[1], Poligono[2])
    A, B, C, D = EcuacionPlano
    #Punto en el plano
    X0, Y0, Z0 = origen[0], origen[1], origen[2]
    #punto 1 de la recta
    Recta =  EcuacionParametricaRecta(Puntos[0], Puntos[1])
    x0, y0, z0 = Recta[0][0], Recta[0][1], Recta[0][2]
    a, b, c = Recta[1][0], Recta[1][1], Recta[1][2]
    t1 = A*(X0-x0)+B*(Y0-y0)+C*(Z0-z0)
    t2 =  (A*a)+(B*b)+(C*c)
    if t2==0:
        return False
    else:
        T= t1/t2
        punto = Recta[0]+(T*Recta[1])
        distancia = BasicosGeometria.distancia_dos_puntos(origen, punto)
        if 0<= T <= 1 and distancia<=alcance:
            #print('T', T)
            return True
    return False







####################################################################
####################################################################

def EcuacionGeneralPlano(A, B, C):
    '''Regresa un arreglo con los coeficientes ax+by+cz+d=0
        del plano generado por A, B, C
        A, B, C puntos en R3
        AB - BC
        B es origen
    '''
    P=[B[0]-A[0], B[1]-A[1], B[2]-A[2]]
    Q=[C[0]-B[0], C[1]-B[1], C[2]-B[2]]
    R= C

    #ae-bd
    coef_z = P[0]*Q[1]-Q[0]*P[1]
    #print('z:', coef_z)

    #-af+dc
    coef_y = Q[0]*P[2]-P[0]*Q[2]
    #print('y:', coef_y)

    #bf-ce
    coef_x = P[1]*Q[2]-P[2]*Q[1]
    #print('x:', coef_x)

    #afh+dbi+gce-aei-dch-bbf
    constante = P[0]*Q[2]*R[1] + Q[0]*P[1]*R[2] +R[0]*P[2]*Q[1] - P[0]*Q[1]*R[2] -Q[0]*P[2]*R[1] -R[0]*P[1]*Q[2]
    #print('constante:', constante)
    plano = [coef_x, coef_y, coef_z, constante]
    return plano 


#R[0]+tR[1]
def EcuacionParametricaRecta(Punto1, Punto2):
    P1=np.array(Punto1)
    P2=np.array(Punto2)
    V = np.array(P2-P1)
    R = np.array([Punto1, V])
    Recta = [R[0], R[1]]
    return Recta



#Poligono = np.array ( [[p1], [p2], [p3],[p4] ] )
#interseccion de un poligono con una recta
# def interseccionPoligonoRecta(Poligono, Puntos):
#     Recta1 = EcuacionParametricaRecta(Poligono[0], Poligono[1])
#     Recta2 = EcuacionParametricaRecta(Poligono[0], Poligono[3])
#     Plano = [Recta1[1], Poligono[0], Recta2[1]]

#     Recta = EcuacionParametricaRecta(Puntos[0], Puntos[1])   
    
#     #Interseccion con recta 1
#     A = np.array([ [Plano[1][0], Plano[2][0], -1*Recta[1][0] ],
#                    [Plano[1][1], Plano[2][1], -1*Recta[1][1] ], 
#                    [Plano[1][2], Plano[2][2], -1*Recta[1][2] ] ])
#     B= np.array([ Recta[0][0]-Plano[0][0], 
#                   Recta[0][1]-Plano[0][1], 
#                   Recta[0][2]-Plano[0][2] ])
#     try:
#         x = np.linalg.solve(A,B)
#         # print(x)
#         if 0<=x[0]<=1 and 0<=x[1]<=1 and 0<=x[2]<=1:
#             return True
#         else:
#             return False
#     except np.linalg.LinAlgError:
#         return False