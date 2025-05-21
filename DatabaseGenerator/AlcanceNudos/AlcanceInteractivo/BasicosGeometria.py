import numpy as np

def punto_distancia_dada(A, B, d):
  """ Regresa un punto a distancia d de A en la direccion B:
  """
  AB= B-A
  norma = (d/np.linalg.norm(AB))
  punto = AB*norma
  return  punto+A

def distancia_dos_puntos(A,B):
  """ Calcula la distancia entre dos puntos:
  """
  p = B-A
  d = np.linalg.norm(p)
  return d

def punto_razon_dada(v,w, razon):
  """ Da un punto en el segmento U V a razon: 1+razon. El punto es mas 
  cercano a v
  """
  x=(w[0]+(v[0]*razon))/(1+razon)
  y=(w[1]+(v[1]*razon))/(1+razon)
  z=(w[2]+(v[2]*razon))/(1+razon)
  m=np.array([x, y, z])
  return m

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def punto_EnLaRecta_con_direccion(a,b,t):
  """ Regresa el punto en la recta a+tb (parametrizacion de la recta)
  """
  punto = []
  punto.append( a+t*b )
  return punto

def dos_puntos_enLaRecta(a,b,t):
  direccion = b-a
  punto = punto_EnLaRecta_con_direccion(a,direccion,t)
  return punto


def Haz_Barra(A, B, num_puntos):
  puntos = []
  for i in range(num_puntos):
    puntos.append( dos_puntos_enLaRecta(A, B, i/num_puntos)[0] )
  puntos.append(B)
  return puntos

def Haz_Barras(Circ1, Circ2, num_puntos):
  puntos = []
  for i in range(8):
    Barra = Haz_Barra(Circ1[i], Circ2[i], num_puntos)
    puntos.append(Barra)
  return puntos

def Puntos_Conexion(intervalos, num_de_barras):
  Puntos = []
  for i in range(num_de_barras):
    limite = intervalos//2
    a= np.random.randint(0, limite)
    b= np.random.randint(limite+1, intervalos)
    Puntos.append([a,b])
  return Puntos

def distancia_punto_a_conjunto(punto, conjunto):
  m = []
  for c in conjunto:
    m.append(distancia_dos_puntos(punto, c))
  M=np.array(m)
  return np.amin(M)

def conjunto_sin_punto(indice, conjunto):
  b=[]
  b=conjunto.copy()
  b=np.delete(b, indice)
  b=np.delete(b, indice+1)
  b=np.delete(b, indice+1)
  b=np.delete(b, indice+1)
  b=np.delete(b, indice-1)
  b=np.delete(b, indice-1)
  b=np.delete(b, indice-1)
  return b


##################################
#############        #############

#recibe dos planos
#regresa una recta  [Q, v]
def Interseccion_Dos_Planos(p1, p2):  
  x=np.array( [ p1[0], p1[1], p1[2] ] )
  y=np.array( [ p2[0], p2[1], p2[2] ] )
  v=np.cross(x,y)

  A=np.array( [ p1[0], p1[1], p1[2], p1[3] ] )
  B=np.array( [ p2[0], p2[1], p2[2], p2[3] ] )
  constantes=np.array( [-A[3], -B[3]] )

  while True:
    try:
      a=np.array( [ [A[0], A[1]], [B[0], B[1]] ] )
      q = np.linalg.solve(a,constantes)  
      break
    except:
      pass
    try:
      a=np.array( [ [A[0], A[2]], [B[0], B[2]] ] )
      q = np.linalg.solve(a,constantes)
      break
    except:
      pass
    try: 
      a=np.array( [ [A[1], A[2]], [B[1], B[2]] ] )
      q = np.linalg.solve(a,constantes)   
      break
    except:
      q = np.array([0,0,0])
      break
  
  Q=[q[0], q[1], 0]
  r=[Q,v]
  return r

##################################

def BaseDeUnPlano(Triangulo):
  #Regresa dos vectores
  inicio = Triangulo[0]
  final1=Triangulo[1]
  final2=Triangulo[2]

  v1 = final1-inicio
  v2 = final2-inicio

  vector1 = unit_vector(v1)
  vector2 = unit_vector(v2)

  #print('producto punto', np.dot(v1, v2))
  #print(angle_dot(v1, v2))

  return vector1, vector2

###################################################
##################################################

def PuntoDeProyeccionPuntoARecta(Punto, Recta):
  Q=Recta[0]
  u=Recta[1]
  #Calcula D
  D=-(Punto[0]*u[0])+(Punto[1]*u[1])+(Punto[2]*u[2])
  lam = -(u[0]*Q[0]+u[1]*Q[1]+u[2]*Q[2]+D)/(u[0]*u[0]+u[1]*u[1]+u[2]*u[2] )
  punto_interseccion =Q+lam*u
  return punto_interseccion

def HazRecta(Recta, Intervalo):
  L=[]
  t = np.linspace(Intervalo[0], Intervalo[1], 2)
  for p in t:
    Nuevo_Punto = Recta[0]+p*Recta[1]
    L.append( Nuevo_Punto )
  print('Puntos de la Recta:')
  print(L[0])
  print(L[1])
  print('....')
  return L

############
#Recibe punto y recta
#Regresa la distancia de un punto a una recta
#funcion escalar
def Distancia_Punto_A_Recta(Punto, Recta):
  '''Regresa la distancia del punto P a la recta
    R=Q+tv
  '''
  print('Punto', Punto)
  Q=np.array(Recta[0])
  v=np.array(Recta[1])

  print('v',v)
  print('Q', Q)
  #print('r', v)
  QP = Punto - Q
  #print('Q-P', QP)
  m = np.cross(QP, v)
  print('m', m)
  
  norm_m = np.linalg.norm(m) 
  norm_v = np.linalg.norm(v)
  if norm_v !=0:
    distancia = norm_m/norm_v
  else:
    distancia = 0
  return distancia


def Distancia_Punto_A_Recta2(P, r):
  '''Regresa la distancia del punto P a la recta
  r=Q+tv
  '''
  Q=np.array(r[0])
  v= unit_vector( np.array(r[1]) )
  #print('Q', Q)
  #print('r', v)
  QP = P - Q
  #print('Q-P', QP)
  m = np.cross(QP, v)
  #print('m', m)
  M = m*v
  S=QP-M
  norm_v = np.linalg.norm(S)
  #print('norm_m', norm_m)
  #print('norm_v', norm_v)
  if norm_v !=0:
    distancia = norm_v
  else:
    distancia = 0
  return distancia

######################
######################

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
    #P=a b c
    #Q=d e f
    #R=g h i

    #a d g
    #b e h
    #c f i

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

def angle_dot(a, b):
  dot_product = np.dot(a, b)
  prod_of_norms = np.linalg.norm(a) * np.linalg.norm(b)
  angle = round(np.degrees(np.arccos(dot_product / prod_of_norms)), 1)
  return angle 

def ChecaAngulos(Nudo_Inicial):
  for p in Nudo_Inicial:
    print(angle_dot(p[1]-p[0], p[2]-p[0]))



