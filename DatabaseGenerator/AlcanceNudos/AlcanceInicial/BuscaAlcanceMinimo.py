#Version 21-Febrero 2023
import numpy as np
##########################
import FuncionesAuxiliares
from AlcancePorEsferas import acotaAlcancePorEsferas as acotaAlcancePorEsferas


##############################################################
#    Verifica que el nudo
# 1) No tenga puntos repetidos 
# 2) Tenga alcance aceptable, menor a 0.03 por default
# 3) Parametrización es de -pi a pi
# 4) x, y, z son arreglos
#############################################################

#Hace tantos nudos como se indique en las ternas
#Lee la terna correspondiente del archivo Ternas.txt
#Valores dafault 100,500, 5, [0,0,0]
#Regresa las rutas de los archivos:
#Ruta
#RutaAlcances
#RutaAumentos
def HazNudos(indice = 100, intervalos = 500, iteraciones = 5, Fase = [1,1,1], umbral=0.001):
  terna = FuncionesAuxiliares.LeeTernas([indice])
  print("-Terna: ", terna[0])
  print("-Fase: ", Fase)
  Ruta, RutaAlcances, RutaAumentos, sirve = TrabajaNudo(Terna=terna[0],
                                                fases=Fase, 
                                                intervalos=intervalos,
                                                iteraciones = iteraciones,
                                                umbral = umbral)
  return Ruta, RutaAlcances, RutaAumentos, sirve

# 1) Hace nudo original
# 2) Acota alcance puntual por esferas
# 3) Escribe archivo de alcances puntuales
# 4) Escribe archivo de aumentos
# 5) Clasifica si sirve o no dependiendo del umbral o si tiene autointersecciones
def TrabajaNudo(Terna, fases, intervalos, iteraciones, umbral):
  a,b,c = Terna[0], Terna[1], Terna[2]
  t = np.linspace(0, 2*np.pi, intervalos, endpoint=False)
  
  # Lissajous:
  x,y,z = np.cos((a*t)+fases[0]), np.cos((b*t)+fases[1]), np.cos((c*t)+fases[2])
  
  # Fibonacci:
  # x,y = np.cos((a*t)+fases[0]), np.cos((b*t)+fases[1]), 
  # z = 0.5*np.cos((c*t)+fases[2]) + 0.5*np.sin((b*t)+fases[1])

  # Trefoil
  # x,y,z = np.sin(t) + 2*np.sin(t*2),  np.cos(t) - 2*np.cos(t*2),  - np.sin(3*t)

  # Eight
  # x,y,z = (2 + np.cos(2*t))*np.cos(3*t), (2 + np.cos(2*t))*np.sin(3*t), np.sin(4*t)

  
  #Hace el nudo con coordenadas [x[i], y[i], z[i]]
  Nudo = [ [x[i], y[i], z[i]] for i in range(intervalos) ]
  Nudo = np.array(Nudo)

  Alcances, Aumentos, autointerseccion = acotaAlcancePorEsferas(Nudo, intervalos, iteraciones)
  RutaAlcances = FuncionesAuxiliares.EscribeAlcancesPuntualesTXT(Alcances, Terna, fases, Nudo, iteraciones)
  RutaAumentos = FuncionesAuxiliares.EscribeAumentosTXT(Aumentos, Terna, fases)

  alcmin = min(Alcances)
  alcmax  = max(Alcances)

  if autointerseccion:
    sirve = False
    motivo = 'Autoint'
    print('No sirve: autointersección. Alcance mín =', round(alcmin,4), 'máx =', round(alcmax,4))
  else:
    if alcmin < umbral:
      sirve = False
      motivo = 'Alcance'
      print('No sirve: alcance menor al umbral =', umbral, ', alcance mín =', round(alcmin,4), 'máx =', round(alcmax,4))
    elif True in Aumentos:
      sirve = False
      motivo = 'Aumentos'
      print('No sirve: saturación. Alcance mín =', round(alcmin,4), 'máx =', round(alcmax,4))
    else:
      sirve = True
      motivo = 'Sirve'
      print('Sí sirve. Alcance mín =', round(alcmin,4), 'máx =', round(alcmax,4))

  ruta = 'Nudos/Sencillos/'
  Ruta = FuncionesAuxiliares.GeneraPLYdeNudo(ruta, Nudo, Terna, fases, alcmin, iteraciones, sirve, motivo)
  return Ruta, RutaAlcances, RutaAumentos, sirve

