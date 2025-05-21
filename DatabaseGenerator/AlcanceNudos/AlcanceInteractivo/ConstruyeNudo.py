import numpy as np
import Coseno
import Informes
import BasicosGeometria
import GeometriaCirculos
import FuncionesAuxiliaresAI
from AlcanceInteractivo import Temporal as Temporal
from AlcanceInteractivo import Corrige as Corrige
from AlcanceInteractivo import Califica as Califica



#PreNudo: Regresa nudo con alcance inicial
#         Regresa información de la curvatura
#NudoAlcancePuntual: hace nudo calculando alcance puntual
#=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#          Prenudo          #
#=-=-=-=-=-=-=-=-=-=-=-=-=-=#
def HacerPreNudo(periodo, fase, intervalos, alcance_inicial):
  Nudo = []
  f = fase
  p = periodo
  #Divide al intervalo (-pi, pi) en "intervalos" iguales
  t = np.linspace(-np.pi, np.pi, intervalos, endpoint=True)

  #Coordenadas (x,y,z) del nudo
  x, y, z = Coseno.Evalua(t, p[0], f[0]), Coseno.Evalua(t, p[1], f[1]), Coseno.Evalua(t, p[2], f[2])
  #x, y, z = np.cos(t+1), np.cos(3*t), np.cos(5*t)

  #Primera derivada 
  #rprimaX, rprimaY, rprimaZ = (-1)*np.sin(t+1), (-3)*np.sin(3*t), (-5)*np.sin(5*t)
  rprimaX, rprimaY, rprimaZ = Coseno.PrimeraDerivada(t, p[0], f[0]), Coseno.PrimeraDerivada(t, p[1], f[1]), Coseno.PrimeraDerivada(t, p[2], f[2])
  
  #Segunda derivada
  #r2primaX, r2primaY, r2primaZ = (-1)*np.cos(t+1), (-9)*np.cos(3*t), (-25)*np.cos(5*t)
  r2primaX, r2primaY, r2primaZ = Coseno.SegundaDerivada(t, p[0], f[0]), Coseno.SegundaDerivada(t, p[1], f[1]), Coseno.SegundaDerivada(t, p[2], f[2])

 
  for j in range(intervalos):
    origen =[ x[j], y[j], z[j] ]
    r1 = [ rprimaX[j], rprimaY[j], rprimaZ[j] ]
    r2 = [ r2primaX[j], r2primaY[j], r2primaZ[j] ]
          
    #tangente
    tan1 = origen+BasicosGeometria.unit_vector( [ rprimaX[j], rprimaY[j], rprimaZ[j] ] )
    tan = BasicosGeometria.punto_distancia_dada(origen, tan1, alcance_inicial)

    #normal
    r2_Cruz_r1 = np.cross( r2, r1 )
    r1_Cruz_r2_Cruz_r1 = np.cross(r1, r2_Cruz_r1)
    N1 = np.linalg.norm(r2_Cruz_r1) #Calculo norma 1
    N2= np.linalg.norm(r1) #Calculo norma 2
    normal1 = origen+BasicosGeometria.unit_vector(r1_Cruz_r2_Cruz_r1)  #Lo hago unitario
    normal = BasicosGeometria.punto_distancia_dada(origen, normal1, alcance_inicial) #Pongo el vector a distancia

    #binormal
    bin1 = np.cross( tan, normal ) #Producto cruz
    bin2 = origen+BasicosGeometria.unit_vector(bin1) #Lo hago unitario
    biN = BasicosGeometria.punto_distancia_dada(origen,bin2, alcance_inicial) 

    #Nudo.append([origen, normal, biN])
    #Nudo.append([origen, tan, biN])
    #Nudo.append([origen, tan, normal])
    Nudo.append([tan, normal, biN])

  return Nudo

#=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#    Nudo Interactivo       #
#=-=-=-=-=-=-=-=-=-=-=-=-=-=#

def AlcanceInteractivo(RutaNudo, RutaAlcances, RutaAumentos, Salida, iteraciones):
    Periodo, Fase, alcance, intervalos = FuncionesAuxiliaresAI.LeeParametros(RutaNudo)
    Alcances = FuncionesAuxiliaresAI.LeeAlcances(RutaAlcances)
    #Aumentos = FuncionesAuxiliaresAI.LeeAumentos(RutaAumentos) 

    #Nudo inicial se hace con un alcance global
    Nudo_Inicial =  HacerPreNudo(Periodo, Fase, intervalos, alcance)

    #Hace poligonos con 8 puntos
    Nudo8 = []
    for j in range(intervalos):
        origen = Nudo_Inicial[j][0]
        Base_j = BasicosGeometria.BaseDeUnPlano(Nudo_Inicial[j])
        circulo = GeometriaCirculos.Circulo(
                                    Centro = origen, 
                                    Base = Base_j, 
                                    radio=Alcances[j], 
                                    particion=16)
        Nudo8.append(circulo)
    
    FuncionesAuxiliaresAI.EscribeArchivoNudo(Nudo8, Periodo, Fase, intervalos, Salida, iteraciones)
