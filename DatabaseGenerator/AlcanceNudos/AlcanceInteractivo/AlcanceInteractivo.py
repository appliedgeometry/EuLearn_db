from InterseccionPoligonos import InterseccionNudoConPoligonoJ as InterseccionJ
import BasicosGeometria
import GeometriaCirculos
import numpy as np
import DefinicionCaras

#Crea un escenario hipotetico
def Temporal(Nudo_Inicial, Nudo8, Alcances, Aumento):
    Alcances_Temporal = np.array([])
    for j in range( len(Nudo_Inicial) ):
        if Aumento[j]==False:
            Alcances_Temporal = np.append(Alcances_Temporal, Alcances[j]+0.005)
        else:
            Alcances_Temporal = np.append(Alcances_Temporal, Alcances[j])
    #va a hacer nudo de 4 vertices con los nuevos alcances
    for j in range( len(Nudo_Inicial) ):
        origen = Nudo_Inicial[j][0]
        Base_j = BasicosGeometria.BaseDeUnPlano(Nudo_Inicial[j])
        Nudo8[j] = GeometriaCirculos.Circulo(
                            Centro = origen, 
                            Base = Base_j, 
                            radio = Alcances_Temporal[j], 
                            particion=8)
    #Van a ser hipoteticos
    return Nudo8, Alcances_Temporal


def Califica(Nudo_Inicial, Nudo_Temporal, Aumento):
    intervalos = len(Nudo_Inicial)
    CarasCilindros = []
    for j in range(intervalos):
        k = (j+1)%intervalos
        CaraJ = DefinicionCaras.CarasPuntoJ(Nudo_Temporal[j], Nudo_Temporal[k])
        CarasCilindros.append(CaraJ)

    for j in range( len(Nudo_Inicial) ):
        origen = Nudo_Inicial[j][0]
        alcance = Alcances_Temporal[j]
        if Aumento[j]== False:
            Aumento[j] = InterseccionJ(Nudo_Inicial, origen, alcance,Nudo_4vertices, j)
        if Aumento[j] == False:
            Alcances[j] = Alcances_Temporal[j]
        #Si es True no cambia el alcance
    return Aumento, Alcances


def Corrige(Nudo_Inicial, Nudo_4vertices, Alcances, Aumento):
    for j in range( len(Nudo_4vertices) ):
        if Aumento[j] == False :
            origen = Nudo_Inicial[j][0]
            Base_j = BasicosGeometria.BaseDeUnPlano(Nudo_Inicial[j])
            Nudo_4vertices[j] = GeometriaCirculos.Circulo(
                                Centro = origen, 
                                Base = Base_j, 
                                radio = Alcances[j], 
                                particion=4)
    return Nudo_4vertices, Alcances, Aumento