import numpy as np

def ArchivoPLY(Nombre_Archivo, vertices=[], caras=[], aristas=[], iteraciones = 5, Periodo = [], Fase= []):
    name = Nombre_Archivo+'.ply'
    numero_vertices = len(vertices)
    numero_caras= len(caras)
    numero_aristas = len(aristas)
    contador = 1

    comentario1 = 'Periodos: '+str(Periodo)
    comentario2 = 'Fases: '+str(Fase)
    comentario3 = 'iteraciones: ' +str(iteraciones)

    with open(name, 'w') as writefile:
        #Encabezado del archivo
        writefile.write("ply\n")
        writefile.write("format ascii 1.0\n")
        writefile.write("comment "+comentario1+"\n")
        writefile.write("comment "+comentario2+"\n")
        writefile.write("comment "+comentario3+"\n")
        writefile.write("element vertex "+str(numero_vertices)+"\n")
        writefile.write("property float x\n")
        writefile.write("property float y\n")
        writefile.write("property float z\n")
        writefile.write("element face "+str(numero_caras)+"\n")
        writefile.write("property list uchar int vertex_indices\n")
        writefile.write("element edge "+str(numero_aristas)+"\n")
        writefile.write("property int32 vertex1 \n")
        writefile.write("property int32 vertex2\n")
        writefile.write("end_header\n")
        #Escribe los vértices
        for w in vertices:
            v=np.array(w)
            y = np.array2string(v,  formatter={'float_kind':lambda x: "%.10f" % x})
            y=y.replace('[', "")
            y=y.replace(']', "")
            writefile.write(""+y+"\n")

        #Escribe las aristas   
        for f in caras:
            puntos = str(len(f))
            g=np.array(f)
            y = np.array2string(g)
            y=y.replace('[', '')
            y=y.replace(']', '')
            writefile.write(puntos+' '+y+"\n")
                #Escribe las aristas   
        for f in aristas:
            g=np.array(f)
            y = np.array2string(g)
            y=y.replace('[', '')
            y=y.replace(']', '')
            writefile.write(' '+y+"\n")

    print('Escribí archivo:', name)
    return True

def ArchivoVertices(Nombre_Archivo, vertices):
    with open(Nombre_Archivo+'.txt', 'w') as writefile:
        #Escribe los vértices
        for w in vertices:
            v=np.array(w)
            y = np.array2string(v,  formatter={'float_kind':lambda x: "%.10f" % x})
            y=y.replace('[', "")
            y=y.replace(']', "")
            writefile.write(""+y+"\n")
    return True
   

def ArchivoParametrizacion(Nombre_Archivo, parametrizacion):
    with open('parametrizacion.txt', 'w') as writefile:
        #Escribe los vértices
        for w in parametrizacion:
            v=np.array(w)
            y = np.array2string(v,  formatter={'float_kind':lambda x: "%.10f" % x})
            y=y.replace('[', "")
            y=y.replace(']', "")
            writefile.write(""+y+"\n")
    return True

def ArchivoAlcances(Nombre_Archivo, Alcances):
    name = 'Alcances/'+Nombre_Archivo+'-Alcances.txt'
    with open(name, 'w') as writefile:
        #Escribe los vértices
        for w in Alcances:
            v=np.array(w)
            y = np.array2string(v,  formatter={'float_kind':lambda x: "%.10f" % x})
            y=y.replace('[', "")
            y=y.replace(']', "")
            writefile.write(""+y+"\n")
    return True
