# Version 21 Febrero 2023
import numpy as np
import random
import sys

#Lee el archivo Fases.txt
#Se debe aumentar 5 por el encabezado
def LeeFases(Indices):
  Ternas = []
  fichero = open('AlcanceInicial/Ternas/Fases.txt')
  lineas = fichero.readlines()
  for i in Indices:
    terna_cadena = lineas[i+5]
    terna_cadena = terna_cadena.replace('\n', '')
    terna_cadena = terna_cadena.replace('[', '')
    terna_cadena = terna_cadena.replace(']', '')
    terna_arreglo = terna_cadena.split()
    terna = [int(elemento) for elemento in terna_arreglo]
    Ternas.append(terna)
  return Ternas

#Lee el archivo Ternas.txt
#Se debe aumentar 5 por el encabezado
def LeeTernas(Indices):
  Ternas = []
  fichero = open('AlcanceInicial/Ternas/Ternas1.txt')
  #fichero = open('Ternas/Ternas.txt')
  lineas = fichero.readlines()
  for i in Indices:
    terna_cadena = lineas[i-1] #[i+5]
    terna_cadena = terna_cadena.replace('\n', '')
    terna_cadena = terna_cadena.replace('[', '')
    terna_cadena = terna_cadena.replace(']', '')
    terna_arreglo = terna_cadena.split()
    terna = [int(elemento) for elemento in terna_arreglo]
    Ternas.append(terna)
  return Ternas

def EscogeTernas(cantidad_de_nudos):
  Ternas = []
  fichero = open('AlcanceMinimo/Ternas/Ternas.txt')
  lineas = fichero.readlines()
  cantidad_de_ternas = len(lineas)
  baraja = [i for i in range(1, cantidad_de_ternas+1)]
  TernasAEscoger = random.sample(baraja, cantidad_de_nudos)
  TernasAEscoger.sort()
  return TernasAEscoger


def ArchivoPLY(Nombre_Archivo, vertices=[], caras=[], aristas=[], terna=[], fases=[], alcance=0):
  name = Nombre_Archivo+'.ply'
  comentario1 = 'Periodos: '+str(terna)
  comentario2 = 'Fases: '+str(fases)
  comentario3 = 'Alcance: '+str(alcance)
  numero_vertices = len(vertices)
  numero_caras= len(caras)
  numero_aristas = len(aristas)
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
  return name

def GeneraPLYdeNudo(carpeta, Nudo, Terna, fase, alcance, iteraciones, sirve, motivo):
  intervalos = len( Nudo )
  Vertices = Nudo
  Aristas = []
  for j in range( intervalos):
    Aristas.append([j%intervalos, (j+1)%intervalos])
  nombre_x = "(" + str(Terna[0]) + "t+" + str(fase[0]) + ')'
  nombre_y = "(" + str(Terna[1]) + "t+" + str(fase[1]) + ')'
  nombre_z = "(" + str(Terna[2]) + "t+" + str(fase[2]) + ')'
  # info_nudo = nombre_x + nombre_y + nombre_z
  info_nudo = str(Terna[0]) +','+str(Terna[1])+','+str(Terna[2])+ '-' +str(round(fase[0],2))+','+str(round(fase[1],2))+','+str(round(fase[2],2))
  if sirve:
    etiqueta = ''
  else:
    etiqueta = '-NoSirve' + motivo
  info_rutina = '_' + str(iteraciones) +'-'+ str(intervalos) + etiqueta
  name = carpeta + info_nudo + info_rutina
  Ruta = ArchivoPLY(name, vertices=Vertices, aristas=Aristas, terna=Terna, fases=fase, alcance=alcance)
  return Ruta

def EscribeAlcancesPuntualesTXT(Alcances, Terna, fase, Nudo, iteraciones):
  npts = len(Nudo)
  nombre_x = "cos(" + str(Terna[0]) + "t+" + str(round(fase[0],4)) + ')_'
  nombre_y = "cos(" + str(Terna[1]) + "t+" + str(round(fase[1],4)) + ')_'
  nombre_z = "cos(" + str(Terna[2]) + "t+" + str(round(fase[2],4)) + ')'
  # nudo = nombre_x + nombre_y + nombre_z
  nudo = str(Terna[0])+','+str(Terna[1])+','+str(Terna[2])+'-'+str(round(fase[0],4))+','+str(round(fase[1],4))+','+str(round(fase[2],4))
  name = 'Nudos/AlcancesIniciales/Alcances/'+'Alcances_'+nudo+'_'+str(npts)+'pts'+'_'+str(iteraciones)+'its'+'.txt'
  with open(name, 'w') as writefile:
    writefile.write('###############################################'+"\n")
    writefile.write("# Alcances iniciales"+"\n")
    writefile.write('# Version 1'+"\n")
    writefile.write('# '+"\n")
    writefile.write('###############################################'+"\n")
    for t in Alcances:
      y = np.array2string(t)
      writefile.write(""+y+"\n")
  return name


def EscribeAumentosTXT(Aumentos, Terna, fase):
  nombre_x = "cos(" + str(Terna[0]) + "t+" + str(round(fase[0],4)) + ')_'
  nombre_y = "cos(" + str(Terna[1]) + "t+" + str(round(fase[1],4)) + ')_'
  nombre_z = "cos(" + str(Terna[2]) + "t+" + str(round(fase[2],4)) + ')'
  # nudo = nombre_x + nombre_y + nombre_z
  nudo = str(Terna[0])+','+str(Terna[1])+','+str(Terna[2])+'-'+str(round(fase[0],4))+','+str(round(fase[1],4))+','+str(round(fase[2],4))
  name = 'Nudos/AlcancesIniciales/Aumentos/'+'Aumentos-'+nudo+'.txt'
  with open(name, 'w') as writefile:
    writefile.write('###############################################'+"\n")
    writefile.write("# Aumentos iniciales"+name+"\n")
    writefile.write('# Version 1'+"\n")
    writefile.write('# '+"\n")
    writefile.write('###############################################'+"\n")
    for t in Aumentos:
      y = str(t)
      writefile.write(""+y+"\n")
  return name