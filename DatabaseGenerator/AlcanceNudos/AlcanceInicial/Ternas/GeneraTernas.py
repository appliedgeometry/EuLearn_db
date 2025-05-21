#Version 12 de Febrero 2023
#Hace ternas ordenadas de números primos relativos
import numpy as np
from itertools import product as cartesian
from math import gcd as gcd
from datetime import datetime

date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

A=[]
for i in range(2,12):
  A.append(i)
Ternas = []
for element in cartesian(A,A,A):
  if element[0] <= element[1] <= element[2]:
    #Califica si son primos relativos
    if gcd(element[0], element[1]) == gcd(element[1], element[2]) == gcd(element[2], element[0]) == 1:
      Ternas.append(np.array(element))

name = 'Ternas.txt'
with open(name, 'w') as writefile:
  cantidad = str(len(Ternas))
  writefile.write('###############################################'+"\n")
  writefile.write("# "+cantidad+' ternas para nudos'+"\n")
  writefile.write('# Version 1'+"\n")
  writefile.write('# '+date+"\n")
  writefile.write('###############################################'+"\n")
  # writefile.write('nx'+','+'ny'+','+'nz'+'\n')
  for t in Ternas:
    x = np.array2string(t[0])
    y = np.array2string(t[1])
    z = np.array2string(t[2])
    w = np.array2string(t)
    writefile.write( w + '\n')
    # writefile.write(x +','+ y +','+ z +'\n')

print("Se ha creado el archivo:", name)