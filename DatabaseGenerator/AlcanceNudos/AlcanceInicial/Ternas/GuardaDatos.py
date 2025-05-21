
# Genera .csv con frecuencias, fases, indices de barras y alcance mínimo

from datetime import datetime

def GuardaDatos(A,B,C,D,E):

  name = 'Datos.csv'

  date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

  with open(name, 'w') as writefile:
    total = str(len(A))
    # writefile.write('###############################################' + '\n')
    # writefile.write('# ' + total + '6-tuplas de frecuencias con fases' + '\n')
    # writefile.write('# ' + date + '\n')
    # writefile.write('###############################################' + '\n')
    writefile.write('knot_type, '+
                    'nx,ny,nz, '+ 
                    'phi_x,phi_y,phi_z, '+
                    'min_reach,max_reach, '+
		    'bars'+'\n')
    for i in range(len(A)):
      a = A[i]
      b = B[i]
      c = C[i]
      d = D[i]
      e = E[i]
      writefile.write('lissajous, '+
                      str(a[0])+','+str(a[1])+','+str(a[2])+', '+ 
                      str(b[0])+','+str(b[1])+','+str(b[2])+', '+
                      str(c)+','+str(d)+', '+'[]'+'\n')

  print("Se ha creado el archivo:", name, 'con', total, 'nudos', date)
