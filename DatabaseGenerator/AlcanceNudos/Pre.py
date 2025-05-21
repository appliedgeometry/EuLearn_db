# Me dice la cantidad de ternas que hay en el archivo 'Ternas.txt'
fichero = open('AlcanceInicial/Ternas/Ternas.txt')
lineas = fichero.readlines()
print('\nNo. of Lines:', len(lineas)-5, '\n')
print('1st line:', lineas[5])
print('Last line:', lineas[-1])