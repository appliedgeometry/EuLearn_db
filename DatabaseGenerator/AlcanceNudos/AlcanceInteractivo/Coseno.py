import numpy as np

# cos(nt+f)
def Evalua(t, n, f):
	return np.cos((n*t)+f)

# -n sin (nt+f)
def PrimeraDerivada(t,n,f):
	return (-n)*np.sin((n*t)+f)

# -n^2 sin (nt+f)
def SegundaDerivada(t, n, f):
	return (-(n*n))*np.cos((n*t)+f)