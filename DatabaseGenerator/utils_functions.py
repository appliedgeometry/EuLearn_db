# Created by Applied Geometry Laboratory (https://github.com/appliedgeometry)
# for "EuLearn: A 3D database for learning Euler characteristics" Project, 2025.
#
#
# https://huggingface.co/datasets/appliedgeometry/EuLearn
# https://github.com/appliedgeometry/EuLearn_db
#
# GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007

from scipy.integrate import quad
import numpy as np

def arclength(nx,ny,nz, fx,fy,fz, ktype=str('lis')):
	if ktype == str('lis'):
		a = lambda t: np.sqrt((nx*np.sin(nx*t + fx))**2 + (ny*np.sin(ny*t + fy))**2 + (nz*np.sin(nz*t + fz))**2)
	elif ktype == str('fib'):
		a = lambda t: np.sqrt((nx*np.sin(nx*t + fx))**2 + (ny*np.sin(ny*t + fy))**2 + 
			(0.5*nz*np.sin(nz*t + fz) + 0.5*ny*np.cos(ny*t + fy))**2)
	elif ktype==str('trefoil'):
		a=lambda t: np.sqrt(  (np.cos(t)+4*np.cos(2*t))**2+ 9*np.cos(3*t)**2  +(-np.sin(t)+4*np.sin(2*t))**2 )
	elif ktype=="eight":
		a=lambda t: np.sqrt( (-3*(np.cos(2*t) + 2)*np.sin(3*t) - 2*np.sin(2*t)*np.cos(3*t))**2+(3*(np.cos(2*t) + 2)*np.cos(3*t) - 2*np.sin(2*t)*np.sin(3*t))**2+16*np.cos(4*t)**2)
		#np.sqrt( -3*np.sin(3*t)*(2+np.cos(2*t))+np.cos(3*t)*(-2*np.sin(2*t)) +  3*np.cos(3*t)*(2+np.cos(2*t))+np.sin(3*t)*(-2*np.sin(2*t)) +4*np.cos(4*t) )

	else:
		pass
	#print('The length of the', ktype, 'knot is:')
	arc = ( round( quad(a,0,0.5*np.pi)[0], 2) + round( quad(a,0.5*np.pi,np.pi)[0], 2) 
		+ round( quad(a,np.pi,1.5*np.pi)[0], 2) + round( quad(a,1.5*np.pi,2*np.pi)[0], 2) )
	return arc
	
	
	

	