# Created by Applied Geometry Laboratory (https://github.com/appliedgeometry)
# for "EuLearn: A 3D database for learning Euler characteristics" Project, 2025.
#
#
# https://huggingface.co/datasets/appliedgeometry/EuLearn
# https://github.com/appliedgeometry/EuLearn_db
#
# GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007

import numpy as np
import sympy as sp
from sympy.vector import CoordSys3D
import sys
import utils_functions as utf
import symbolic_Cfunctions as scf

class lissajous:
	def __init__(self,nx,ny,nz,phi_x,phi_y,phi_z):
		def f(t):
			x=np.cos(nx*t+phi_x)
			y=np.cos(ny*t+phi_y)
			z=np.cos(nz*t+phi_z)
			return np.array((x,y,z))
		s=sp.Symbol('s')
		x_sp=sp.cos(nx*s+phi_x)
		y_sp=sp.cos(ny*s+phi_y)
		z_sp=sp.cos(nz*s+phi_z)
		
		E=CoordSys3D('E')
		param=x_sp*E.i+y_sp*E.j+z_sp*E.k
		"""
		tan_sp_x=sp.diff(x_sp,s)
		tan_sp_y=sp.diff(y_sp,s)
		tan_sp_z=sp.diff(z_sp,s)
		
		def tangent(t):
			tx=sp.lambdify(s,tan_sp_x)		
			ty=sp.lambdify(s,tan_sp_y)		
			tz=sp.lambdify(s,tan_sp_z)		
			return np.array( (tx(t),ty(t),tz(t)) )
		self.tangent=tangent	
		"""	
		self.parametrization=f
		self.parametrization.arclength=utf.arclength(nx,ny,nz,phi_x,phi_y,phi_z,'lis')
		self.symbolic=param
		self.label="lissajous_"+str(nx)+","+str(ny)+","+str(nz)+","+str(phi_x)+","+str(phi_y)+","+str(phi_z)+"_"
		c=scf.lissajous_c.replace("@knx",str(nx))
		c=c.replace("@kny",str(ny))
		c=c.replace("@knz",str(nz))
		c=c.replace("@phi_x",str(phi_x))
		c=c.replace("@phi_y",str(phi_y))
		c=c.replace("@phi_z",str(phi_z))
		self.parametrization.c=c

class fibonacci:
	def __init__(self,nx,ny,nz,phi_x,phi_y,phi_z):
		A=0.5
		def f(t):
			x = np.cos(nx*t + phi_x)
			y = np.cos(ny*t + phi_y)
			z = A*np.cos(nz*t + phi_z) + A*np.sin(ny*t + phi_y)
			return np.array((x,y,z))
			
		s=sp.Symbol('s')
		x_sp=sp.cos(nx*s+phi_x)
		y_sp=sp.cos(ny*s+phi_y)
		z_sp=A*sp.cos(nz*s+phi_z)+ A*sp.sin(ny*s + phi_y)
		
		E=CoordSys3D('E')
		param=x_sp*E.i+y_sp*E.j+z_sp*E.k
		
		self.parametrization = f
		self.parametrization.arclength=utf.arclength(nx,ny,nz,phi_x,phi_y,phi_z,'fib')
		self.symbolic=param
		self.label = "fibonacci_"+str(nx)+","+str(ny)+","+str(nz)+","+str(phi_x)+","+str(phi_y)+","+str(phi_z)+","+str(A)+"_"
		c=scf.fibonacci_c.replace("@knx",str(nx))
		c=c.replace("@kny",str(ny))
		c=c.replace("@knz",str(nz))
		c=c.replace("@phi_x",str(phi_x))
		c=c.replace("@phi_y",str(phi_y))
		c=c.replace("@phi_z",str(phi_z))
		self.parametrization.c=c

class trefoil:
	def __init__(self):
		def f(t):
			x=np.sin(t)+2*np.sin(t*2)
			y=np.cos(t)-2*np.cos(t*2)
			z=-np.sin(3*t)
			return np.array((x,y,z))
		s=sp.Symbol('s')
		x_sp=sp.sin(s)+2*sp.sin(s*2)
		y_sp=sp.cos(s)-2*sp.cos(s*2)
		z_sp=-sp.sin(3*s)
		
		E=CoordSys3D('E')
		param=x_sp*E.i+y_sp*E.j+z_sp*E.k
		self.symbolic=param
		
		self.parametrization=f
		self.parametrization.arclength=utf.arclength(0,0,0,0,0,0,'trefoil')
		self.label="trefoil"
		self.parametrization.c=scf.trefoil_c

class circle:
	def __init__(self):
		def f(t):
			x=np.cos(t)
			y=np.sin(t)
			z=0
			return np.array((x,y,z))
		
		s=sp.Symbol('s')
		x_sp=sp.cos(s)
		y_sp=sp.sin(s)
		z_sp=0
		
		E=CoordSys3D('E')
		param=x_sp*E.i+y_sp*E.j+z_sp*E.k
		self.symbolic=param
		
		self.parametrization=f
		self.parametrization.arclength=2*np.pi
		self.label="circle"
		self.parametrization.c=scf.circle_c

class eight:
    def __init__(self):
        def f(t):
            x = (2 + np.cos(2*t))*np.cos(3*t)
            y = (2 + np.cos(2*t))*np.sin(3*t)
            z = np.sin(4*t)
            return np.array((x,y,z))
        s = sp.Symbol('s')
        x_sp = (2 + sp.cos(2*s))*sp.cos(3*s)
        y_sp = (2 + sp.cos(2*s))*sp.sin(3*s)
        z_sp = sp.sin(4*s)
        
        E=CoordSys3D('E')
        param=x_sp*E.i+y_sp*E.j+z_sp*E.k
        self.symbolic=param
        
        self.parametrization=f
        self.parametrization.arclength=utf.arclength(0,0,0,0,0,0,'eight')
        self.label="eight"
        self.parametrization.c=scf.eight_c
        
class core:
	def __init__(self,parametrization):
		self.parametrization=parametrization # lambda function
		n_knot_nodes=parametrization.n_nodes
		self.n_knot_nodes=n_knot_nodes
		
		point_list=[parametrization.function(s) for s in parametrization.domain]
		
		knot=np.zeros((n_knot_nodes,3)) 
		for i in range(n_knot_nodes):
			knot[i][0]=point_list[i][0]
			knot[i][1]=point_list[i][1]
			knot[i][2]=point_list[i][2]
		self.knot=knot
		self.label=parametrization.label
		#self.symbolic=parametrization.symbolic
		
	def add_bars(self,list_of_bars):
		bars=[]
		for bar in list_of_bars:
			t_ini,t_fin=bar
			SSn=self.parametrization.n_nodes
			#print(SSn)
			p_ini=self.parametrization.function(t_ini)
			p_fin=self.parametrization.function(t_fin)
			
			v_ssn=np.linspace(-1,1,SSn)
			bar_array=np.zeros((SSn,3))
			
			for ss in range(len(v_ssn)):
				n_point=0.5*(1-v_ssn[ss])*p_ini+0.5*(1+v_ssn[ss])*p_fin
				bar_array[ss][0]=n_point[0]
				bar_array[ss][1]=n_point[1]
				bar_array[ss][2]=n_point[2]
			bars.append(bar_array)
		self.bars=bars
		
class parametrization:
	def __init__(self,function, symbolic,domain,label):
		self.function=function
		self.domain=domain
		self.n_nodes=len(domain)#n_nodes
		self.label=label
		self.symbolic=symbolic
		self.arclength=function.arclength
		self.c=function.c
	
