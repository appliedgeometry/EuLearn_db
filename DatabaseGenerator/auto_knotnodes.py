# Created by Applied Geometry Laboratory (https://github.com/appliedgeometry)
# for "EuLearn: A 3D database for learning Euler characteristics" Project, 2025.
#
#
# https://huggingface.co/datasets/appliedgeometry/EuLearn
# https://github.com/appliedgeometry/EuLearn_db
#
# GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007

import par_parameters as par
import par_dirnames as dn
import utils_functions
import numpy as np
import knotClasses as kc
import cmbp_submethods as methods
import os

def get_polygonal_arclength(core):
	knot=core.knot
	length=sum([np.linalg.norm(knot[i]-knot[(i+1)%len(knot)]) for i in range(len(knot))])
	return length


def get_opt_cores(knots_file, tolerance, max_it,delta_nodes,progress,ascii_names,out_dir):
	
	knots_tree_directory,main_dirs=dn.get_directories(knots_file,ascii_names)
	save_polygonal_knots_dir=os.path.join(out_dir,main_dirs["polygonal_knots"])

	if os.path.exists(save_polygonal_knots_dir)==False:
		os.mkdir(save_polygonal_knots_dir)
	if (progress == 'True'):
		from tqdm import tqdm
		loop_for_automatic_node_number='tqdm(range(@iterator_nodes), desc=" OptimNodes")'
	else:
		loop_for_automatic_node_number='range(@iterator_nodes)'

	f=open(knots_file,'r')
	knot_list=f.readlines()
	knot_list=[line for line in knot_list if ',' in line]
	f.close()

	cores=[]
	iterator_nodes=eval(loop_for_automatic_node_number.replace("@iterator_nodes","len(knot_list)"))
	for i in iterator_nodes:#range(len(knot_list)):
		line=knot_list[i]
		index1=line.find(",")
		index2=line.find("[")
		
		knot_type=line[:index1]
		bars_str=line[index2:].rstrip()
		bars=eval(bars_str)
		knot_params_str=line[index1+1:index2-1].split(",")
		knot_params=[eval(i) for i in knot_params_str]

		if knot_type=="lissajous":
			nx,ny,nz,phi_x,phi_y,phi_z=knot_params
			knot_type_1=kc.lissajous(nx,ny,nz,phi_x,phi_y,phi_z)
		elif knot_type=="fibonacci":
			nx,ny,nz,phi_x,phi_y,phi_z=knot_params
			knot_type_1=kc.fibonacci(nx,ny,nz,phi_x,phi_y,phi_z)
		elif knot_type=="trefoil":
			knot_type_1=kc.trefoil()
		elif knot_type=="circle":
			knot_type_1=kc.circle()
		elif knot_type=="eight":
			knot_type_1=kc.eight()
		else:
			break
		n_knot_nodes=101
		t=np.linspace(0,2*np.pi,n_knot_nodes,endpoint=False)
		knot_=kc.parametrization(knot_type_1.parametrization,knot_type_1.symbolic,t,knot_type_1.label)
		knot_core=kc.core(knot_)
		knot_core.add_bars(bars)
		
		arclength_integral= knot_core.parametrization.arclength
		#arclength_prev=-1.0E5
		knot_optimized=False
		optimized_str='False'
		iteration=0
		

		while knot_optimized==False and iteration<max_it:
			t=np.linspace(0,2*np.pi,n_knot_nodes,endpoint=False)


			knot_=kc.parametrization(knot_type_1.parametrization,knot_type_1.symbolic,t,knot_type_1.label)
			knot_core=kc.core(knot_)
		
			knot_core.add_bars(bars)

			arclength=get_polygonal_arclength(knot_core)
			#arclength=knot_core.parametrization.arclength
			
			iteration+=1
			if abs(arclength_integral - arclength)<=tolerance:
				
				#if buffer>=it_buffer:
				knot_optimized=True
				optimized_str='True'
			#arclength_prev=arclength
			n_knot_nodes+=delta_nodes
			#arclength_prev=arclength
		knot_core.optimized=knot_optimized
		filename=os.path.join(save_polygonal_knots_dir,knots_tree_directory[i])
		methods.knot_to_file(knot_core.knot,filename,optimized_str)
		cores.append(knot_core)
	return cores
	