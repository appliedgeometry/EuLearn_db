# Created by Applied Geometry Laboratory (https://github.com/appliedgeometry)
# for "EuLearn: A 3D database for learning Euler characteristics" Project, 2025.
#
#
# https://huggingface.co/datasets/appliedgeometry/EuLearn
# https://github.com/appliedgeometry/EuLearn_db
#
# GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007

import knotClasses as kc
import os
import numpy as np


def replace_chars(text_str, ascii_names):
	if ascii_names=="False":
		replacement_dict={"/":"÷",
					"np.pi":"п",
					"*":"×"}
	else:
		replacement_dict={"/":"",
					"np.pi":"pi",
					"*":""}
	result=text_str
	for key in replacement_dict.keys():
		if key in text_str:
			result=result.replace(key,replacement_dict[key])
		else:
			pass
	#new_symbol="-"
	# ÷
	# ×
	# п
	return result
	
	
def subdir_check(host_dir,subdir):
	x=True
	if subdir not in os.listdir(host_dir):
		x=False
	return(x)


def get_directories(knots_file,ascii_names):
	f=open(knots_file,'r')
	knot_list=f.readlines()
	knot_list=[line for line in knot_list if ',' in line]
	f.close()


	knots_tree_directory=[]

	for i in range(len(knot_list)):
		line=knot_list[i]
		index1=line.find(",")
		index2=line.find("[")
		
		knot_type=line[:index1]
		bars_str=line[index2:].rstrip()
		bars=eval(bars_str)
		knot_params_str=replace_chars(line[index1+1:index2-1],ascii_names).split(",")
		
		
		if knot_type in['lissajous', 'fibonacci' ]:
			nxnynz_str="_"+knot_params_str[0].rstrip().strip()+","+knot_params_str[1].rstrip().strip()+","+knot_params_str[2].rstrip().strip()
			fxfyfz_str=','+knot_params_str[3].rstrip().strip()+","+knot_params_str[4].rstrip().strip()+","+knot_params_str[5].rstrip().strip()
		else:
			nxnynz_str=''
			fxfyfz_str=""
		parameters_str=nxnynz_str+fxfyfz_str
		#knot_params=[eval(i) for i in knot_params_str]
		knot_dir=str(i)+"_"+knot_type+parameters_str+"_b"+str(len(bars))
		knots_tree_directory.append(knot_dir)
	# Output subdirectories by process
	main_dirs={"blowup_dir":"Max_BlowUP_Computation",
				"pump_dir":"mq_Fields",
				"marchingQ_dir":"STL",
				"polygonal_knots":"Optimal-discrete_knots"}
	return(knots_tree_directory,main_dirs)


def get_cores(knots_file,n_knot_nodes):
	f=open(knots_file,'r')
	knot_list=f.readlines()
	f.close()


	knots_tree_directory=[]
	cores=[]
	for i in range(len(knot_list)):
		line=knot_list[i]
		index1=line.find(",")
		index2=line.find("[")
		
		knot_type=line[:index1]
		bars_str=line[index2:].rstrip()
		bars=eval(bars_str)
		knot_params_str=line[index1+1:index2-1].split(",")

		knot_params=[eval(i) for i in knot_params_str]

		t=np.linspace(0,2*np.pi,n_knot_nodes,endpoint=False)

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

		knot_=kc.parametrization(knot_type_1.parametrization,knot_type_1.symbolic,t,knot_type_1.label)
		knot_core=kc.core(knot_)
		
		knot_core.add_bars(bars)
	
		cores.append(knot_core)
	return(cores)
#print(knots_tree_directory)



