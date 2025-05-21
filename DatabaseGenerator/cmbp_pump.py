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
import knotClasses as kc
import cmbp_submethods as methods
import numpy as np
import sympy as sp
import os
import datetime as dt

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.autoinit import context
from pycuda.compiler import SourceModule

if (par.args.pr == 'True'):
	from tqdm import tqdm
	loop='tqdm(range(@iterator), desc=" ScalarFld")'
else:
	loop='range(@iterator)'

method=par.args.m
n_knot_nodes=par.args.n
min_voxels_per_Axis=par.args.mvx
offset=par.args.bboffset

out_dir=par.args.o
knots_file=par.args.kf
log_file=par.args.log
max_blocks_per_batch=par.args.bpb
ascii_names=par.args.ascii_names

log_file=os.path.join(out_dir,log_file)

knots_tree_directory,main_dirs=dn.get_directories(knots_file,ascii_names)
if method=="automatic":
	reach_repo=par.args.reachrepo
	auto_submethod=par.args.auto_submethod

	knots_tree_directory,main_dirs=dn.get_directories(knots_file,ascii_names)
	save_polygonal_knots_dir=os.path.join(out_dir,main_dirs["polygonal_knots"])

	if (os.path.exists(save_polygonal_knots_dir)==False) or (len(os.listdir(save_polygonal_knots_dir))==0):
		#arclength_tolerance=0.01
		max_it=1.0E3
		delta_nodes=1
		cores=auto.get_opt_cores(knots_file, arclength_tolerance, max_it,delta_nodes,progress,ascii_names,out_dir)
	else:
		cores=dn.get_cores(knots_file,3)
		for j in range(len(cores)):
			route=os.path.join(save_polygonal_knots_dir,knots_tree_directory[j])
			discrete_knot=np.loadtxt(route)
			N_opt_nodes=len(discrete_knot)
			cores[j].knot=discrete_knot
			cores[j].n_nodes=N_opt_nodes
			cores[j].domain=np.linspace(0,2*np.pi,N_opt_nodes,endpoint=False)
			ff=open(route,'r')
			optimized_str=ff.readline().split(':')[1]
			#if "False" in optimized_str:
			#	opt=False
			#else:
			#	opt=True
			cores[j].optimized=eval(optimized_str.strip().rstrip())
else:
	cores=dn.get_cores(knots_file,n_knot_nodes)


input_route=os.path.join(out_dir,main_dirs['blowup_dir'])

route=os.path.join(out_dir,main_dirs['pump_dir'])

if dn.subdir_check(out_dir, main_dirs['pump_dir'])==False:
	os.mkdir(route)


t_cmbp_init=dt.datetime.now()

iterator=eval(loop.replace("@iterator","len(cores)"))
#
if method=="automatic":
	knot_data={}
	if auto_submethod=='reach':
		f=open(log_file,'r')
		data=f.read()
		f.close()
		try:
			data=data.replace('#',' ')
		except:
			pass
		d=data.split('SUB-PROCESS:')
		d1=d[1:]
		for sbp in d1:
			if "VOXEL RESOLUTION" in sbp:
				lines=[x.strip().rstrip() for x in sbp.split('\n')]
				for i in range(len(lines)):
					line=lines[i]
					if "KNOT INDEX:" in line:
						knot_index=eval(line.split(":")[1])
					if "VOXELS PER AXIS:" in line:
						voxels_per_axis=eval(line.split(":")[1])
					if "BOUNDING BOX OFFSET" in line:
						offset=2*eval(line.split(":")[1])
				knot_data[knot_index]={"min_voxels_per_Axis":voxels_per_axis,"offset":offset}
#
for i in iterator:#(len(cores)):
	t_cmbp_init=dt.datetime.now()
	core=cores[i]
	savedir=os.path.join(out_dir,main_dirs['pump_dir'],knots_tree_directory[i])
	input_subdir=os.path.join(input_route,knots_tree_directory[i])
	#print(savedir)
	if dn.subdir_check(route,knots_tree_directory[i])==False:
		os.mkdir(savedir)
	filename=os.path.join(savedir,knots_tree_directory[i])+".txt"

	directories=[os.path.join(input_subdir,x) for x in os.listdir(input_subdir)]

	if method=="minimal" or method=="manual":
		field=methods.minimal_field(core,directories,min_voxels_per_Axis,offset,max_blocks_per_batch)
	elif method=="sinusoidal":
		field=methods.minimal_field(core,directories,min_voxels_per_Axis,offset,max_blocks_per_batch)
	elif method=="automatic":
		arclength_tolerance=par.args.arclength_tolerance
		
		if auto_submethod=='reach':
			min_voxels_per_Axis=knot_data[i]["min_voxels_per_Axis"]
			offset=knot_data[i]["offset"]
			#field=methods.minimal_field(core,directories,min_voxels_per_Axis,offset,max_blocks_per_batch)
			field=methods.automatic_field(core,directories,offset,max_blocks_per_batch)
	else:
		print("Unknown method - cmbp pump")
	#print(field)
	pass
	methods.array3D_to_file(field,filename)

	t_stamp=dt.datetime.now()
	delta=t_stamp-t_cmbp_init

	stamp_dict={"Knot index":i,
				"Knot":core.label,
				"Knot directory":knots_tree_directory[i],
				"Grid size":field.shape}
	methods.savelog('pump (scalar field gen)',log_file,delta.total_seconds(),stamp_dict)