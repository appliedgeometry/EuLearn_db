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
import auto_knotnodes as auto
import numpy as np
import sympy as sp
import os
import datetime as dt
import sys 


if (par.args.pr == 'True'):
	from tqdm import tqdm
	loop='tqdm(range(@iterator), desc=" MaxBlowUP")'
else:
	loop='range(@iterator)'


method=par.args.m
n_knot_nodes=par.args.n
min_voxels_per_Axis=par.args.mvx
offset=par.args.bboffset
progress=par.args.pr
out_dir=par.args.o
knots_file=par.args.kf
log_file=par.args.log
max_blocks_per_batch=par.args.bpb
auto_voxel_buffer=15#
#

ascii_names=par.args.ascii_names

log_file=os.path.join(out_dir,log_file)

r_neighborhood=par.args.r_neigh # Only used if method==manual


knots_tree_directory,main_dirs=dn.get_directories(knots_file,ascii_names)

f=open(knots_file,'r')
knot_list=f.readlines()
f.close()

# 
pi=np.pi
if method in ["minimal","sinusoidal"]:
	voxel_activation_scale=par.args.vx_ActDistSc # formerly: np.sqrt(3.0) # <- max voxel diagonal 
#	

if method=="automatic":
	arclength_tolerance=par.args.arclength_tolerance
	reach_repo=par.args.reachrepo
	auto_submethod=par.args.auto_submethod
	knots_tree_directory,main_dirs=dn.get_directories(knots_file,ascii_names)
	save_polygonal_knots_dir=os.path.join(out_dir,main_dirs["polygonal_knots"])

	if (os.path.exists(save_polygonal_knots_dir)==False) or (len(os.listdir(save_polygonal_knots_dir))==0):
		#arclength_tolerance=0.01
		max_it=200 # 1.0E3
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

			cores[j].optimized=eval(optimized_str.strip().rstrip())
else:
	cores=dn.get_cores(knots_file,n_knot_nodes)


route=os.path.join(out_dir,main_dirs['blowup_dir'])

if dn.subdir_check(out_dir, main_dirs['blowup_dir'])==False:
	os.mkdir(route)


iterator=eval(loop.replace("@iterator","len(cores)"))


original_print_output=sys.stdout
for i in iterator:#range(len(cores)):
	t_cmbp_init=dt.datetime.now()
	core=cores[i]
	savedir=os.path.join(out_dir,main_dirs['blowup_dir'],knots_tree_directory[i])
	if dn.subdir_check(route,knots_tree_directory[i])==False:
		os.mkdir(savedir)

	if method=='minimal':
		#methods.BB_lenght(core)
		infx,supx,infy,supy,infz,supz,res,used_voxels=methods.grid_info(core,min_voxels_per_Axis,offset)
		for j in range(1+len(core.bars)):
			if j==0:
				filename=os.path.join(savedir,knots_tree_directory[i])
			else:
				filename=os.path.join(savedir,"bar_"+str(j))
			#array=0.5*res*np.sqrt(3)*np.ones(n_knot_nodes)
			#array=0.5*res*voxel_activation_scale*np.ones(n_knot_nodes)
			array=res*voxel_activation_scale*np.ones(n_knot_nodes)
			methods.array1D_to_file(array,filename+".txt")
		#pass
		#mx_blowup=methods.minimal(core,n_knot_nodes,min_voxels_per_Axis,offset)
	elif method=="manual":
		delta=r_neighborhood
		infx,supx,infy,supy,infz,supz,res,used_voxels=methods.grid_info(core,min_voxels_per_Axis,offset)
		for j in range(1+len(core.bars)):
			if j==0:
				filename=os.path.join(savedir,knots_tree_directory[i])
			else:
				filename=os.path.join(savedir,"bar_"+str(j))
			#array=delta*np.sqrt(3.0)*np.ones(n_knot_nodes)
			array=delta*np.ones(n_knot_nodes)
			methods.array1D_to_file(array,filename+".txt")
	elif method=="sinusoidal":
		sine_const=par.args.sine_const
		sine_amp=par.args.sine_amp
		sine_freq=par.args.sine_freq
		sine_phase=par.args.sine_phase
		#print(sine_phase)
		sine_phase=eval(sine_phase)
		#print(sine_phase)
		def sine_Deformer(cte,amp,fq,ph,N_nodes):
			height=np.linspace(0,2*np.pi,N_nodes)
			return cte*np.ones(N_nodes)+amp*np.cos(fq*height+ph)
		#
		epsilon=1
		infx,supx,infy,supy,infz,supz,res,used_voxels=methods.grid_info(core,min_voxels_per_Axis,offset)
		for j in range(1+len(core.bars)):
			if j==0:
				filename=os.path.join(savedir,knots_tree_directory[i])
				#array=res*np.array([1+epsilon*np.sin(th) for th in np.linspace(0,np.pi,n_knot_nodes)])
				array=res*sine_Deformer(sine_const,sine_amp,sine_freq,sine_phase,n_knot_nodes)
				methods.array1D_to_file(array,filename+".txt")
			else:
				filename=os.path.join(savedir,"bar_"+str(j))
				#array=0.5*res*np.sqrt(3)*np.ones(n_knot_nodes)
				#array=0.5*res*voxel_activation_scale*np.ones(n_knot_nodes)
				array=res*voxel_activation_scale*np.ones(n_knot_nodes)
				methods.array1D_to_file(array,filename+".txt")
	elif method=="automatic":
		n_knot_nodes=len(cores[i].knot)
		if auto_submethod=='reach':
			if reach_repo not in sys.path:
				routes=[reach_repo]+[os.path.join(reach_repo,route) for route in ["AlcanceInicial","AlcanceInteractivo"]]
				sys.path=sys.path+[directory for directory in routes if directory not in sys.path]
			else:
				pass
			import BuscaAlcanceMinimo
			import ConstruyeNudo
			from AlcancePorEsferas import acotaAlcancePorEsferas as acotaAlcancePorEsferas
			intervalos= len(core.knot)
			
			iteraciones=int(par.args.reach_iterations) #5
			core.reach_iterations=str(iteraciones)
			umbral=float(par.args.reach_threshold) #0.001

			Nudo=core.knot
			if i==0:
				fifo_type='w'
			else:
				fifo_type='a'
			reach_file=open(os.path.join(out_dir,"reach_file.txt"),fifo_type)
			sys.stdout=reach_file
			print(core.label)
			#TrabajaNudo(Terna, fases, intervalos, iteraciones, umbral)
			Nudo=core.knot
			Alcances, Aumentos, autointerseccion = acotaAlcancePorEsferas(Nudo, intervalos, iteraciones)
			sys.stdout=original_print_output
			tubular=Alcances#compute_reach(core)
			for j in range(1+len(core.bars)):
				if j==0:
					filename=os.path.join(savedir,knots_tree_directory[i])
					array=tubular
					methods.array1D_to_file(array,filename+".txt")
				else:
					pass
					#filename=os.path.join(savedir,"bar_"+str(j))
			bb_length=methods.BB_length(core)
			minimal_neighborhood,n_voxels=methods.minimal_neighborhood_Nvoxels(tubular,bb_length)
			minimal_neighborhood=min(tubular)
			infx,supx,infy,supy,infz,supz,res,min_voxels_per_Axis=methods.grid_info(core,n_voxels,offset)
		else:
			pass
	else:
		pass

	t_stamp=dt.datetime.now()
	delta=t_stamp-t_cmbp_init

	stamp_dict={"Knot index":i,
				"Knot":core.label,
				"Bars":len(core.bars),
				"Save Dir":savedir,
				"knot_nodes":n_knot_nodes,
				"Arc Length":core.parametrization.arclength,
				"Voxels per Axis":min_voxels_per_Axis,
				"Bounding Box Offset [vx]":offset,
				"Bounding Box origin":str(infx)+","+str(infy)+","+str(infz),
				"Voxel Resolution (size)":res,
				"MINIMAL NEIGHBORHOOD":np.min(array),
				"MAX NEIGHBORHOOD":np.max(array),
				"AVERAGE NEIGHBORHOOD":np.mean(array),
				"Method":method}
	if method=="automatic":
		stamp_dict["OPTIMIZED NUMBER OF NODES"]=core.optimized
		stamp_dict["REACH ITERATIONS"]=core.reach_iterations
	if method=="sinusoidal":
		stamp_dict["MetaParam Deformer (constant shift)"]=sine_const
		stamp_dict["MetaParam Deformer (amplitude)"]=sine_amp
		stamp_dict["MetaParam Deformer (frequency)"]=sine_freq
		stamp_dict["MetaParam Deformer (phase)"]=par.args.sine_phase #sine_phase
	methods.savelog('compute_max_blow_up',log_file,delta.total_seconds(),stamp_dict)

