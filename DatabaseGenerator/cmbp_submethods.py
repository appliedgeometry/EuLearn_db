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
import os

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.autoinit import context
from pycuda.compiler import SourceModule

def knot_to_file(array,filename,optimized_str):
	f=open(filename,'w')
	f.write("# optimized: "+optimized_str+"\n")
	for i in range(len(array)):
		f.write(str(array[i][0])+" "+str(array[i][1])+" "+str(array[i][2])+"\n")
	f.close()
	return

def array1D_to_file(array,filename):
	f=open(filename,'w')
	for i in range(len(array)):
		f.write(str(array[i])+"\n")
	f.close()
	return

def array3D_to_file(array, filename):
	with open(filename, 'w') as out:
		out.write("#Grid_size: "+str(array.shape)+"\n")
		for sub_array in array:
			np.savetxt(out, sub_array.astype(int), fmt='%i')
			out.write('\n')
	#print("done")
	return

def savelog(process,log_file,seconds,stamp_dict):
	#f=open(log_file,'a')
	header="#"*50
	dic_str='SUB-PROCESS:   '+process+"\n"+"SUB-time [s]: "+str(seconds)+"\n"
	for key, value in stamp_dict.items():
		dic_str=dic_str+" "+key.upper()+":  "+str(value)+"\n"
	#with open(log_file, 'a') as out:
	fifo_type='a'
	if not os.path.exists( log_file ):
		fifo_type='w'
	with open(log_file, fifo_type) as out:
		out.write(header + '\n')
		out.write(dic_str + '\n')
	#f.close()
	return

#
def BB_length(core): # Bounding Box
			knot_curve=core.knot
			xm=min([p[0] for p in knot_curve])
			xM=max([p[0] for p in knot_curve])
			ym=min([p[1] for p in knot_curve])
			yM=max([p[1] for p in knot_curve])
			zm=min([p[2] for p in knot_curve])
			zM=max([p[2] for p in knot_curve])
			return max([xM-xm,yM-ym,zM-zm])

def grid_info(core,n_voxels,vox_bb_offset): # Bounding Box
			knot_curve=core.knot

			xm=min([p[0] for p in knot_curve])
			xM=max([p[0] for p in knot_curve])
			ym=min([p[1] for p in knot_curve])
			yM=max([p[1] for p in knot_curve])
			zm=min([p[2] for p in knot_curve])
			zM=max([p[2] for p in knot_curve])
			resx=(xM-xm)/(n_voxels)
			resy=(yM-ym)/(n_voxels)
			resz=(zM-zm)/(n_voxels)
	
			res=max([resx,resy,resz])
			infx,supx=xm-vox_bb_offset*res,xM+vox_bb_offset*res
			infy,supy=ym-vox_bb_offset*res,yM+vox_bb_offset*res
			infz,supz=zm-vox_bb_offset*res,zM+vox_bb_offset*res
			#print("sup:", abs(supy-res*m-infy), res)
			return infx,supx,infy,supy,infz,supz,res,n_voxels+2*vox_bb_offset

def minimal_neighborhood_Nvoxels(tubular_array,bb_length):
	minimal_neighborhood=min(tubular_array)
	min_voxels_per_Axis=10+int(np.ceil(bb_length/minimal_neighborhood))
	return minimal_neighborhood,min_voxels_per_Axis


#
# former minimal_field2
def minimal_field(core,directories,min_voxels_per_Axis,offset,max_blocks_per_batch):
	streams=[]
	data=[]
	data_gpu=[]
	
	grid_sizes=[]
	
	infx,supx,infy,supy,infz,supz,res,used_voxels=grid_info(core,min_voxels_per_Axis,offset)
	array_nodes=used_voxels+1 # per axis
	flatten_arrayfield_dim=(array_nodes)**3
	flatten_arrayfield=np.ones(flatten_arrayfield_dim)

	total_blocks=int(np.ceil(flatten_arrayfield_dim/1024))
	total_threads=int(np.ceil(flatten_arrayfield_dim/1024))*1024

	#

	#if int(np.ceil(total_blocks/max_blocks_per_batch))<max_blocks_per_batch: ### previous
	if total_blocks<max_blocks_per_batch:
		blocks_per_batch=int(np.ceil(total_blocks/max_blocks_per_batch))
		grid_sizes=[(blocks_per_batch,1,1)]
	else:
		blocks_per_batch=max_blocks_per_batch
		n=int(total_blocks/max_blocks_per_batch)
		grid_sizes=[(blocks_per_batch,1,1) for i in range(n)]
		m=total_blocks-n*max_blocks_per_batch
		if m>0:
			grid_sizes.append((m,1,1))

	threads_per_batch_per_block=1024
	threads_per_batch=threads_per_batch_per_block*max_blocks_per_batch
	total_batches=len(grid_sizes)

	streams=[cuda.Stream() for i in range(total_batches)]
	data=[np.ones(threads_per_batch) for i in range(total_batches)]
	data_gpu=[gpuarray.to_gpu(data[i].astype(np.float64)) for i in range(total_batches)]
	
	batch_origins=[]
	for i in range(total_batches):
		z=int(i*threads_per_batch/(array_nodes**2))
		x=int((i*threads_per_batch-z*(array_nodes**2))%array_nodes)
		y=int((i*threads_per_batch-z*(array_nodes**2))/array_nodes)
		batch_origins.append( (np.float64(x*res),np.float64(y*res),np.float64(z*res)) )

	f=open("cmbp_fieldgenerator.c",'r')
	fg_str=f.read()
	f.close()
	
	
	fg_str=fg_str.replace("@nx",str(array_nodes)) # max number of nodes in whole process
	fg_str=fg_str.replace("@ny",str(array_nodes))
	fg_str=fg_str.replace("@nz",str(array_nodes))
	
	fg_str=fg_str.replace("@dim_grid",str(threads_per_batch)) #max number of nodes per batch

	fg_str=fg_str.replace("@nkp",str(len(core.knot)))
	
	FG=SourceModule(fg_str)
	fg_ScalarField=FG.get_function("scalar_field")

	for ii in range(len(directories)):
		tubular=np.loadtxt(directories[ii])
		is_link=np.int32(0)
		if "bar" in directories[ii]:
			is_link=np.int32(1)
			#bar=np.zeros(core.n_knot_nodes)
			#print(directories[ii])
			bar_index=eval(directories[ii].strip().rstrip().split("bar_")[1][:-4])
			bar_array=core.bars[bar_index-1]
			#print(len(core.bars),bar_array.shape)
			array_gpu=gpuarray.to_gpu(bar_array.astype(np.float64))
		else:
			array_gpu=gpuarray.to_gpu(core.knot.astype(np.float64))
		
		reach=np.loadtxt(directories[ii])
		reach_gpu=gpuarray.to_gpu(reach.astype(np.float64))
		output_GPU=[]
		for k in range(total_batches):
			b=(threads_per_batch_per_block,1,1)
			g=grid_sizes[k]
			res_gpu=np.float64(res)
			infx,infy,infz=np.float64(infx),np.float64(infy),np.float64(infz)#batch_origins[k]
			batch_number=np.int32(k)
			fg_ScalarField(data_gpu[k],array_gpu,infx,infy,infz,res_gpu,reach_gpu,is_link,batch_number,block=b,grid=g,stream=streams[k])
			#pass
			output_GPU.append(data_gpu[k].get_async(stream=streams[k]))
			#print(directories[ii],k)
	data_gpu=[gpuarray.to_gpu(np.copy(output_GPU[kk]).astype(np.float64)) for kk in range(total_batches)]
	#print("len(out_GPU)",len(output_GPU),output_GPU[0].shape, output_GPU[-1].shape)
	out_list=[]
	for k in range(total_batches-1):
		out_list=out_list+list(output_GPU[k])
	n_out_threads=len(out_list)
	#print(len(out_list))
	counter=0
	while(len(out_list)<flatten_arrayfield_dim and counter<threads_per_batch):
		out_list.append(output_GPU[-1][counter])
	
	field=np.array(out_list)
	field=field.reshape((array_nodes,array_nodes,array_nodes))	
		
	return(field)


# 
# automatic method

def compute_blowup_neighborhood(core,n_directions,bb_length,max_blocks_per_batch):
	n_knot_nodes=len(core.knot)

	f=open("auto_max_blwp.c",'r')
	blp_str=f.read()
	f.close()

	dim_container=n_knot_nodes*n_directions
	total_blocks=int(np.ceil(dim_container/1024))
	total_threads=total_blocks*1024

	total_batches=int(np.ceil(total_blocks/max_blocks_per_batch) )

	if total_blocks<max_blocks_per_batch:
		blocks_per_batch=total_blocks#int(np.ceil(total_blocks/max_blocks_per_batch))
		grid_sizes=[(blocks_per_batch,1,1)]
	else:
		blocks_per_batch=max_blocks_per_batch
		n=int(total_blocks/max_blocks_per_batch)
		grid_sizes=[(blocks_per_batch,1,1) for i in range(n)]
		m=total_blocks-n*max_blocks_per_batch
		if m>0:
			grid_sizes.append((m,1,1))
	threads_per_batch_per_block=1024
	threads_per_batch=threads_per_batch_per_block*blocks_per_batch#max_blocks_per_batch
	total_batches=len(grid_sizes)

	streams=[cuda.Stream() for i in range(total_batches)]
	data=[1.0E4*np.ones(threads_per_batch) for i in range(total_batches)]
	data_gpu=[gpuarray.to_gpu(data[i].astype(np.float64)) for i in range(total_batches)]
	

	blp_str=blp_str.replace("// @curve_parametrization",core.parametrization.c)
	blp_str=blp_str.replace("@nkp",str(len(core.knot)))
	blp_str=blp_str.replace("@dim_container",str(dim_container))# n_knot_nodes*n_directions
	blp_str=blp_str.replace("@ndirec",str(n_directions))
	blp_str=blp_str.replace("@threads_per_batch",str(threads_per_batch))#threads_per_batch_per_block*blocks_per_batch
	#n_directions_gpu=np.int32(n_directions)
	BLP=SourceModule(blp_str)
	auto_Compute_MBU=BLP.get_function("Auto_blow_up_computation")

	output_GPU=[]

	for k in range(total_batches):
		b=(threads_per_batch_per_block,1,1)
		g=grid_sizes[k]		
		batch_number=np.int32(k)
		auto_Compute_MBU(data_gpu[k],batch_number,block=b,grid=g,stream=streams[k])
		output_GPU.append(data_gpu[k].get_async(stream=streams[k]))
	
	counter=0
	out_list=[]

	for k in range(total_batches-1):
		out_list=out_list+list(output_GPU[k])
	while(len(out_list)<dim_container and counter<threads_per_batch):
		out_list.append(output_GPU[-1][counter])
		counter+=1

	tubular=np.array(out_list).reshape((n_knot_nodes,n_directions))

	return tubular

#

def automatic_field2(core,directories,offset,max_blocks_per_batch):
	bb_length=BB_length(core)
	voxels_per_Axis=11
	n_knot_nodes=len(core.knot)
	
	for file in [x for x in directories if "bar" not in x]: # no bars in this version
		tubular_array=np.loadtxt(file) #medial_axis_ray
		n_nodes,n_rays=tubular_array.shape

		minimal_neighborhood,vx_p_a=minimal_neighborhood_Nvoxels(tubular_array,bb_length)
		voxels_per_Axis=max(voxels_per_Axis,vx_p_a)

	infx,supx,infy,supy,infz,supz,res,used_voxels=grid_info(core,voxels_per_Axis,offset)
	#voxels_per_Axis=voxels_per_Axis+2*offset
	array_nodes=used_voxels+1
	dim_container=array_nodes**3
	print("*** CHECK: ",core.label, dim_container)
	field=np.ones(int(dim_container)) # flatten

	#
	out_list=[]
	f=open("auto_fieldgenerator.c",'r')
	fieldgen_str=f.read()
	f.close()

	#n_knot_nodes_per_batch=n_knot_nodes 

	max_knodes_per_batch=50
	if n_knot_nodes<=max_knodes_per_batch:
		n_knot_nodes_per_batch=[n_knot_nodes]
	else:
		total=int(n_knot_nodes/max_knodes_per_batch)
		n_knot_nodes_per_batch=[max_knodes_per_batch for iii in range(total)]
		n_knot_nodes_per_batch=n_knot_nodes_per_batch+[n_knot_nodes-total]

	# GPU workload settings

	threads_per_batch_per_block=1024
	total_blocks=int(np.ceil(dim_container/1024))
	total_threads=total_blocks*1024

	total_batches=int(np.ceil(total_blocks/max_blocks_per_batch) )
	threads_per_batch=threads_per_batch_per_block*max_blocks_per_batch

	if total_blocks<max_blocks_per_batch:
		blocks_per_batch=int(np.ceil(total_blocks/max_blocks_per_batch))
		grid_sizes=[(blocks_per_batch,1,1)]
	else:
		blocks_per_batch=max_blocks_per_batch
		n=int(total_blocks/max_blocks_per_batch)
		grid_sizes=[(blocks_per_batch,1,1) for i in range(n)]
		m=total_blocks-n*max_blocks_per_batch
		if m>0:
			grid_sizes.append((m,1,1))
	data=[np.ones(threads_per_batch) for i in range(total_batches)]
	data_gpu=[gpuarray.to_gpu(data[i].astype(np.float64)) for i in range(total_batches)]
	
	batch_origins=[]
	for i in range(total_batches):
		z=int(i*threads_per_batch/(array_nodes**2))
		x=int((i*threads_per_batch-z*(array_nodes**2))%array_nodes)
		y=int((i*threads_per_batch-z*(array_nodes**2))/array_nodes)
		batch_origins.append( (np.float64(x*res),np.float64(y*res),np.float64(z*res)) )
	#

	fieldgen_str=fieldgen_str.replace("// @curve_parametrization",core.parametrization.c)
	fieldgen_str=fieldgen_str.replace("@nkp",str(n_knot_nodes))
	fieldgen_str=fieldgen_str.replace("@n_directions",str(n_rays))
	fieldgen_str=fieldgen_str.replace("@max_knodes_per_batch",str(max_knodes_per_batch))
	fieldgen_str=fieldgen_str.replace("@dim_grid",str(threads_per_batch)) #max number of nodes per batch
	fieldgen_str=fieldgen_str.replace("@nx",str(array_nodes)) # max number of nodes in whole process
	fieldgen_str=fieldgen_str.replace("@ny",str(array_nodes))
	fieldgen_str=fieldgen_str.replace("@nz",str(array_nodes))

	streams=[]
	data=[]
	data_gpu=[]

	streams=[cuda.Stream() for i in range(total_batches)]
	
	res_gpu=np.float64(res)
	for k in range(total_batches):
			# for over batch_index (streams)
			fieldgen_str=fieldgen_str.replace("@nkp_per_batch",str(n_knot_nodes_per_batch[k]))# k=batch_index
			FG=SourceModule(fieldgen_str)
			auto_fieldgenerator=FG.get_function("Auto_pump")

			b=(threads_per_batch_per_block,1,1)
			g=grid_sizes[k]		
			infx,infy,infz=np.float64(infx),np.float64(infy),np.float64(infz)#batch_origins[k]
			batch_number=np.int32(k)
			for file in [x for x in directories if "bar" not in x]: # no bars in this version
				tubular_array=np.loadtxt(file)
				for indx_k in range(len(n_knot_nodes_per_batch)):
					if indx_k!=(len(n_knot_nodes_per_batch)-1):
						tb=tubular[indx_k:(indx_k+1)][0:n_rays]
					else:
						tb=tubular[-n_knot_nodes_per_batch[-1]:][0:n_rays]
					medialAxis_gpu=gpuarray.to_gpu(tb.astype(np.float64))
					indx_k_gpu=np.int32(indx_k)

					auto_fieldgenerator(data_gpu[k],indx_k_gpu,infx,infy,infz,res_gpu,medialAxis_gpu,batch_number,block=b,grid=g,stream=streams[k])
			#pass
			output_GPU.append(data_gpu[k].get_async(stream=streams[k]))
	#
	data_gpu=[gpuarray.to_gpu(np.copy(output_GPU[kk]).astype(np.float64)) for kk in range(total_batches)]
	#print("len(out_GPU)",len(output_GPU),output_GPU[0].shape, output_GPU[-1].shape)
	out_list=[]
	for k in range(total_batches-1):
		out_list=out_list+list(output_GPU[k])
	n_out_threads=len(out_list)
	#print(len(out_list))
	counter=0
	while(len(out_list)<dim_container and counter<threads_per_batch):
		out_list.append(output_GPU[-1][counter])
	field=np.array(out_list)
	field=field.reshape((array_nodes,array_nodes,array_nodes))	
		
	
	return(field)


def automatic_field(core,directories,offset,max_blocks_per_batch):
	bb_length=BB_length(core)
	voxels_per_Axis=11
	n_knot_nodes=len(core.knot)
	
	for file in [x for x in directories if "bar" not in x]: # no bars in this version
		tubular_array=np.loadtxt(file) #medial_axis_ray
		#n_nodes,n_rays=tubular_array.shape
		n_nodes=tubular_array.shape
		minimal_neighborhood,vx_p_a=minimal_neighborhood_Nvoxels(tubular_array,bb_length)
		voxels_per_Axis=vx_p_a#max(voxels_per_Axis,vx_p_a)

	infx,supx,infy,supy,infz,supz,res,used_voxels=grid_info(core,voxels_per_Axis,offset)
	tub=0.5*res*np.sqrt(3)*np.ones(n_knot_nodes)




	streams=[]
	data=[]
	data_gpu=[]
	
	grid_sizes=[]
	
	#infx,supx,infy,supy,infz,supz,res,used_voxels=grid_info(core,min_voxels_per_Axis,offset)
	array_nodes=used_voxels+1 # per axis
	flatten_arrayfield_dim=(array_nodes)**3
	flatten_arrayfield=np.ones(flatten_arrayfield_dim)

	total_blocks=int(np.ceil(flatten_arrayfield_dim/1024))
	total_threads=int(np.ceil(flatten_arrayfield_dim/1024))*1024

	#max_blocks_per_batch=2

	#if int(np.ceil(total_blocks/max_blocks_per_batch))<max_blocks_per_batch: ### previous
	if total_blocks<max_blocks_per_batch:
		blocks_per_batch=int(np.ceil(total_blocks/max_blocks_per_batch))
		grid_sizes=[(blocks_per_batch,1,1)]
	else:
		blocks_per_batch=max_blocks_per_batch
		n=int(total_blocks/max_blocks_per_batch)
		grid_sizes=[(blocks_per_batch,1,1) for i in range(n)]
		m=total_blocks-n*max_blocks_per_batch
		if m>0:
			grid_sizes.append((m,1,1))

	threads_per_batch_per_block=1024
	threads_per_batch=threads_per_batch_per_block*max_blocks_per_batch
	total_batches=len(grid_sizes)

	streams=[cuda.Stream() for i in range(total_batches)]
	data=[np.ones(threads_per_batch) for i in range(total_batches)]
	data_gpu=[gpuarray.to_gpu(data[i].astype(np.float64)) for i in range(total_batches)]
	
	batch_origins=[]
	for i in range(total_batches):
		z=int(i*threads_per_batch/(array_nodes**2))
		x=int((i*threads_per_batch-z*(array_nodes**2))%array_nodes)
		y=int((i*threads_per_batch-z*(array_nodes**2))/array_nodes)
		batch_origins.append( (np.float64(x*res),np.float64(y*res),np.float64(z*res)) )

	f=open("cmbp_fieldgenerator.c",'r')
	fg_str=f.read()
	f.close()
	
	
	fg_str=fg_str.replace("@nx",str(array_nodes)) # max number of nodes in whole process
	fg_str=fg_str.replace("@ny",str(array_nodes))
	fg_str=fg_str.replace("@nz",str(array_nodes))
	
	fg_str=fg_str.replace("@dim_grid",str(threads_per_batch)) #max number of nodes per batch

	fg_str=fg_str.replace("@nkp",str(len(core.knot)))
	
	FG=SourceModule(fg_str)
	fg_ScalarField=FG.get_function("scalar_field")

	for ii in range(len(directories)):
		tubular=np.loadtxt(directories[ii])
		is_link=np.int32(0)
		if "bar" in directories[ii]:
			pass
		else:
			array_gpu=gpuarray.to_gpu(core.knot.astype(np.float64))
		
			reach=np.loadtxt(directories[ii])
			#reach_gpu=gpuarray.to_gpu(tub.astype(np.float64))
			reach_gpu=gpuarray.to_gpu(reach.astype(np.float64))
			output_GPU=[]
			for k in range(total_batches):
				b=(threads_per_batch_per_block,1,1)
				g=grid_sizes[k]
				res_gpu=np.float64(res)
				infx,infy,infz=np.float64(infx),np.float64(infy),np.float64(infz)#batch_origins[k]
				batch_number=np.int32(k)
				fg_ScalarField(data_gpu[k],array_gpu,infx,infy,infz,res_gpu,reach_gpu,is_link,batch_number,block=b,grid=g,stream=streams[k])
				#pass
				output_GPU.append(data_gpu[k].get_async(stream=streams[k]))
				#print(directories[ii],k)
	data_gpu=[gpuarray.to_gpu(np.copy(output_GPU[kk]).astype(np.float64)) for kk in range(total_batches)]
	#print("len(out_GPU)",len(output_GPU),output_GPU[0].shape, output_GPU[-1].shape)
	out_list=[]
	for k in range(total_batches-1):
		out_list=out_list+list(output_GPU[k])
	n_out_threads=len(out_list)
	#print(len(out_list))
	counter=0
	while(len(out_list)<flatten_arrayfield_dim and counter<threads_per_batch):
		out_list.append(output_GPU[-1][counter])
		counter+=1
	field=np.array(out_list)
	field=field.reshape((array_nodes,array_nodes,array_nodes))	
		
	return(field)
