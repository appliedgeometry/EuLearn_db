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
import numpy as np
import os
import trimesh
import shutil
import os
import sys


out_dir=par.args.o
knots_file=par.args.kf
ascii_names=par.args.ascii_names
name_format=par.args.name_format
name_separator="_"
log_file=par.args.log
fxnormals=par.args.fxnormals
copy_from=par.args.copy_from
scalarfields_dir=par.args.fields_dir
nsmooth_dir=par.args.nsmooth_dir
blowup_dir=par.args.bup_dir

fxnormals=os.path.join(out_dir,fxnormals)
log_file=os.path.join(out_dir,log_file)
copy_from=os.path.join(out_dir,copy_from)
scalarfields_dir= os.path.join(out_dir,scalarfields_dir)
nsmooth_dir=os.path.join(out_dir,nsmooth_dir)
blowup_dir=os.path.join(out_dir,blowup_dir)

genus_class_dir=out_dir



parameters_dictionary={"out_dir":out_dir,"fixed_dir":fxnormals,#"post_processing_dir":post_processing_dir,
"knots_file":knots_file,"ascii_filenames":ascii_names,"name_format":name_format,
"name_separator":name_separator,"log_file":log_file}

cwd=os.getcwd()
sys.path.append(cwd)
from bpy_utils import *


def get_summary(log_file,fxnormals,summary_file):
	genus_dict={}

	f=open(knots_file,'r')
	knot_list=f.readlines()
	knot_list=[line for line in knot_list if ',' in line]
	f.close()

	knots_tree,main_dirs=dn.get_directories(knots_file,ascii_names)
	knots_tree.sort()
	#print(knots_tree)
	blp={}
	f=open(log_file,'r')
	data=f.read()
	f.close()
	g=open(summary_file,'w')
	header="# knot type, nx,ny,nz, phix,phiy,phiz, genus, nodes, arc-length, voxel size, bars, voxels per axis, offset(2x), knot index, filename, method"
	header=header+", blowup(s), scalar field(s), marchingQ(s), total time(s)"+'\n'
	g.write(header)
	try:
		data=data.replace('#',' ')
	except:
		pass
	d=data.split('SUB-PROCESS:')
	d1=d[1:]
	stl_knots=[elem for elem in os.listdir(fxnormals)]
	#print(stl_knots)

	dictionaries={"genus":{},
				"time":{'pump':{},'marchingQ':{},'blowup':{}},
				"knot_info":{},
				"stlnames":{},
				"labels":{},
				"area":{},
				"ns_area":{}}

	for kk in range(len(knots_tree)):
		knot=knots_tree[kk]
		knot_index_str=eval(knot.split("_")[0])
		fields_dir="mq_Fields"
		label=filename_format(knot,knot_index_str,parameters_dictionary,fields_dir)#filename_format(knot,knot_index_str)
		dictionaries["labels"][knot_index_str]=label
	#print(dictionaries["labels"])
	for kk in range(len(knots_tree)):
		knot=knots_tree[kk]
		knot_index_str=eval(knot.split("_")[0])	
		l_aux=[dictionaries["labels"][knot_index_str]+"_oriented.stl"]
		if len(l_aux)>0:
			path=os.path.join(fxnormals,l_aux[0])
			#print(path)
			mesh = trimesh.load_mesh(path)
			# ------------------------
			path2=os.path.join(copy_from,l_aux[0].replace("_oriented",""))
			mesh2 = trimesh.load_mesh(path2)
			A=round(mesh2.area,3)
			dictionaries["area"][knot_index_str]=A
			
			path3=os.path.join(nsmooth_dir,l_aux[0].replace("_oriented",""))
			mesh3 = trimesh.load_mesh(path3)
			ns_A=round(mesh3.area,3)
			dictionaries["ns_area"][knot_index_str]=ns_A
			
			# --------------------------
			euler_char=mesh.euler_number
			genus=int((2-euler_char)/2)
			# genus
			dictionaries["genus"][knot_index_str]=genus
			dictionaries["stlnames"][knot_index_str]=dictionaries["labels"][knot_index_str]#label
	for sbp in d1:
		if "VOXEL RESOLUTION" in sbp:
			subp="blowup"
		elif "STL PATCHES" in sbp:
			subp="marchingQ"
		elif "pump" in sbp:
			subp="pump"
		else:
			pass
		#
		# TIME
		knot_index=sbp.split("KNOT INDEX:")[1]
		knot_index=knot_index.split("\n")[0]
		time_sec=sbp.split("SUB-time [s]:")[1].split("\n")[0]
		dictionaries["time"][subp][eval(knot_index)]=eval(time_sec)

		# # knot info
		if subp=="blowup":
			lines=[x.strip().rstrip() for x in sbp.split('\n')]
			for i in range(len(lines)):
				line=lines[i]
				if "KNOT INDEX:" in line:
					knot_index=eval(line.split(":")[1])
				if "VOXEL RESOLUTION" in line:
					scale_factor=eval(line.split(":")[1])
				if "BARS:" in line:
					n_bars=eval(line.split(":")[1])
				if "KNOT_NODES:" in line:
					knot_nodes=eval(line.split(":")[1])
				if "VOXELS PER AXIS:" in line:
					voxels_per_axis=eval(line.split(":")[1])
				if "ARC LENGTH:" in line:
					arc_length=eval(line.split(":")[1])
				if "METHOD:" in line:
					method=line.split(":")[1]
					method=method.strip().rstrip()
				if "BOUNDING BOX OFFSET" in line:
					offset=2*eval(line.split(":")[1])
			dictionaries["knot_info"][knot_index]=[knot_index,scale_factor,n_bars,knot_nodes,voxels_per_axis,offset,arc_length,method]
		#
		
	#print(dictionaries["time"])
	sorted_keys=list(dictionaries["labels"].keys())
	#print(sorted_keys)
	sorted_keys.sort()
	for key in sorted_keys:
		knot_index,scale_factor,n_bars,knot_nodes,voxels_per_axis,offset,arc_length,method=dictionaries["knot_info"][key]
		row=knot_list[knot_index].strip().rstrip()
		index_bar=row.find("[")
		row=row[:index_bar]
		knotparams=row.split(',')
		r=knotparams[0]+","
		for i in range(1,len(knotparams)-1):
			#print(knotparams[i],key,row)
			val=eval(knotparams[i])

			r=r+str(val)+","
		r=r+str(dictionaries["genus"][knot_index])+","+str(knot_nodes)+","+str(arc_length)+","+str(scale_factor)+","+str(n_bars)+","
		r=r+str(voxels_per_axis)+","+str(offset)+","+str(knot_index)+","+dictionaries["stlnames"][knot_index].replace(",","_")+","+method
		total_s=str(dictionaries["time"]["blowup"][knot_index]+dictionaries["time"]["pump"][knot_index]+dictionaries["time"]["marchingQ"][knot_index])
		r=r+","+str(dictionaries["time"]["blowup"][knot_index])+","+str(dictionaries["time"]["pump"][knot_index])+","+str(dictionaries["time"]["marchingQ"][knot_index])+","+str(total_s)
		r=r+"\n"
		g.write(r)
	g.close()
	return dictionaries


dictionaries=get_summary(log_file,fxnormals,os.path.join(out_dir,'summary.csv'))
if len(out_dir)>len("output") and os.path.sep in out_dir:
	out_name=out_dir.split("/")[-1].replace("output","")
	shutil.copy(os.path.join(out_dir,"summary.csv"),os.path.join(out_dir,"summary"+out_name+".csv") )
	#shutil.copy(log_file,os.path.join(out_dir,"log_"+out_name+".txt") )


if not os.path.exists( genus_class_dir ):
	os.mkdir(genus_class_dir)

knots=os.listdir(copy_from)

#print(dictionaries["genus"])

for key in dictionaries["genus"].keys():
	
	knot_index=key#genus_dict[key][1]
	name=dictionaries["stlnames"][knot_index]
	genus=dictionaries["genus"][knot_index]


	genus_label='g'+str(int(genus))
	#save_path=os.path.join(genus_class_dir,genus_label)
	save_path=genus_class_dir
	

	if not os.path.exists( save_path ):
		os.mkdir(save_path)

	stl_name=genus_label+'_'+name+".stl"

	stl_original=os.path.join(copy_from,name+".stl")

	#shutil.copy(stl_original,os.path.join(save_path,stl_name) )
	#shutil.copy(stl_original,os.path.join(save_path,stl_name[:-4]+"_a"+str(dictionaries["area"][knot_index])+".stl") )
	shutil.copy(stl_original,os.path.join(save_path,stl_name[:-4]+".stl") )

	# Copy other files with same name
	scfields=os.listdir(scalarfields_dir)
	for dir_flds in scfields:
		ind=eval(dir_flds.split("_")[0])
		#print(ind,type(ind),type(knot_index),dir_flds)
		if ind==knot_index:
			#print(scalarfields_dir,dir_flds,fn)
			fn=os.listdir(os.path.join(scalarfields_dir,dir_flds))[0]
			#shutil.copy(os.path.join(scalarfields_dir,dir_flds,fn),os.path.join(save_path,stl_name[:-4]+"_a"+str(dictionaries["area"][knot_index])+"_sf.txt") )
			shutil.copy(os.path.join(scalarfields_dir,dir_flds,fn),os.path.join(save_path,stl_name[:-4]+"_sf.txt") )
			# copy blow-up files without bars
			shutil.copy(os.path.join(blowup_dir,dir_flds,fn),os.path.join(save_path,stl_name[:-4]+"_bup.txt") )
			#
	stlNS_dirs=os.listdir(nsmooth_dir)

	for stl3d in stlNS_dirs:
		#ind=eval(stl3d.split("_")[0])
		if stl3d in stl_name:
			#fn=os.listdir(os.path.join(scalarfields_dir,dir_flds))[0]
			fn=os.path.join(nsmooth_dir,stl3d)
			#shutil.copy(fn,os.path.join(save_path,stl_name[:-4]+"_a"+str(dictionaries["ns_area"][knot_index])+"_ns.stl") )
			shutil.copy(fn,os.path.join(save_path,stl_name[:-4]+"_ns.stl") )
