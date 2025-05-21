# Created by Applied Geometry Laboratory (https://github.com/appliedgeometry)
# for "EuLearn: A 3D database for learning Euler characteristics" Project, 2025.
#
#
# https://huggingface.co/datasets/appliedgeometry/EuLearn
# https://github.com/appliedgeometry/EuLearn_db
#
# GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007

import bpy
import os
import sys

out_dir=val = os.getenv('output_dir')
merged_dir=val = os.getenv('merged_STL_dir')
fixed_dir=val = os.getenv('fix_Normals_dir')
knots_file=os.getenv('knots_file')
ascii_filenames=os.getenv('ascii_filenames')
name_format=os.getenv("name_format")
name_separator="_"#os.getenv("name_separator")
log_file = os.getenv('log_file')




merged_dir=os.path.join(out_dir,merged_dir)
fixed_dir=os.path.join(out_dir,fixed_dir)
log_file=os.path.join(out_dir,log_file)

parameters_dictionary={"out_dir":out_dir,"merged_dir":merged_dir,"fixed_dir":fixed_dir,
"knots_file":knots_file,"ascii_filenames":ascii_filenames,"name_format":name_format,
"name_separator":name_separator,"log_file":log_file}


bpy.ops.object.select_all(action="DESELECT")

bpy.data.objects['Cube'].select_set(True)
bpy.ops.object.delete()



dirs=[os.path.join(out_dir,y) for y in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir,y))]
dirs=[d for d in dirs if d!=fixed_dir and d!=merged_dir]

for subdir in dirs:
	sd=[os.path.join(subdir,y) for y in os.listdir(subdir) if os.path.isdir(os.path.join(subdir,y))]
	knot_subdirs_sum=sum([len([x for x in os.listdir(sd[j]) if '.stl' in x]) for j in range(len(sd))])
	if knot_subdirs_sum>0:
		input_dir=subdir
		break
cwd=os.getcwd()
sys.path.append(cwd)
from bpy_utils import *

KnotDirs=[os.path.join(input_dir,x) for x in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir,x))]
KnotDirs.sort()

for i in range(len(KnotDirs)):
	dir=KnotDirs[i]

	knot_index_str=eval(dir.split(os.sep)[-1].split("_")[0])
	knot_name=dir.split(os.sep)[-1]

	sub_stls=[os.path.join(dir,x) for x in os.listdir(dir) if '.stl' in x]

	for file in sub_stls:
		bpy.ops.import_mesh.stl(filepath=file)

	from pathlib import Path
	context = bpy.context
	scene = context.scene
	viewlayer = context.view_layer
	obs = [o for o in scene.objects if o.type == 'MESH']
	bpy.ops.object.select_all(action='DESELECT')    

	import bmesh
	from bpy import context
	bm = bmesh.new()
	for ob in bpy.context.scene.objects:
		if ob.type == 'MESH':
			ob.select_set(True)

	bpy.ops.object.join()
	
	fields_dir="mq_Fields"
	stl_path = os.path.join(merged_dir,filename_format(knot_name,knot_index_str,parameters_dictionary,fields_dir)+".stl")
	bpy.ops.export_mesh.stl(filepath=stl_path,use_selection=True)
	bpy.ops.object.delete()

#
