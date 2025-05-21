# Created by Applied Geometry Laboratory (https://github.com/appliedgeometry)
# for "EuLearn: A 3D database for learning Euler characteristics" Project, 2025.
#
#
# https://huggingface.co/datasets/appliedgeometry/EuLearn
# https://github.com/appliedgeometry/EuLearn_db
#
# GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007

import bmesh
import bpy
from bpy import context

import sys
import os

out_dir = os.getenv('output_dir')
log_file = os.getenv('log_file')
fixed_dir = os.getenv('fix_Normals_dir')
post_processing_dir = os.getenv('post_processing_dir')

fixed_dir=os.path.join(out_dir,fixed_dir)
post_processing_dir=os.path.join(out_dir,post_processing_dir)
log_file=os.path.join(out_dir,log_file)


translate_to_origin_CONDITION = os.getenv('translate_to_origin')
scale_CONDITION = os.getenv('scale')
smooth_CONDITION = os.getenv('smooth')

knots_file=os.getenv('knots_file')
ascii_filenames=os.getenv('ascii_filenames')
name_format=os.getenv("name_format")
name_separator="_"#os.getenv("name_separator")

if smooth_CONDITION=='True':
	smooth_factor=float(eval(os.getenv('smooth_factor')))
	smooth_iterations=int(eval(os.getenv('smooth_iterations')))
else:
	pass


parameters_dictionary={"out_dir":out_dir,"fixed_dir":fixed_dir,#"post_processing_dir":post_processing_dir,
"knots_file":knots_file,"ascii_filenames":ascii_filenames,"name_format":name_format,
"name_separator":name_separator,"log_file":log_file}

bpy.ops.object.select_all(action="DESELECT")

bpy.data.objects['Cube'].select_set(True)
bpy.ops.object.delete()


dirs=[os.path.join(fixed_dir, y) for y in os.listdir(fixed_dir)]

dirs=[d for d in dirs if os.path.isdir(d)]
dirs.sort()

cwd=os.getcwd()
sys.path.append(cwd)
from bpy_utils import *

input_knots=[knot[:-13] for knot in os.listdir(fixed_dir) if ".stl" in knot]
#print(input_knots)
if len(input_knots)>0:
	scales=get_ScaleFactors(log_file)
	#print(scales)
	fields_dir="mq_Fields"
	knot_files=os.listdir(os.path.join(out_dir,"Max_BlowUP_Computation"))
	for knot in knot_files:
		knot_index=eval(knot.split("_")[0])
		knotname=filename_format(knot,knot_index,parameters_dictionary,fields_dir)
		if knotname in input_knots:
			#print("--->",knotname,knot,knotname in input_knots,knot_index)
			scale=scales[knot_index]
			scene = bpy.context.scene
			nfixed_knotname=[elem for elem in os.listdir(fixed_dir) if knotname in elem][0]
			bpy.ops.import_mesh.stl(filepath=os.path.join(fixed_dir,nfixed_knotname))
			mesh_obs = [o for o in scene.objects if o.type == 'MESH']
			for ob in mesh_obs:
				stl_path = os.path.join(post_processing_dir,knotname+".stl")
				if translate_to_origin_CONDITION=="True":
					bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN')
				if scale_CONDITION=="True":
					bpy.data.objects[ob.name].scale = (scale,scale,scale)
				if smooth_CONDITION=="True":

					modif=ob.modifiers.new(type="SMOOTH",name='smooth_mod')
					modif.iterations=smooth_iterations #30
					modif.factor=smooth_factor #0.25

				bpy.ops.export_mesh.stl(filepath=str(stl_path),use_selection=True)
				bpy.ops.object.delete()
