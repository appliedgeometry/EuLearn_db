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
merged_dir = os.getenv('merged_STL_dir')
fixed_dir = os.getenv('fix_Normals_dir')


merged_dir=os.path.join(out_dir,merged_dir)
fixed_dir=os.path.join(out_dir,fixed_dir)


bpy.ops.object.select_all(action="DESELECT")

bpy.data.objects['Cube'].select_set(True)
bpy.ops.object.delete()

sd=[os.path.join(merged_dir,y) for y in os.listdir(merged_dir) if '.stl' in y]
names=os.listdir(merged_dir)
#print(sd)
for index in range(len(sd)):
		file=sd[index]
		#print(file)
		bpy.ops.import_mesh.stl(filepath=file)
		bm = bmesh.new()
		meshes = set(o.data for o in context.selected_objects if o.type == 'MESH')
		for mesh in meshes:
			bm.from_mesh(mesh)
			bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
			bm.to_mesh(mesh)
			bm.clear()
			mesh.update()
		bm.free()
		from pathlib import Path
		context = bpy.context
		scene = context.scene
		viewlayer = context.view_layer
		obs = [o for o in scene.objects if o.type == 'MESH']
		bpy.ops.object.select_all(action='DESELECT')    
		for ob in obs:
			viewlayer.objects.active = ob
			ob.select_set(True)
			stl_path = os.path.join(fixed_dir,names[index][:-4]+"_oriented"+".stl")
			bpy.ops.export_mesh.stl(filepath=str(stl_path),use_selection=True)
			ob.select_set(False)
		bpy.ops.object.delete()


