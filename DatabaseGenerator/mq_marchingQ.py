# Created by Applied Geometry Laboratory (https://github.com/appliedgeometry)
# for "EuLearn: A 3D database for learning Euler characteristics" Project, 2025.
#
#
# https://huggingface.co/datasets/appliedgeometry/EuLearn
# https://github.com/appliedgeometry/EuLearn_db
#
# GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007

# ############################################
import par_parameters as par
import par_dirnames as dn
import cmbp_submethods as methods
import os
from skimage import measure

out_dir=par.args.o
knots_file=par.args.kf
max_n_vertices=par.args.vtx_split
log_file=par.args.log
ascii_names=par.args.ascii_names


log_file=os.path.join(out_dir,log_file)

if (par.args.pr == 'True'):
	from tqdm import tqdm
	loop='tqdm(range(@iterator), desc=" MarchingQ")'
else:
	loop='range(@iterator)'
	
# ############################################

import datetime as dt
from stl import mesh

import os
import numpy as np
import sys


knots_tree_directory,main_dirs=dn.get_directories(knots_file,ascii_names)

input_route=os.path.join(out_dir,main_dirs['pump_dir'])

#listado de nudos_dir
fields_dir=os.listdir(input_route)#[os.path.join(input_route,x) for x in os.listdir(input_route)]
fields_dir.sort()

marchingQ_dir=os.path.join(out_dir,main_dirs["marchingQ_dir"])


init_marchingQ=dt.datetime.now()  # <-----------------------------------
#if par.args.p==True:
iterator=eval(loop.replace("@iterator","len(fields_dir)"))


if dn.subdir_check(out_dir, main_dirs['marchingQ_dir'])==False:
	os.mkdir(marchingQ_dir)

for it in iterator:
	knot_dir=fields_dir[it]
	knot_field=os.listdir(os.path.join(input_route,knot_dir))[0]
	
	if dn.subdir_check(marchingQ_dir,knot_dir)==False:
		os.mkdir(os.path.join(marchingQ_dir,knot_dir))

	
	filename=os.path.join(input_route,knot_dir,knot_field)
	
	field = np.loadtxt(filename)
	aux_file=open(filename,'r')
	grid_size=eval(aux_file.readline().split(":")[1])
	aux_file.close()
	field = field.reshape(grid_size)
	a,b,c=field.shape
	#
	verts,faces,normals,values=measure.marching_cubes(field,level=0.0)
	knot=mesh.Mesh(np.zeros(len(faces),dtype=mesh.Mesh.dtype))
	for ii,f in enumerate(faces):
		for jj in range(3):
			knot.vectors[ii][jj]=verts[f[jj]]
	stlname=os.path.join(marchingQ_dir,knot_dir,knot_dir)
	knot.save(stlname+'.stl')
	fin_marchingQ=dt.datetime.now()
	delta=fin_marchingQ-init_marchingQ

	stamp_dict={"Knot index":it,
				"INPUT Field":filename,
				"Save File":stlname+" *",
				"STL patches":1}
	methods.savelog('marchingQ',log_file,delta.total_seconds(),stamp_dict)
	#



