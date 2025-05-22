# Created by Applied Geometry Laboratory (https://github.com/appliedgeometry)
# for "EuLearn: A 3D database for learning Euler characteristics" Project, 2025.
#
#
# https://huggingface.co/datasets/appliedgeometry/EuLearn
# https://github.com/appliedgeometry/EuLearn_db
#
# GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007

import os
import sys

def get_ScaleFactors(score_file):
	scales={}
	f=open(score_file,'r')
	data=f.read()
	f.close()
	try:
		data=data.replace('#',' ')
	except:
		pass
	d=data.split('SUB-PROCESS:')
	for sub_p in d:
		if "VOXEL RESOLUTION" in sub_p:
			lines=[x.strip().rstrip() for x in sub_p.split('\n')]
			for line in lines:
				if "KNOT INDEX:" in line:
					#knot_directory=line.split(":")[1].split('/')[-1]
					knot_index=eval(line.split(":")[1])
				if "VOXEL RESOLUTION" in line:
					scale_factor=eval(line.split(":")[1])
			scales[knot_index]=scale_factor
	return scales

def get_voxels(x,out_dir,fields_dir):
	# 
	matrix_file=os.listdir(os.path.join(out_dir,fields_dir,x))[0]
	f=open(os.path.join(out_dir,fields_dir,x,matrix_file),'r')
	array_size=f.readline()
	f.close()
	vx,vy,vz=eval(array_size.split(":")[1])
	result=vx-1
	return result

def get_minimal_vx(score_file):
	min_vx={}
	f=open(score_file,'r')
	data=f.read()
	f.close()
	try:
		data=data.replace('#',' ')
	except:
		pass
	d=data.split('SUB-PROCESS:')
	for sub_p in d:
		if "VOXEL RESOLUTION" in sub_p:
			lines=[x.strip().rstrip() for x in sub_p.split('\n')]
			for line in lines:
				if "KNOT INDEX:" in line:
					#knot_directory=line.split(":")[1].split('/')[-1]
					knot_index=eval(line.split(":")[1])
				if "VOXELS PER AXIS" in line:
					voxels=eval(line.split(":")[1])
				if "BOUNDING BOX OFFSET" in line:
					bb_offset=eval(line.split(":")[1])
			min_vx[knot_index]=voxels # +bb_offset
	return min_vx

def get_tubular_mMavg(score_file):
	tubular={}
	f=open(score_file,'r')
	data=f.read()
	f.close()
	try:
		data=data.replace('#',' ')
	except:
		pass
	d=data.split('SUB-PROCESS:')
	for sub_p in d:
		if "VOXEL RESOLUTION" in sub_p:
			lines=[x.strip().rstrip() for x in sub_p.split('\n')]
			for line in lines:
				if "KNOT INDEX:" in line:
					#knot_directory=line.split(":")[1].split('/')[-1]
					knot_index=eval(line.split(":")[1])
				if "SAVE DIR:" in line:
					#knot_directory=line.split(":")[1].split('/')[-1]
					svd=line.split(":")[1]
					svd=svd.rstrip().strip()#.replace(" ","")
			#print(svd," * ")		
			svf=[os.path.join(svd,x) for x in os.listdir(svd) if "bar" not in x]
			tmp_file=open(svf[0],'r')
			array_str=tmp_file.readlines()
			tmp_file.close()
			array_num=[eval(x.strip().rstrip()) for x in array_str if x[0]!="#"]
			min_neigh=min(array_num)
			max_neigh=max(array_num)
			avg_neigh=sum(array_num)/len(array_num)
			tubular[knot_index]=(min_neigh,max_neigh,avg_neigh)
	return tubular

def get_tubular_mMavg2(score_file):
	tubular={}
	f=open(score_file,'r')
	data=f.read()
	f.close()
	try:
		data=data.replace('#',' ')
	except:
		pass
	d=data.split('SUB-PROCESS:')
	for sub_p in d:
		if "VOXEL RESOLUTION" in sub_p:
			lines=[x.strip().rstrip() for x in sub_p.split('\n')]
			for line in lines:
				if "KNOT INDEX:" in line:
					#knot_directory=line.split(":")[1].split('/')[-1]
					knot_index=eval(line.split(":")[1])
				if "MINIMAL NEIGHBORHOOD:" in line:
					#knot_directory=line.split(":")[1].split('/')[-1]
					min_neigh=eval(line.split(":")[1])
				if "MAX NEIGHBORHOOD:" in line:
					max_neigh=eval(line.split(":")[1])
				if "AVERAGE NEIGHBORHOOD:" in line:
					avg_neigh=eval(line.split(":")[1])
			tubular[knot_index]=(min_neigh,max_neigh,avg_neigh) # +bb_offset
	return tubular

def get_method(score_file):
	methods={}
	f=open(score_file,'r')
	data=f.read()
	f.close()
	try:
		data=data.replace('#',' ')
	except:
		pass
	d=data.split('SUB-PROCESS:')
	for sub_p in d:
		if "VOXEL RESOLUTION" in sub_p:
			lines=[x.strip().rstrip() for x in sub_p.split('\n')]
			for line in lines:
				if "KNOT INDEX:" in line:
					#knot_directory=line.split(":")[1].split('/')[-1]
					knot_index=eval(line.split(":")[1])
				if "METHOD:" in line:
					#knot_directory=line.split(":")[1].split('/')[-1]
					method_type=line.split(":")[1]
					method_type=method_type.strip().rstrip()
			methods[knot_index]=method_type
	return methods
def get_iterations(score_file):
	d_iterations={}
	f=open(score_file,'r')
	data=f.read()
	f.close()
	try:
		data=data.replace('#',' ')
	except:
		pass
	d=data.split('SUB-PROCESS:')
	for sub_p in d:
		if "REACH ITERATIONS" in sub_p:
			lines=[x.strip().rstrip() for x in sub_p.split('\n')]
			for line in lines:
				if "KNOT INDEX:" in line:
					#knot_directory=line.split(":")[1].split('/')[-1]
					knot_index=eval(line.split(":")[1])
				if "REACH ITERATIONS:" in line:
					#knot_directory=line.split(":")[1].split('/')[-1]
					iterations_number=line.split(":")[1]
					iterations_number=iterations_number.strip().rstrip()
			d_iterations[knot_index]=iterations_number
	return d_iterations
	
def get_method_parameters(score_file):
	metaparameter_dictionary={}
	mparamList=[]
	f=open(score_file,'r')
	data=f.read()
	f.close()
	try:
		data=data.replace('#',' ')
	except:
		pass
	d=data.split('SUB-PROCESS:')
	for sub_p in d:
		if "VOXEL RESOLUTION" in sub_p:
			lines=[x.strip().rstrip() for x in sub_p.split('\n')]
			for line in lines:
				if "KNOT INDEX:" in line:
					knot_index=eval(line.split(":")[1])
				if "METHOD:" in line:
					method_type=line.split(":")[1]
					method_type=method_type.strip().rstrip()
				if "METAPARAM" in line:
					mparam_name,mparam_value=line.split(":")
					mparam_value=mparam_value.strip().rstrip()
					mparamList.append([mparam_name,mparam_value])
			metaparameter_dictionary[knot_index]={"method":method_type}
			for mp in mparamList:
				metaparameter_dictionary[knot_index][mp[0]]=mp[1]
		mparamList=[]
	return metaparameter_dictionary

def filename_format(x,knot_index,par_dictionary,fields_dir):
	#fields_dir="mq_Fields"
	out_dir=par_dictionary["out_dir"]
	fixed_dir=par_dictionary["fixed_dir"]
	knots_file=par_dictionary["knots_file"]
	ascii_filenames=par_dictionary["ascii_filenames"]
	name_format=par_dictionary["name_format"]
	name_separator=par_dictionary["name_separator"]
	log_file=par_dictionary["log_file"]
	try:
		merged_dir=par_dictionary["merged_dir"]
	except:
		pass

	name_keys=name_format.split(",")
	#print(x)
	input_name=x.split("_")
	#print(input_name)
	if (len(input_name)==4):
		index,type,parameters,Nbars=x.split("_")
	else:
		index,type,Nbars=x.split("_")
		parameters=''

	scales=get_ScaleFactors(log_file)
	min_voxels=get_minimal_vx(log_file)
	tubular_info=get_tubular_mMavg(log_file)

	methods=get_method(log_file)
	reach_iterations=get_iterations(log_file)
	
	metaparameter_info=get_method_parameters(log_file)

	Nvoxels="v"+str(min_voxels[knot_index])

	vxRes=scales[knot_index]#get_voxelsize(x,knot_index)
	tubular_min,tubular_max,tubular_avg= tubular_info[knot_index]

	metaparameter_method=metaparameter_info[knot_index]
	name_dict={}
	name=''
	for elem in name_keys:
		if "vxRes" in elem:
			fmt_digits=elem.split("_f")[1]
			
			total_chars,decimals=fmt_digits.split(".")
			fmt="%"+fmt_digits.replace("f","")+"f"
			#print("fmy: ", fmt_digits, " ** ", fmt)
			flt=float(fmt %vxRes)
			vxRes=vxRes_str=str(flt)#round(vxRes,decimals)
			#print(total_chars,decimals,flt)
			if total_chars=="0":
				dot_index=vxRes_str.find(".")
				vxRes=vxRes_str=vxRes_str[dot_index+1:]
			name_dict[elem]="res"+vxRes#eval(elem)
		else:
			if "rmin" in elem:
				fmt_digits=elem.split("_f")[1]
				total_chars,decimals=fmt_digits.split(".")
				fmt="%"+fmt_digits.replace("f","")+"f"
				flt=float(fmt %tubular_min)
				tubular_min=tubular_min_str=str(flt)#round(vxRes,decimals)
				
				if total_chars=="0":
					dot_index=tubular_min_str.find(".")
					tubular_min=tubular_min_str=tubular_min_str[dot_index+1:]
				name_dict[elem]="r"+tubular_min
			elif "rmax" in elem:
				fmt_digits=elem.split("_f")[1]
				total_chars,decimals=fmt_digits.split(".")
				fmt="%"+fmt_digits.replace("f","")+"f"
				flt=float(fmt %tubular_max)
				tubular_max=tubular_max_str=str(flt)#round(vxRes,decimals)
				
				if total_chars=="0":
					dot_index=tubular_max_str.find(".")
					tubular_max=tubular_max_str=tubular_max_str[dot_index+1:]
				name_dict[elem]=tubular_max
			elif "ravg" in elem:
				fmt_digits=elem.split("_f")[1]
				total_chars,decimals=fmt_digits.split(".")
				fmt="%"+fmt_digits.replace("f","")+"f"
				flt=float(fmt %tubular_avg)
				tubular_avg=tubular_avg_str=str(flt)#round(vxRes,decimals)
				
				if total_chars=="0":
					dot_index=tubular_avg_str.find(".")
					tubular_avg=tubular_avg_str=tubular_avg_str[dot_index+1:]
				name_dict[elem]="ravg"+tubular_avg
			elif "method" in elem:
				name_dict[elem]=methods[knot_index]
			elif 'sineconst' in elem and metaparameter_method['method']=='sinusoidal':
				#print(elem)
				fmt_digits=elem.split("_f")[1]
				total_chars,decimals=fmt_digits.split(".")
				fmt="%"+fmt_digits.replace("f","")+"f"
				sine_constant=float(fmt %eval(metaparameter_method['METAPARAM DEFORMER (CONSTANT SHIFT)']))
				if decimals=="0":
					sine_constant=int(sine_constant)
				sine_constant_str=str(sine_constant)#round(vxRes,decimals)
				#
				if total_chars=="0":
					dot_index=sine_constant_str.find(".")
					sine_constant=sine_constant_str=sine_constant_str[dot_index+1:]
				name_dict[elem]=""+sine_constant_str
			elif 'sineamp' in elem and metaparameter_method['method']=='sinusoidal':
				fmt_digits=elem.split("_f")[1]
				total_chars,decimals=fmt_digits.split(".")
				fmt="%"+fmt_digits.replace("f","")+"f"
				sine_amp=float(fmt %eval(metaparameter_method['METAPARAM DEFORMER (AMPLITUDE)']))
				if decimals=="0":
					sine_amp=int(sine_amp)
				sine_amp_str=str(sine_amp)
				if total_chars=="0":
					dot_index=sine_amp_str.find(".")
					sine_amp=sine_amp_str=sine_amp_str[dot_index+1:]
				#name_dict[elem]="±"+sine_amp_str
				name_dict[elem]=""+sine_amp_str
			elif 'sinefreq' in elem and metaparameter_method['method']=='sinusoidal':
				fmt_digits=elem.split("_f")[1]
				total_chars,decimals=fmt_digits.split(".")
				fmt="%"+fmt_digits.replace("f","")+"f"
				sine_freq=float(fmt %eval(metaparameter_method['METAPARAM DEFORMER (FREQUENCY)']))
				if decimals=="0":
					sine_freq=int(sine_freq)
				sine_freq_str=str(sine_freq)
				
				if total_chars=="0":
					dot_index=sine_freq_str.find(".")
					sine_freq=sine_freq_str=sine_freq_str[dot_index+1:]
				name_dict[elem]=""+sine_freq_str.zfill(2)
			elif 'sinephase' in elem:
				name_dict[elem]=""+metaparameter_method['METAPARAM DEFORMER (PHASE)']#+metaparameter_info[knot_index][]
			elif 'rch_its' in elem:
				if len(reach_iterations.keys())>0:
					name_dict[elem]="its"+str(reach_iterations[knot_index]).zfill(2)
				else:
					name_dict[elem]=""
			else:
				name_dict[elem]=eval(elem)

		#'index,knot_type,parameters,Nbars,Nvoxels'
	for key in [y for y in name_dict.keys() if y in name_keys]:
		name=name+name_dict[key]+name_separator
	name=name[:-1]
	if "__" in name:
		name=name.replace("__","_")
	return name
