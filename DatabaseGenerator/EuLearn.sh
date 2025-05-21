#!/bin/bash

while getopts k:p: flag
do
    case "${flag}" in
        k) knots_file=${OPTARG};;
        p) parameters_file=${OPTARG};;
    esac
done

source $parameters_file
echo " "
echo `date +"%Y-%m-%d_%H-%M"`
echo "KNOTS FILE:" $knots_file
echo "EuLearn PARAMETERS FILE:" $parameters_file
if [[ $overwrite == *"False"* ]] ; then
		temporal_var=`date +"%Y-%m-%d_%H-%M"`
    	log_file=log_$temporal_var.log
    	output_dir=$output_dir"_"$temporal_var
fi
if [[ $overwrite == *"True"* && $compute_maxblowup == *"True"* ]] ; then
       rm -f $output_dir/$log_file
fi
echo "LOG FILE:" $output_dir/$log_file
echo " "
export output_dir=$output_dir
export bup_dir="Max_BlowUP_Computation"
export merged_STL_dir="Merged"
export fix_Normals_dir="FixedNormals"
export post_processing_dir="Post-Processing"
export fields_dir="mq_Fields"
export scaled_translated_dir=$post_processing_dir/"ScaleTranslationNS"
#export nsmooth_dir="STL"
export nsmooth_dir=$scaled_translated_dir

mkdir -p $output_dir


    if [[ $compute_maxblowup == *"True"* ]] ; then
        if [[ $method == *"manual"* ]] ; then
    	python cmbp_method.py --method $method --k_nodes $knot_nodes --min_voxels $min_voxels_per_Axis --bboffset $boundingbox_offset --r_neigh $r_neighborhood --pr "$show_progress" --out_dir "$output_dir" --kf "$knots_file" --log $log_file --ascii_names $ascii_filenames

        elif [[ $method == *"minimal"* ]] ; then
        python cmbp_method.py --method $method --k_nodes $knot_nodes --min_voxels $min_voxels_per_Axis --bboffset $boundingbox_offset --pr "$show_progress" --out_dir "$output_dir" --kf "$knots_file" --log $log_file --ascii_names $ascii_filenames --vx_ActDistSc $voxel_activation_distance_scale

        elif [[ $method == *"sinusoidal"* ]]; then
        python cmbp_method.py --method $method --k_nodes $knot_nodes --min_voxels $min_voxels_per_Axis --bboffset $boundingbox_offset --pr "$show_progress" --out_dir "$output_dir" --kf "$knots_file" --log $log_file --ascii_names $ascii_filenames --sine_const $sine_const --sine_amp $sine_amp --sine_freq $sine_freq --sine_phase $sine_phase --vx_ActDistSc $voxel_activation_distance_scale

        elif [[ $method == *"automatic"* ]] ; then
        python cmbp_method.py --method $method --bboffset $boundingbox_offset --pr "$show_progress" --out_dir "$output_dir" --kf "$knots_file" --log $log_file --ascii_names $ascii_filenames --bpb $blocks_per_batch --alt $arclength_tolerance --reachrepo $reach_scripts --auto_submethod $automatic_submethod --reach_threshold $reach_threshold --reach_iterations $reach_iterations 

        else
            echo "UNKNOWN METHOD"
        fi

    fi
	#
    if [[ $pump == *"True"* ]] ; then
        if [[ $method == *"manual"* ||  $method == *"minimal"* ||  $method == *"sinusoidal"* ]] ; then
            python cmbp_pump.py --method $method --k_nodes $knot_nodes --min_voxels $min_voxels_per_Axis --bboffset $boundingbox_offset --bpb $blocks_per_batch --pr "$show_progress" --out_dir "$output_dir" --kf "$knots_file" --log $log_file --ascii_names $ascii_filenames
        elif [[ $method == *"automatic"* ]] ; then
            python cmbp_pump.py --method $method --bboffset $boundingbox_offset --bpb $blocks_per_batch --pr "$show_progress" --out_dir "$output_dir" --kf "$knots_file" --log $log_file --ascii_names $ascii_filenames --alt $arclength_tolerance --reachrepo $reach_scripts --auto_submethod $automatic_submethod
        else 
            echo "UNKNOWN METHOD (pump)"
        fi

    fi
	#
    if [[ $marchingQ == *"True"* ]] ; then
    	python mq_marchingQ.py --out_dir "$output_dir"  --pr "$show_progress" --kf "$knots_file" --vtx_split $max_STL_vertices --log $log_file --ascii_names $ascii_filenames
    fi
	#
    if [[ $merge_STL == *"True"* ]] ; then
        mkdir -p $output_dir/$merged_STL_dir
        rm -rf $merged_STL_dir/*
        export name_format=$name_format
        export knots_file=$knots_file
        export ascii_filenames=$ascii_filenames
        export log_file=$log_file
    	$blender_route/blender  --background  --python bpy_mergeSTLs.py 
    fi
	#
    if [[ $fix_normals_STL == *"True"* ]] ; then
        mkdir -p $output_dir/$fix_Normals_dir
        rm -rf $fix_Normals_dir/*
    	$blender_route/blender  --background --python bpy_fixorientationSTL.py
    fi
    #
    if [[ $scale == *"True"* ||  $translate_to_origin == *"True"* || $smooth == *"True"* ]] ; then
        mkdir -p $output_dir/$post_processing_dir
        rm -rf $output_dir/$post_processing_dir/*
        export translate_to_origin=$translate_to_origin
        export smooth=$smooth
        export scale=$scale
        export log_file=$log_file
        export name_format=$name_format
        export knots_file=$knots_file
        export ascii_filenames=$ascii_filenames
        if [[ $smooth == *"True"* ]] ; then 
            export smooth_iterations=$smooth_iterations
            export smooth_factor=$smooth_factor
        fi
        $blender_route/blender  --background --python bpy_SmoothCenterScale.py
        #
        mkdir -p $output_dir/$scaled_translated_dir/
        export post_processing_dir=$scaled_translated_dir
        #rm -rf $output_dir/$post_processing_dir/*
        export translate_to_origin=$translate_to_origin
        export smooth="False"
        export scale=$scale
        #export log_file=$log_file
        #export name_format=$name_format
        #export knots_file=$knots_file
        #export ascii_filenames=$ascii_filenames
        $blender_route/blender  --background --python bpy_SmoothCenterScale.py
        export post_processing_dir="Post-Processing"
    fi
    #
    if [[ $generate_summary == *"True"* ]] ; then
        #python summary_gen.py --log $log_file --fxnormals $fix_Normals_dir --out_dir "$output_dir" --ascii_names $ascii_filenames --kf "$knots_file" --copy_from $post_processing_dir --name_format "$name_format" --fields_dir "$fields_dir" --nsmooth_dir "$nsmooth_dir"
        python summary_gen.py --log $log_file --fxnormals $fix_Normals_dir --out_dir "$output_dir" --ascii_names $ascii_filenames --kf "$knots_file" --copy_from $post_processing_dir --name_format "$name_format" --fields_dir "$fields_dir" --nsmooth_dir "$nsmooth_dir" --bup_dir "$bup_dir"
    fi


echo " "
echo "EuLearn - done: "`date +"%Y-%m-%d_%H-%M"`

