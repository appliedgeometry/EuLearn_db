@echo off

set python_route="C:\Users\my_user\anaconda3\python.exe"

set Arg1=%1
set Arg3=%3


if %Arg1%==-k (
  set knots_file=%2
  set parameters_file=%4)
if %Arg3%==-k (
  set knots_file=%4
  set parameters_file=%2)

for /F "tokens=1,2 EOL=# delims='='" %%a in (%parameters_file%) do (
  set %%a=%%b
)
::  echo %%a, %%b

echo " "
echo inicio: %DATE% %time%
echo " "
echo KNOTS FILE: %knots_file%
echo EuLearn PARAMETERS FILE: %parameters_file%

echo LOG FILE: %log_file%

set output_dir=%output_dir%


set bup_dir=Max_BlowUP_Computation
set Merged=Merged
set FixedNormals=FixedNormals
set Post-Processing=Post-Processing
set fields_dir=mq_Fields
::set nsmooth_dir=STL
set scaled_translated_dir=%Post-Processing%/"ScaleTranslationNS"
set nsmooth_dir=%scaled_translated_dir%

set merged_STL_dir=%Merged%
set fix_Normals_dir=%FixedNormals%

set post_processing_dir=%Post-Processing%


:: Added
set log_file=%log_file%
set name_format=%name_format%
set export knots_file=%knots_file%
set export ascii_filenames=%ascii_filenames%
set translate_to_origin=%translate_to_origin%
set smooth=%smooth%
set scale=%scale%
:: END of Added


if not exist %output_dir% mkdir %output_dir%


:: Continue to MaxBlowUp process
if %compute_maxblowup%==True (
  if %method%==manual (
    %python_route% .\cmbp_method.py --method %method% --k_nodes %knot_nodes% --min_voxels %min_voxels_per_Axis% --bboffset %boundingbox_offset% --r_neigh %r_neighborhood% --pr %show_progress% --out_dir %output_dir% --kf %knots_file% --log %log_file% --ascii_names %ascii_filenames%)
  if %method%==minimal (
    %python_route% .\cmbp_method.py --method %method% --k_nodes %knot_nodes% --min_voxels %min_voxels_per_Axis% --bboffset %boundingbox_offset% --pr %show_progress% --out_dir %output_dir% --kf %knots_file% --log %log_file% --ascii_names %ascii_filenames% --vx_ActDistSc %voxel_activation_distance_scale%)
  if %method%==sinusoidal (
    %python_route% .\cmbp_method.py --method %method% --k_nodes %knot_nodes% --min_voxels %min_voxels_per_Axis% --bboffset %boundingbox_offset% --pr %show_progress% --out_dir %output_dir% --kf %knots_file% --log %log_file% --ascii_names %ascii_filenames% --sine_const %sine_const% --sine_amp %sine_amp% --sine_freq %sine_freq% --sine_phase %sine_phase% --vx_ActDistSc %voxel_activation_distance_scale%)
  if %method%==automatic (
    %python_route% .\cmbp_method.py --method %method% --bboffset %boundingbox_offset% --pr %show_progress% --out_dir %output_dir% --kf %knots_file% --log %log_file% --ascii_names %ascii_filenames% --bpb %blocks_per_batch% --alt %arclength_tolerance% --reachrepo %reach_scripts% --auto_submethod %automatic_submethod% --reach_threshold %reach_threshold% --reach_iterations %reach_iterations%))
  
:: Continue to Pump process
if %pump%==True (
  if %method%==manual (
    %python_route% .\cmbp_pump.py --method %method% --k_nodes %knot_nodes% --min_voxels %min_voxels_per_Axis% --bboffset %boundingbox_offset% --bpb %blocks_per_batch% --pr %show_progress% --out_dir %output_dir% --kf %knots_file% --log %log_file% --ascii_names %ascii_filenames%)
  if %method%==minimal (
    %python_route% .\cmbp_pump.py --method %method% --k_nodes %knot_nodes% --min_voxels %min_voxels_per_Axis% --bboffset %boundingbox_offset% --bpb %blocks_per_batch% --pr %show_progress% --out_dir %output_dir% --kf %knots_file% --log %log_file% --ascii_names %ascii_filenames%)
  if %method%==sinusoidal (
    %python_route% .\cmbp_pump.py --method %method% --k_nodes %knot_nodes% --min_voxels %min_voxels_per_Axis% --bboffset %boundingbox_offset% --bpb %blocks_per_batch% --pr %show_progress% --out_dir %output_dir% --kf %knots_file% --log %log_file% --ascii_names %ascii_filenames%)
  if %method%==automatic (
    %python_route% .\cmbp_pump.py --method %method% --bboffset %boundingbox_offset% --bpb %blocks_per_batch% --pr %show_progress% --out_dir %output_dir% --kf %knots_file% --log %log_file% --ascii_names %ascii_filenames% --alt %arclength_tolerance% --reachrepo %reach_scripts% --auto_submethod %automatic_submethod%))

:: Continue to MarchingQ process
if %marchingQ%==True (
    %python_route% .\mq_marchingQ.py --out_dir %output_dir%  --pr %show_progress% --kf %knots_file% --vtx_split %max_STL_vertices% --log %log_file% --ascii_names %ascii_filenames%)

:: Continue to Merge STL
if %merge_STL%==True (
     if not exist "%output_dir%\%merged_STL_dir%" mkdir "%output_dir%\%merged_STL_dir%"
    "%blender_route%\blender.exe" --background --python .\bpy_mergeSTLs.py)

:: Continue to Fix Orientation 
if %fix_normals_STL%==True (
    if not exist "%output_dir%\%FixedNormals%" mkdir "%output_dir%\%FixedNormals%"
    "%blender_route%\blender.exe" --background --python .\bpy_fixorientationSTL.py)

:: Continue to SmoothCenterScale
if %smooth%==True (
  set smooth_factor="%smooth_factor%"
  set smooth_iterations="%smooth_iterations%")
if %scale%==True (
  if not exist "%output_dir%\%post_processing_dir%" mkdir "%output_dir%\%post_processing_dir%"
  "%blender_route%\blender.exe" --background --python .\bpy_SmoothCenterScale.py
  if not exist "%output_dir%\%scaled_translated_dir%" mkdir "%output_dir%\%scaled_translated_dir%"
    set translate_to_origin="True"
    set smooth="False"
    set scale="True"
    set post_processing_dir=%scaled_translated_dir%
    "%blender_route%\blender.exe" --background --python .\bpy_SmoothCenterScale.py
    set post_processing_dir=%Post-Processing%)

:: Continue to summary generation
if %generate_summary%==True (
  %python_route% .\summary_gen.py --log %log_file% --fxnormals %fix_Normals_dir% --out_dir %output_dir% --ascii_names %ascii_filenames% --kf %knots_file% --copy_from %post_processing_dir% --name_format %name_format%  --fields_dir %fields_dir% --nsmooth_dir %nsmooth_dir% --bup_dir %bup_dir%)


echo done: %DATE% %time%