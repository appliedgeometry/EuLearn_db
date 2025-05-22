# EuLearn

This is a GPU-based, Linux, Windows and Google Colab compatible, open-source software package to generate a diversity of **controlled genus surfaces** in STL format by constructing tubular neighborhoods around closed curves immersed in $\mathbb{R}^3$. These curves come from **knot parameterizations** where the 1D circle $S^1$ is mapped to $\mathbb{R}^3$ via a function $K(t)$, where $t\in[0,2\pi]$.

<img width="404" alt="EuLearn Sample Image" src="EuLearn.png"/>

The [`EuLearn_notebook.ipynb`](EuLearn_notebook.ipynb) can be used to replicate the [EuLearn database](https://huggingface.co/datasets/appliedgeometry/EuLearn) or generate completely new examples defined by the user. These examples consist of 15 knot parameterizations for each genus from 0 to 10, with 20 cosine-dependent radius variations, which totals 3,300 surfaces with their respective scalar field and blow-up profile, and an additional smoothed surface.

The available parameterizations include some basic knots:
- Unknot (circle)
- Trefoil
- Eight

And the knot families:
- Lissajous
- Fibonacci

The most versatile is the **Lissajous** family, whose parameterization is:

$$K(t) = \big(\ \cos(n_xt + \phi_x)\ ,\ \cos(n_yt + \phi_y)\ ,\ \cos(n_zt + \phi_z)\ \big)\ ,$$

where $n_x,n_y,n_z$ are positive integers that determine the frequency of the cosine function in each coordinate, and $\phi_x,\phi_y,\phi_z$ are real numbers within the $[0,2\pi]$ interval that produce a phase shift in each coordinate. Fractions of $\pi$ are specially relevant as values for the phase shift.

All knots provide genus 1 surfaces once the tubular neighborhood is constructed.

There are two ways to modify the implicit genus 1 knot parameterizations:
1. Allow for singular double-points in the 1D parameterization, resulting in **singular knots**.
2. Attach secant segments to $S^1$.

The software requires as input:
- Type of knot parameterization
- Frequencies $n_x,n_y,n_z$
- Phase shifts $\phi_x,\phi_y,\phi_z$
- The list `[(p1,p2), (p3,p4), ...]` of endpoints in $[0,2\pi]$, where each pair of endpoints defines a secant segment attached to $S^1$

In the following fashion:

`knot_type, nx,ny,nz, ϕx,ϕy,ϕz, [(p1,p2), (p3,p4), ...]`

For example, if I want to generate a surface around the Lissajous knot 

$K(t) = \big(\ \cos(2t)\ ,\ \cos\left(3t + \dfrac{\pi}{2}\right)\ ,\ \cos(5t)\ \big)\ ,$ 

which has 6 singular points and thus produces a genus 7 surface, and I don't want to attach segments, then I have to create a text file, say `knot.csv`, with the following content:
```
lissajous,2,3,5,0,pi2,0,[]
```
The empty brackets are required if no segments are added. The `pi2` notation represents $\frac{\pi}{2}$, as well as `pi3` represents $\frac{\pi}{3}$, and so on.

If I want to generate in the same round the unknot, trefoil knot, and eight knot with two attached segments each, say from $0$ to $\pi$ and from $\frac{\pi}{2}$ to $\frac{3\pi}{2}$, then I write:
```
circle,0,0,0,0,0,0,[(0,np.pi),(np.pi/2,3*np.pi/2)]
trefoil,0,0,0,0,0,0,[(0,np.pi),(np.pi/2,3*np.pi/2)]
eight,0,0,0,0,0,0,[(0,np.pi),(np.pi/2,3*np.pi/2)]
```
For the unknot, the knot type is `circle`. The unknot, trefoil knot, and eight knot follow a different parameterization than Lissajous, so we zero out the frequencies and phase shifts, although they actually have specific frequencies:

- Unknot (circle): $\ U(t) = \big(\ \cos(t)\ ,\ \sin(t)\ ,\ 0\ \big)$

- Trefoil: $\ T(t) = \big(\ \sin(2t) + 2\sin(2t)\ ,\ \cos(2t) - 2\cos(2t)\ ,\ -\sin(3t)\ \big)$

- Eight: $\ E(t) = \Big(\ \big(2+\cos(2t)\big)\cos(3t)\ ,\ \big(2+\cos(2t)\big)\sin(3t)\ ,\ \sin(4t)\ \Big)$

While the **Fibonacci** family is compatible with the Lissajous frequencies and phase shifts according to:

$$\ F(t) = \big(\ \cos(n_xt + \phi_x)\ ,\ \cos(n_yt + \phi_y)\ ,\ 0.5\sin(n_yt + \phi_y) + 0.5\cos(n_zt + \phi_z)\ \big)$$

The frequencies are supposed to follow triplets of the Fibonacci sequence: $1,1,2,3,5,8,13,21,34,55,...$, say $(1,1,2)$, $(1,2,3)$, $(2,3,5)$, $(3,5,8)$, but from $5,8,13$ onwards the curve is very long and complex already. If you have a brave spirit, playing with positive integers outside the Fibonacci sequence result in beautiful figures, though. The drawback is that these curves have an exponential growth, requiring very thin tubular neighborhoods and they are memory-expensive. So sticking with frequencies less than 10 might be recommendable.


The [`knot_batch_01.csv`](knot_batch_01.csv) file contains a sample list of knots with segments.


### On Linux or Google Colab run:
```
./EuLearn.sh -k knot_file -p parameters_file
```

### On Windows run:
```
.\EuLearn_win.cmd -k knot_file -p parameters_file
```
The 2nd line of `\EuLearn_win.cmd` needs to be updated with the current Python path:

`set python_route="C:\Users\**my_user**\anaconda3\python.exe"`

replacing `**my_user**`.

Also, the parameter file `paramaters.par` may require to edit the `System Config` section with the Blender installation path on Windows, typically:

`blender_route='C:\Program Files\Blender Foundation\Blender 4.0'`


## Dependencies
```
pip install -r requirements.txt
```
Or more explicitly:
```
pip install pycuda python-dev-tools setuptools numpy numpy-stl scikit-image scipy sympy trimesh tqdm
```
For Anaconda, we additionally require:
```
pip install libboost python-setuptools
```
Or if you prefer to run on Colab, the [`EuLearn_notebook.ipynb`](EuLearn_notebook.ipynb) includes all dependencies.

Install Blender version 3.0 up to 4.1 (for the Post-Processing section, not mandatory). To download and install (decompress) Blender on Linux:
```
wget https://ftp.halifax.rwth-aachen.de/blender/release/Blender4.0/blender-4.0.0-linux-x64.tar.xz
tar -xvf blender-4.0.0-linux-x64.tar.xz
```

## Parameters

The parameter files contain the following data:

#### EuLearn Config

| **Config Option**                    | **Notes** |
|-------------------------------------|--------------------------|
| show_progress=True                  |                          |
| max_STL_vertices=250                |                          |
| knot_nodes=500                      | Important for automatic method, keep low |
| boundingbox_offset=5                | To avoid trimming, often needs to increase |
| min_voxels_per_Axis=90              | Typically 50 is enough and 200 is heavy    |
| voxel_activation_distance_scale=1.0 |                          |

#### Methods

It is recommended to comment with `#` the methods that are not used, to have them as reference when switching between methods.

| **Config Option**        | **Notes** |
|--------------------------|--------------------------|

| method=minimal           | Calculates minimal constant radius|
|--------------------------|--------------------------|

| method=manual            |                          |
|--------------------------|--------------------------|
| r_neighborhood=0.1       | Manual constant radius   |

| method=sinusoidal        |                          |
|--------------------------|--------------------------|
| sine_const=1.0           |                          |
| sine_amp=0.1             |                          |
| sine_freq=1              | This determines the radius variation |
| sine_phase=0             |                          |

| method=automatic         |                          |
|--------------------------|--------------------------|
| automatic_submethod=reach|                          |
| reach_iterations=5       | Typically between 1 and 15, more iterations increase radius |
| reach_threshold=1e-16    |                          |
| arclength_tolerance=0.01 |                          |

#### System Config

| **Config Option**                      | **Notes** |
|----------------------------------------|--------------------------|
| blocks_per_batch=10                    |                          |
| overwrite=True                         |                          |
| output_dir='output'                    |                          |
| log_file='output.log'                  |                          |
| ascii_filenames=True                   |                          |
| blender_route='../blender-3.5.0-linux-x64/'| Important to check correct path |
| reach_scripts='./AlcanceNudos/'        |                          |
| name_format='type,parameters,rmin_f1.3,rmax_f1.3,Nvoxels' | Pick from the full list in the sample parameter files `*.par`|

In `name_format`, the suffix `f1.3` for min, avg and max radius, for example `rmin_f1.3`, stands for "at least 1 character wide" integer part followed by 3 decimals.

#### Select Processes

| **Config Option**        | **Description / Notes** |
|--------------------------|--------------------------|
| compute_maxblowup=True   |                          |
| pump=True                |                          |
| marchingQ=True           |                          |
| merge_STL=True           | Deprecated               |
| fix_normals_STL=True     |                          |

#### Blender Post-Processing

| **Config Option**        | **Description / Notes** |
|--------------------------|--------------------------|
| scale=True               |                          |
| translate_to_origin=True |                          |
| smooth=True              |                          |
| smooth_factor=0.25       |                          |
| smooth_iterations=25     |                          |
| generate_summary=True    |                          |
