# Benchmark for the EuLearn database

<p align="center">
<img width="400" alt="EuLearn Sample Image" src="DatabaseGenerator/EuLearn.png"/>
</p>
<p align="center">
  <em>Figure 1. Representative EuLearn surfaces spanning genera 0 through 10, with one sample surface shown for each genus.</em>
</p>

<br><br>

This repo contains the benchmark deep learning architectures that were trained and evaluated on the following database:

[EuLearn: A 3D database for learning Euler characteristics](https://huggingface.co/datasets/appliedgeometry/EuLearn)


### Architectures

The [Architectures](Architectures) folder contains the benchmark architectures.


### Database Generator

The [Database Generator](DatabaseGenerator) folder contains the software to generate the full database or any other surfaces required by the user.


### Sampling

The [Sampling](Sampling) folder contains point clouds with ~3000 vertices sampled from the original EuLearn database, which are the inputs for the [Architectures](Architectures).

<br><br>

<img width="1200" alt="Lissajous singular knot (4,5,7) phase sweeping animation" src="DatabaseGenerator/457phase.gif"/>
<p align="center">
  <em>Figure 2. Continuous deformation of the Lissajous singular knot (4,5,7) as the phase parameter in the z-coordinate is swept from 0 to 2π.</em>
</p>
