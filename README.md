# Benchmark for the EuLearn database

<img width="404" alt="EuLearn Sample Image" src="DatabaseGenerator/EuLearn.png"/>

This repo contains the benchmark deep learning architectures that were trained and evaluated on the following database:

[EuLearn: A 3D database for learning Euler characteristics](https://huggingface.co/datasets/appliedgeometry/EuLearn)


### Architectures

The [Architectures](Architectures) folder contains the benchmark architectures.


### Database Generator

The [Database Generator](DatabaseGenerator) folder contains the software to generate the full database or any other surfaces required by the user.


### Sampling

The [Sampling](Sampling) folder contains point clouds with ~3000 vertices sampled from the original EuLearn database, which are the inputs for the [Architectures](Architectures).
