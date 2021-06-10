Experiments for "Context-Aware Online Collective Inference for Templated Graphical Models".

### Simple Execution
Assuming all requirements are met, all experiments can be run using the `run.sh` script.

```
./run.sh
```

### Requirements
These experiments expect that you are running on a POSIX (Linux/Mac) system.
The specific application dependencies are as follows:
 - Bash >= 4.0
 - Java >= 7
 - Postgres >= 9.5

#### Postgres and PSL
The Postgres website has all the directions you should need for [installing Postgres.](https://www.postgresql.org/download/)
Once you have Postgres installed, you need to setup a database and user.

##### Database
Creating a database in Postgres can be done with the createdb command. 
Call your database psl.
```createdb psl```

##### User
Creating a user in Postgres can be done with the createuser command. 
It is easiest to create your psl user as a superuser (`-s`).
```createuser -s jay```

##### Configuration
These are fully optional and PSL will run fine without them. 
They were used to run experiments in the ICML21 
Context-Aware Online Collective Inference for Templated Graphical Models 
paper.

```
# shared_buffers = 1/4 of system memory
# effective_cache_size = 1/2 of system memory
# For example, on a 16GB machine:
shared_buffers = 4GB
effective_cache_size = 8GB

# No Durability
fsync = off
full_page_writes = off
synchronous_commit = off

# Query Optimization
# Disable nested loops.
enable_nestloop = off

# Limit the write-ahead log (WAL)
# max_wal_size = 1/2 of shared_buffers
max_wal_size = 2GB
wal_buffers = 32MB
wal_level = minimal
max_wal_senders = 0
checkpoint_timeout = 30
```

### Advanced Execution
If you want to modify the experiment procedures or parameters then start with the `run.sh` script.
There are two main steps:
 - Setup the psl example scripts and models using the `scripts/setup_psl_examples.sh` script.
 - Run the inference ,regret, and stability experiments scripted in the `scripts/` directory.
     - The Movielens and Bikeshare experiments are scripted in the `scripts/run_atom_update_experiments.sh`
     - The Epinions experiments are scripted in the `scripts/run_template_modification_experiments.sh`


### Result Analysis
The jupyter notebook `scripts/parselogs.ipynb` will run the analysis necessary to reproduce the plots in the paper.
This notebook assumes all of the experiments have been run and the results are in the base directory of this repository.


### Data Construction
By default, the experiment scripts will fetch the PSL formatted data. 
However, if you would like to rerun the data construction step using the raw datasets, 
you can do so with the `construct.sh` script for each online psl example.
```
./online-psl-examples/movielens-1m/scripts/construct.sh
```
```
./online-psl-examples/bikeshare/scripts/construct.sh
```
```
./online-psl-examples/epinions/scripts/construct.sh
```
If you would like to modify the data construction process, 
then you should start with the `construct.py` file in the `scripts/data-construction` directory 
for the online psl example you are interested in reconstructing.

### Citation
All of these experiments are discussed in the following paper:

```
@conference{dickens:icml21,
	author = {Charles Dickens* and Connor Pryor* and Eriq Augustine and Alex Miller and Lise Getoor},
	title = {Context Aware Online Collective Inference for Templated Graphical Models},
	booktitle = {ICML International Conferernce on Machine Learning},
	year = {2021},
}
```
