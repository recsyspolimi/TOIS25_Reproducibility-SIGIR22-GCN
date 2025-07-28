# Reproducibility and Artifact Consistency of the SIGIR 2022 Recommender Systems Papers Based on Message Passing - Source Code

This repository was developed by <a href="https://mauriziofd.github.io/" target="_blank">Maurizio Ferrari Dacrema</a>, Researcher at Politecnico di Milano. See our [website](http://recsys.deib.polimi.it/) for more information on our research group.
This reporsitory contains the source code of the article: _**"Reproducibility and Artifact Consistency of the SIGIR 2022 Recommender Systems Papers Based on Message Passing", ACM TOIS 2025**_. The paper is available on [ACM DL (TODO)](), [ArXiv](https://arxiv.org/abs/2503.07823).

Please cite our articles if you use this repository or our implementations of baseline algorithms [BibTex (TODO)](), remember also to cite the original authors if you use our porting of the algorithms based on Message Passing. 

## Full Results and Hyperparameters
The full results, hyperparameters configurations and search spaces for all algorithms based on Message Passing and the baselines are accessible [here](TOIS%202024%20-%20Reproducibility%20and%20Artifact%20Consistency%20of%20the%20SIGIR%202022%20Recommender%20Systems%20Papers%20Based%20on%20Message%20Passing%20-%20Additional%20Online%20Material.pdf).
For information on the requirements and how to install this repository, see the following [Installation](#Installation) section.

## Code organization
This repository is organized in several subfolders. 

#### Algorithms Based on Message Passing
All the papers we attempt to reproduce are in SIGIR2022 folder. For each algorithm there are two folders:
* A folder named "_github" which contains the full original repository, with the minor fixes needed for the code to run.
* A folder named "_our_interface" which contains the python wrappers needed to allow its testing in our framework. The main class for that algorithm has the "Wrapper" suffix in its name. This folder also contains the functions needed to read and split the data in the appropriate way.

#### Baseline Algorithms
In the folder "Recommenders" there are several other folders like "KNN", "GraphBased", "MatrixFactorization", "SLIM", "EASE_R", ... that contain all the baseline algorithms we used in our experiments.
The complete list of baselines, the details on all algorithms and references can be found [here](TOIS%202024%20-%20Reproducibility%20and%20Artifact%20Consistency%20of%20the%20SIGIR%202022%20Recommender%20Systems%20Papers%20Based%20on%20Message%20Passing%20-%20Additional%20Online%20Material.pdf).

#### Evaluation
The folder _Evaluation_ contains the evaluator object (_EvaluatorHoldout_) which compute all the metrics we report.

#### Data
The data to be used for each experiment is gathered from specific _DataReader_ objects within each SIGIR 2022 algorithm's folder. 
Those will load the original data split, if available. If not, automatically download the dataset and perform the split with the appropriate methodology. If the dataset cannot be downloaded automatically, a console message will display the link at which the dataset can be manually downloaded and instructions on where the user should save the compressed file.

The folder _Data_manager_ contains a number of _DataReader_ objects each associated to a specific dataset, which are used to read datasets for which we did not have the original split. 

Whenever a new dataset is downloaded and parsed, the preprocessed data is saved in a new folder called _Data_manager_split_datasets_, which contains a subfolder for each dataset. The data split used for the experimental evaluation is saved within the result folder for the relevant algorithm, in a subfolder _data_ . 

#### Hyperparameter Optimization
Folder _HyperparameterTuning_ contains all the code required to tune the hyperparameters of the baselines. 
The object _SearchBayesianSkopt_ does the hyperparameter optimization for a given recommender instance and hyperparameter space, saving the explored configuration and corresponding recommendation quality. 

## Run The Experiments 

See the following [Installation](#Installation) section for information on how to install this repository.
After the installation is complete you can run the experiments.

### Comparison with Baselines Algorithms

This repository contains one script for each of the 9 papers we attempt to reproduce. Each script starts with _run__, the conference name and the year of publication.
The scripts have the following boolean optional parameters (all default values are False except for the print-results flag):
* '--baseline_tune': Run baseline hyperparameter search
* '--article_default': Train the algorithm with the original hyperparameters
* '--article_tune': Run hyperparameter search for the graph algorithm (available only for GDE, RGCF)
* '--print_results': Generate the latex tables for this experiment


For example, if you want to run all the experiments for LightGCN, you should run this command:
```console
python run_SIGIR20_LightGCN_experiment.py --baseline_tune==True --article_default==True --print_results==True
```

If you want to run the optimization and comparison of all baselines and GNN methods, you should run this command:
```console
python run_ALL_comparison.py --baseline_tune==True --article_default==True --print_results==True
```
Note that the repository contains the same train-validation-test split that we used for the experiments reported in the paper.


## Installation
You can install the required environment directly on your local machine, or launch a Docker container to ensure a clean and isolated setup.

### Conda
We recommend using Conda to manage your environment.

    > ✅ Tested with Conda 25.3.1, Ubuntu 22.04.4 LTS.

1. Update Conda First

    Before proceeding, ensure that you are using a recent version of Conda.
    Older versions may take a very long time to resolve dependencies or even fail to complete the environment setup. 
    You can update Conda with:
    ```bash
    conda update -n base -c defaults conda
    ```

2. Install the Conda environment:
    ```bash
    conda env create -f environment.yml
    ```

    > ⚠️ **_NOTE:_** The environment requires *CUDA 11.6*.

3. Install system dependencies (for Cython)
    ```bash
    sudo apt install gcc python3-dev
    ```

4. Compile Cython Modules
    ```bash
    conda activate SIGIRReproducibility
    python3 run_compile_all_cython.py
    ```
   *Note: Warnings during compilation are expected.*

### Docker
You can also reproduce the environment using Docker, ensuring a consistent and portable setup.
To run docker with a GPU, you first need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

1. Build the Docker Image
    From the root of the repository (where the Dockerfile is located):
    ```bash
    sudo docker build -t SIGIRReproducibility-docker .
    ```
2. Start a Container
    ```bash
    sudo docker run --gpus all -it SIGIRReproducibility-docker 
    ```
   This launches an interactive session where all dependencies are already installed.

## License

This project is licensed under the AGPL-3.0 license. See the [LICENSE](LICENSE) file for details.

