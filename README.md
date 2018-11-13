# Parallel AES Cryptography
![Github Issues](https://img.shields.io/github/issues/gurupunskill/parallel-aes.svg) 
![GitHub pull requests](https://img.shields.io/github/issues-pr/gurupunskill/parallel-aes.svg)
![GitHub](https://img.shields.io/github/license/gurupunskill/parallel-aes.svg)

A CUDA and OpenMP implementation of [_`Fei.et.al Practical parallel AES algorithms on cloud for massive users and their performance evaluation`_](https://doi.org/10.1002/cpe.3734).

## Introduction
Advanced Encryption Standard (AES) is a symmetric-key algorithm and is one of the strongest standards for electronic encryption in the world. It is widely used in software, hardware as well as in cloud applications. However, AES takes quite some time to encrypt messages especially when the number of users as well as the length of the text is quite large, which is the case in cloud enviroments.  
This is primarily because AES is implemented in a serial fashion. In the paper titled _`Fei.et.al Practical parallel AES algorithms on cloud for massive users andtheir performance evaluation`_ , the authors have brought out a parallel implementation for AES on the GPU as well as the CPU, which can be easily virtualised on cloud enviroments.  
AES is parallelisable as it is a symmetric block ciper and the encryption of each block is independent of the other blocks and as a result this can be done in a parallel fashion. The paper uses the concepts of _coalescing_ and _slicing_.
1. **Coalescing** : Putting together all the users' data together from the buffer to a contingous memory location.
2. **Slicing** : Dividing the coalesced data into equal parts so workload amongst threads is distributed evenly.  
  
The paper defines 6 algorithms, namely:  
 * `GCS`  : GPU Coalescing and Slicing
 * `GCNS` : GPU Coalescing and no Slicing
 * `GNC`  : GPU no Coalescing and no Slicing
 * `CCS`  : CPU Coalescing and Slicing
 * `CCNS` : CPU Coalescing and no Slicing
 * `CNC`  : CPU no Coalescing and no Slicing  
  
All these algorithms have been implemented in this project and the results have been recorded, visualised and verified.  

## Dependencies
1. [`g++`](https://askubuntu.com/questions/481807/how-to-install-g-in-ubuntu-14-04) or any other standardized C++ compiler
2. A NVIDIA GPU for CUDA based algorithms
3. [`nvcc`](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions), from the CUDA development kit
4. `OpenMP`, for CPU parallelisation

## Instructions for Use
1. Clone this repository `git clone https://github.com/gurupunskill/parallel-aes.git`.
2. Change directory to `parallel-aes/src`.
    1. Change directory to `generator`.
    2. Run `g++ generate.cpp` and `./a.out`. This creates a dataset which will be used by the algorithms.
    3. Change directory to any one of the 7 implemented AES algorithm. For instance, `gcs` would hold the GCS algorithm.
    4. Run `g++ <algo-name>.cpp` and `./a.out`. You will view the corresponding times on the console.
    5. Change directory to `prop_dataset` to view the ciper texts in your text editor.

## File Structure
* [`docs`](docs) : All the documentation for the project is present in this folder.
    * [`img`](docs/img) : Images of graphs showing results of each algorithm are present here.
    * [`AES-explanatory-paper.pdf`](docs/AES-explanatory-paper.pdf) : The paper that was used to learn AES from.
    * [`Fei_et_al-2016-Concurrency_and_Computation%3A_Practice_and_Experience.pdf`](docs/Fei_et_al-2016-Concurrency_and_Computation%253A_Practice_and_Experience.pdf) : The research paper being implemented.
    * [`NIST FIPS AES.pdf`](parallel-aes/docs/NIST%20FIPS%20AES.pdf) : The paper that was used to learn AES from.
* [`src`](src) : Source code for the project is present in this folder
    * [`ccns`](src/ccns) : Source code for the implementation of CCNS algorithm.
    * [`ccs`](src/ccs) : Source code for the implementation of CCS algorithm.
    * [`cnc`](src/cnc) : Source code for the implementation of CNC algorithm.
    * [`gcns`](src/gcns) : Source code for the implementation of GCNS algorithm.
    * [`gcs`](src/gcs) : Source code for the implementation of GCS algorithm.
    * [`generator`](src/generator) : Source code for producing a normalised random dataset which is used for encryption tests and analysis.
    * [`gnc`](src/gnc) : Source code for the implementation of GNC algorithm.
    * [`include`](src/include) : Contains the source code for the header files defined by us.
        * [`aeslib.hpp`](src/include/aeslib.hpp) : Source code for the header file containing AES code.
        * [`genlib.hpp`](src/include/genlib.hpp) : Source code for the header file containing general common functions.
        * [`paralellcore.cuh`](src/include/parallelcore.cuh) : Source code for the header file containing functions and kernels for CUDA implementation of AES.
        * [`parallelcpu.hpp`](src/include/parallelcpu.hpp) : Source code for the header file containing unctions for OpenMP implementation of AES.
    * [`norm_dataset`](src/norm_dataset) : The normalised dataset along with cipher texts (this file will be present once code is run on local repository)
    * [`sequential`](src/sequential) : Source code for the implementation of serial AES algorithm.
    * [`final_data.csv`](src/final_data.csv) : CSV file where results are dumped into. Used for plotting graphs.
* [`.gitignore`](.gitignore) : Files to be ignored during commits by git.
* [`License`](LICENSE) : MIT license
* [`README.md`](README.md) : Document giving a brief overview of the project. 


## Software Tools, Languages and Frameworks Used
1. `C++` : C++ was the primary programming language used for implementing the algorithms.
2. `python` : The Python programming language was used to visulaise the results by plotting graphs.
3. `CUDA` : CUDA platform was used to implement the three GPU algorithms as it provides architecture for parallel computing on the GPU.
4. `OpenMP` : OpenMP platform was used to implement the three CPU algorithms as it provides architecture for parallel computing on the CPU.
5. `VSCode` : VSCode was the primary text editor used.
6. `github` : Github was used for collaboration and as a version control system.


## Results
![alt text](docs/img/Comparing-Algorithms.png)

The code was executed and tested with a uniformly random dataset of 100 files to 1000 files. Each file had random data from 30KB to 150KB generated using a pseudo random generator.  

The results are concurrent with that of those published in the paper.  

## Contributors
1. Gurupungav Narayanan 16CO128 
2. Nihal Haneef 16CO114  
