# Parallel AES Cryptography
![Github Issues](https://img.shields.io/github/issues/gurupunskill/parallel-aes.svg) 
![GitHub pull requests](https://img.shields.io/github/issues-pr/gurupunskill/parallel-aes.svg)
![GitHub](https://img.shields.io/github/license/gurupunskill/parallel-aes.svg)

A CUDA and OpenMP implementation of _`Fei.et.al Practical parallel AES algorithms on cloud for massive users andtheir performance evaluation`_.

## Introduction

Advanced Encryption Standard (AES) is a symmetric-key algorithm and is one of the strongest standards for electronic encryption in the world. It is widely used in software, hardware as well as in cloud applications. However, AES takes quite some time to encrypt messages especially when the number of users as well as the length of the text is quite large, which is the case in cloud enviroments.  
This is primarily because AES is implemented in a serial fashion. In the paper titled _`Fei.et.al Practical parallel AES algorithms on cloud for massive users andtheir performance evaluation`_ , the authors have brought out a parallel implementation for AES on the GPU as well as the CPU, which can be easily virtualised on cloud enviroments.  
AES is parallelisable as it is a symmetric block ciper and the encryption of each block is independent of the other blocks and as a result this can be done in a parallel fashion. The paper uses the concepts of _coalescing_ and _slicing_.
1. Coalescing: Putting together all the users' data together from the buffer to a contingous memory location.
2. Slicing: Dividing the coalesced data into equal parts so workload amongst threads is distributed evenly.  
  
The paper talks about 6 algorithms, namely:  

 * GCS &nbsp: GPU Coalescing and Slicing
 * GCNS : GPU Coalescing and no Slicing
 * GNC &nbsp: GPU no Coalescing and no Slicing
 * CCS &nbsp: CPU Coalescing and Slicing
 * CCNS : CPU Coalescing and no Slicing
 * CNC &nbsp: CPU no Coalescing and no Slicing  
  
All these algorithms have been implemented in this project and the results have been recorded, visualised and verified.  

## Instructions for Use

1. Open terminal and exceute _git clone https://github.com/gurupunskill/parallel-aes.git_ to clone the repository.

## File Structure



## Results