#include <iostream>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <ctime>
#include <omp.h>

#include "../include/aeslib.hpp"
#include "../include/genlib.hpp"
#include "../include/parallelcpu.hpp"

using namespace std;

void CNC(vector<byte *> &uData, vector<int> &uLens, vector<byte *> &uKeys, vector<byte *> &ciphers){

    int n;                      //variable to store the length of themessage in the loop
    byte expandedKey[176];      //expanded key variable, differs for every user key -> 44*4 = 176

    for(int i = 0; i < uData.size(); i++) {

        n = uLens[i];
        byte *cipher = new byte[n];
        
        KeyExpansion(uKeys[i], expandedKey);

        /*
            According to the paper, the total number of parallel regions to be created is = max number of threads possible(N).
            After this in each of these parallel regions, I will have a private counting variable which counts from:
                    (message_len/N)*tid to (message_len/N)*(tid+1) in steps of 16

            I can run this command : cat /proc/cpuinfo | grep processor | wc -l , to get the max number of threads in a PC
        */

        int WORK_THREADS = 4, curr_index;
        omp_set_num_threads(WORK_THREADS); 

        #pragma omp parallel private(curr_index)
        {
            int tid = omp_get_thread_num();
            for(curr_index = (uLens[i]/WORK_THREADS)*tid ; curr_index<(uLens[i]/WORK_THREADS)*(tid+1) ; curr_index+=16){

                AddRoundKey(uData[i] + curr_index , expandedKey);

                for(int n_rounds = 1 ; n_rounds<=10 ; ++n_rounds)
                    Round(uData[i] + curr_index, expandedKey + (n_rounds*16), (n_rounds==10));
            }
        }
        
        /*
        #pragma omp parallel for 
        for(int curr_index = 0 ; curr_index<uLens[i] ; curr_index+=16){

            AddRoundKey(uData[i] + curr_index , expandedKey);

            for(int n_rounds = 1 ; n_rounds<=10 ; ++n_rounds){

                Round(uData[i] + curr_index, expandedKey + (n_rounds*16), (n_rounds==10));
            }
        }
        */

        /*
        blocksize = BLOCKSIZE;
        gridsize = ceil (uLens[i]/(BLOCKSIZE*16));
        
        if(uLens[i] <= BLOCKSIZE) gridsize = 1;

        Cipher <<< gridsize, blocksize>>> (d_message, n, d_expandedKey, d_sbox, d_mul2, d_mul3);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaMemcpy(cipher, d_message, n, cudaMemcpyDeviceToHost));
        ciphers.push_back(move(cipher));

        gpuErrchk(cudaFree(d_message));
        */
    }

}


int main(){

    vector<byte*> uData;
    vector<int> uLens;
    vector<byte*> uKeys;

    //get_data(vars, uData, uLens, uKeys, i, j);
    vector<byte*> ciphers;
            
    CNC(uData, uLens, uKeys, ciphers);

    return 0;
}