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

       int WORK_THREADS = 4;
        omp_set_num_threads(WORK_THREADS);
        
        #pragma omp parallel for 
        for(int curr_index = 0 ; curr_index<uLens[i] ; curr_index+=16){

            AddRoundKey(uData[i] + curr_index , expandedKey);
            for(int n_rounds = 1 ; n_rounds<=10 ; ++n_rounds)
                Round(uData[i] + curr_index, expandedKey + (n_rounds*16), (n_rounds==10));
        }
        

       //omp_set_num_threads(4);
       /*#pragma omp parallel for
       for(int j = 1 ; j <= 4 ; ++j){
           for(int curr_index = ((uLens[i]/4)*(j-1)); curr_index < ((uLens[i]/4)*j) ; curr_index+=16){

               cout <<  j << endl;
               AddRoundKey(uData[i] + curr_index , expandedKey);
               for(int n_rounds = 1 ; n_rounds<=10 ; ++n_rounds)
                    Round(uData[i] + curr_index, expandedKey + (n_rounds*16), (n_rounds==10));
           }
       }*/

        /*
       for(int curr_index = 0; curr_index<uLens[i] ; curr_index+=16){

            AddRoundKey(uData[i] + curr_index , expandedKey);

            for(int n_rounds = 1 ; n_rounds<=10 ; ++n_rounds)
                Round(uData[i] + curr_index, expandedKey + (n_rounds*16), (n_rounds==10));
        }
        */

       /*
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
        */

        cipher = uData[i];
        ciphers.push_back(move(cipher));
    }
}

int main(){

    vector<byte*> uData;
    vector<int> uLens;
    vector<byte*> uKeys;
    vector<byte*> ciphers;

    byte message[] = {0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34, 0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34, 0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34};
    byte key[] = {0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};

    uData.push_back(move(message));
    uLens.push_back(48);
    uKeys.push_back(move(key));

    //get_data(vars, uData, uLens, uKeys, i, j);
    
    CNC(uData, uLens, uKeys, ciphers);
    cout << hex(ciphers[0], 48) << endl;

    return 0;
}

/*
    byte message[] = {0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34, 0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34, 0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34};
    byte key[] = {0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
*/