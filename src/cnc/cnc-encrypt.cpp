#include <iostream>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <ctime>

#include "../include/aeslib.hpp"
#include "../include/genlib.hpp"

using namespace std;

void CNC(vector<byte *> &uData, vector<int> &uLens, vector<byte *> &uKeys, vector<byte *> &ciphers){

    int n;                      //variable to store the length of themessage in the loop
    byte expandedKey[176];      //expanded key variable, differs for every user key -> 44*4 = 176

    for(int i = 0; i < uData.size(); i++) {

        n = uLens[i];
        byte *cipher = new byte[n];
        
        KeyExpansion(uKeys[i], expandedKey);
        
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