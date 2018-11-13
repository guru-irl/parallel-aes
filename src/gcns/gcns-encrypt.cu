#include <iostream>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <cuda.h>
#include <vector>
#include <chrono> 

#include "../include/aeslib.hpp"
#include "../include/genlib.hpp"
#include "../include/parallelcore.cuh"

using namespace std;

void GCNS(vector<byte *> &uData, vector<int> &uLens, vector<byte *> &uKeys, vector<byte *> &ciphers) {
    
    byte *d_sbox;
    byte *d_mul2;
    byte *d_mul3;
    byte *d_rcon;

    CUDA_ERR_CHK(cudaMalloc((void **) &d_sbox, 256));
    CUDA_ERR_CHK(cudaMalloc((void **) &d_mul2, 256));
    CUDA_ERR_CHK(cudaMalloc((void **) &d_mul3, 256));
    CUDA_ERR_CHK(cudaMalloc((void **) &d_rcon, 256));

    CUDA_ERR_CHK(cudaMemcpy(d_sbox, sbox, 256, cudaMemcpyHostToDevice));
    CUDA_ERR_CHK(cudaMemcpy(d_mul2, mul2, 256, cudaMemcpyHostToDevice));
    CUDA_ERR_CHK(cudaMemcpy(d_mul3, mul3, 256, cudaMemcpyHostToDevice));
    CUDA_ERR_CHK(cudaMemcpy(d_rcon, rcon, 256, cudaMemcpyHostToDevice));

    int n = uData.size();
    byte *h_uData[n];
    byte *h_uKeys[n];

    for(int i = 0; i < n; i++) {
        CUDA_ERR_CHK(cudaMalloc((void **) &h_uData[i], uLens[i]));
        CUDA_ERR_CHK(cudaMalloc((void **) &h_uKeys[i], 16));
        CUDA_ERR_CHK(cudaMemcpy(h_uData[i], uData[i], uLens[i], cudaMemcpyHostToDevice))
        CUDA_ERR_CHK(cudaMemcpy(h_uKeys[i], uKeys[i], 16, cudaMemcpyHostToDevice));
    }

    byte **d_uData;
    byte **d_uKeys;
    int *d_uLens;
    CUDA_ERR_CHK(cudaMalloc((void **) &d_uData, n*sizeof(byte*)));
    CUDA_ERR_CHK(cudaMalloc((void **) &d_uKeys, n*sizeof(byte*)));
    CUDA_ERR_CHK(cudaMalloc((void **) &d_uLens, n*sizeof(int)));
    CUDA_ERR_CHK(cudaMemcpy(d_uData, h_uData, n*sizeof(byte*), cudaMemcpyHostToDevice));
    CUDA_ERR_CHK(cudaMemcpy(d_uKeys, h_uKeys, n*sizeof(byte*), cudaMemcpyHostToDevice));
    CUDA_ERR_CHK(cudaMemcpy(d_uLens, &(uLens[0]), n*sizeof(int), cudaMemcpyHostToDevice));

    
    int gridsize, blocksize;
    blocksize = BLOCKSIZE;
    gridsize = n; 
    GCNS_Cipher <<< gridsize, blocksize>>> (d_uData, d_uKeys, d_uLens, n, d_sbox, d_mul2, d_mul3, d_rcon);
    CUDA_ERR_CHK(cudaPeekAtLastError());
    CUDA_ERR_CHK(cudaThreadSynchronize()); 


    for(int i = 0; i < n; i++) {
        byte *cipher = new byte[uLens[i]];
        CUDA_ERR_CHK(cudaMemcpy(cipher, h_uData[i], uLens[i], cudaMemcpyDeviceToHost));
        // if(i == 0) cout << hex(cipher, uLens[i]);
        ciphers.push_back(move(cipher));
        CUDA_ERR_CHK(cudaFree(h_uData[i]));
        CUDA_ERR_CHK(cudaFree(h_uKeys[i]));
    }

    CUDA_ERR_CHK(cudaFree(d_sbox));
    CUDA_ERR_CHK(cudaFree(d_mul2));
    CUDA_ERR_CHK(cudaFree(d_mul3));
    CUDA_ERR_CHK(cudaFree(d_rcon));


    CUDA_ERR_CHK(cudaFree(d_uData));
    CUDA_ERR_CHK(cudaFree(d_uKeys));
    CUDA_ERR_CHK(cudaFree(d_uLens));
}

void get_data(opts vars, vector<byte*> &msgs, vector<int> &lens, vector<byte*> &keys, int i, int j) {

    if(i < vars.n_files_start || i > vars.n_files_end || j < 0 || j >= vars.m_batches ) {
        cout << "Invalid getdata params";
        return;
    }

	string msg_path, key_path;
    ifstream f_msg, f_key;

    int k, n;
    for(k = 0; k < i; k++) {
        msg_path = vars.path + "/" + to_string(i) + "/" + to_string(j) + "/" + to_string(k);
        key_path = msg_path+"_key";

        f_msg.open(msg_path, ios::binary);
        f_key.open(key_path, ios::binary);

	    if(f_msg && f_key) {

		    f_msg.seekg(0, f_msg.end);
	        n = f_msg.tellg();
    		f_msg.seekg(0, f_msg.beg);

            byte *message = new byte[n];
		    byte *key = new byte[16];

            f_msg.read( reinterpret_cast<char *> (message), n);
		    f_key.read( reinterpret_cast<char *> (key), 16);

            msgs.push_back(move(message));
            lens.push_back(n);
            keys.push_back(move(key));

            f_msg.close();
            f_key.close();
        }
        else {
            cout << "read failed";
        }
    }
}


int main() {
    opts vars = get_defaults();
    ofstream data_dump;
    data_dump.open(vars.datadump, fstream::app);

    int i, j;
    for(i = vars.n_files_start; i <= vars.n_files_end; i += vars.step) {
        for(j = 0; j < vars.m_batches; j++) {   
            vector<byte*> uData;
            vector<int> uLens;
            vector<byte*> uKeys;

            get_data(vars, uData, uLens, uKeys, i, j);
            vector<byte*> ciphers;
            ciphers.reserve(i);
      
            auto start = chrono::high_resolution_clock::now();
            GCNS(uData, uLens, uKeys, ciphers);
            auto end = chrono::high_resolution_clock::now();
			
            string out_path;
            ofstream fout;
            for(int k = 0; k < i; k++) {
                out_path = vars.path + "/" + to_string(i) + "/" + to_string(j) + "/" + to_string(k) + "_cipher_gcns";
                fout.open(out_path, ios::binary);
                fout.write(reinterpret_cast<char *> (ciphers[k]), uLens[k]);
                fout.close();
                delete[] uData[k];
                delete[] uKeys[k];
                delete[] ciphers[k];
            }

            auto _time = chrono::duration_cast<chrono::milliseconds>(end - start);
        	printf("\n N_FILES: %5d | BATCH: %2d | TIME: %10ld ms", i, j, _time.count());
            data_dump << vars.path << ",GCNS," << i << "," << j << "," << _time.count() << endl;
        }
        cout << endl;
    }

    data_dump.close();

    return 0;
}
