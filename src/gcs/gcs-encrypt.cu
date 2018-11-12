#include <iostream>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <cuda.h>
#include <vector>
#include <ctime>

#include "../include/aeslib.hpp"
#include "../include/genlib.hpp"
#include "../include/parallelcore.cuh"

using namespace std;

int slice(vector<byte *> &uData, vector<int> &uLens, vector<byte *> &slicedData, vector<int> &Table, int totLens, int slicelen) {
    int n_slices = ceil((float)totLens/ (float)slicelen);
    slicedData.reserve(n_slices);
    Table.reserve(n_slices);
    
    int i, j, k, l;
    int n = uData.size();
    for(i = 0; i < n; i++) {
        l = uLens[i];
        for(j = 0; j < l; j+=slicelen) {
            byte* slice = new byte[slicelen]; 
            for(k = 0; k < slicelen; k++) {
                if(j + k < l) {
                    slice[k] = uData[i][j+k];
                }
                else {
                    slice[k] = 0;
                }
            }
            Table.push_back(i);
            slicedData.push_back(move(slice));
        }
    }

    return slicedData.size();
}

void GCS(vector<byte *> &uData, vector<int> &uLens, vector<byte *> &uKeys, vector<byte *> &ciphers, int totLens, int sliceLen) {
    
    // The published algorithm copies the ciphers back to uData
    // But I'm gonna put them in a separate array in case I need the raw user data for something.

    // The following variables are stored in global memory
    // They will be further copied to shared memory in the kernel
    // The idea being to reduce memory latency 

    vector<byte *> slicedData;
    vector<int> key_table;

    int n_slices = slice(uData, uLens, slicedData, key_table, totLens, sliceLen);

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
    byte *h_slicedData[n_slices];
    byte *h_uKeys[n];

    for(int i = 0; i < n_slices; i++) {
        CUDA_ERR_CHK(cudaMalloc((void **) &h_slicedData[i], sliceLen));
        CUDA_ERR_CHK(cudaMemcpy(h_slicedData[i], slicedData[i], sliceLen, cudaMemcpyHostToDevice))
        if(i < n) {
            CUDA_ERR_CHK(cudaMalloc((void **) &h_uKeys[i], 16));
            CUDA_ERR_CHK(cudaMemcpy(h_uKeys[i], uKeys[i], 16, cudaMemcpyHostToDevice));
        }
    }

    byte **d_slicedData;
    byte **d_uKeys;
    int *d_key_table;
    CUDA_ERR_CHK(cudaMalloc((void **) &d_slicedData, n_slices*sizeof(byte*)));
    CUDA_ERR_CHK(cudaMalloc((void **) &d_uKeys, n*sizeof(byte*)));
    CUDA_ERR_CHK(cudaMalloc((void **) &d_key_table, n_slices*sizeof(int)));

    CUDA_ERR_CHK(cudaMemcpy(d_slicedData, h_slicedData, n_slices, cudaMemcpyHostToDevice));
    CUDA_ERR_CHK(cudaMemcpy(d_uKeys, h_uKeys, n, cudaMemcpyHostToDevice));
    CUDA_ERR_CHK(cudaMemcpy(d_key_table, &(key_table[0]), n_slices, cudaMemcpyHostToDevice));

    
    int gridsize, blocksize;
    blocksize = BLOCKSIZE;
    gridsize = n_slices; 
    GCS_Cipher <<< gridsize, blocksize>>> (d_slicedData, d_uKeys, d_key_table, sliceLen, n_slices, d_sbox, d_mul2, d_mul3, d_rcon);
    CUDA_ERR_CHK(cudaPeekAtLastError());
    
    /*
    int cur_slice = 0, j, number_of_slices_in_i;
    for(int i = 0; i < n; i++) {
        number_of_slices_in_i = ceil((float)uLens[i] / (float)sliceLen);
        byte *cipher = new byte[number_of_slices_in_i * sliceLen];
        for(j = 0; j < number_of_slices_in_i; j++) {
            CUDA_ERR_CHK(cudaMemcpy(cipher + j*sliceLen, h_slicedData[cur_slice], sliceLen, cudaMemcpyDeviceToHost));
            CUDA_ERR_CHK(cudaFree(h_slicedData[cur_slice]));  
            cur_slice++;
        }
        ciphers.push_back(move(cipher));
        CUDA_ERR_CHK(cudaFree(h_uKeys[i]));
    }*/

    CUDA_ERR_CHK(cudaFree(d_slicedData));
    CUDA_ERR_CHK(cudaFree(d_uKeys));
}

long long get_data(opts vars, vector<byte*> &msgs, vector<int> &lens, vector<byte*> &keys, int i, int j) {

    if(i < vars.n_files_start || i > vars.n_files_end || j < 0 || j >= vars.m_batches ) {
        cout << "Invalid getdata params";
        return i;
    }

	string msg_path, key_path;
    ifstream f_msg, f_key;
    long long sum = 0;

    int k, n;
    for(k = 0; k < i; k++) {
        msg_path = vars.path + "/" + to_string(i) + "/" + to_string(j) + "/" + to_string(k);
        key_path = msg_path+"_key";

        f_msg.open(msg_path, ios::binary);
        f_key.open(key_path, ios::binary);

	    if(f_msg && f_key) {

		    f_msg.seekg(0, f_msg.end);
            n = f_msg.tellg();
            sum += n;
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

    return sum;
}


int main() {
    opts vars = get_defaults();
	clock_t start, end;
    int i, j;
    for(i = vars.n_files_start; i <= vars.n_files_end; i += vars.step) {
        
        long long isum = 0;
        for(j = 0; j < vars.m_batches; j++) {
            vector<long> batchtimes;
			long sum = 0;
            
            vector<byte*> uData;
            vector<int> uLens;
            vector<byte*> uKeys;
            long long totLens;

            // Need to calculate this properly
            int sliceLen = SLICELEN;

            totLens = get_data(vars, uData, uLens, uKeys, i, j);
            vector<byte*> ciphers;
            ciphers.reserve(i);
            
            start = clock();
            GCS(uData, uLens, uKeys, ciphers, totLens, sliceLen);
            end = clock();
            batchtimes.push_back((end-start));
			sum += (end-start);
			printf("\n N_FILES: %5d | BATCH: %2d | TIME: %10.4lf ms", i, j, ((double)sum * 100)/CLOCKS_PER_SEC);
			isum += sum;

            string out_path;
            ofstream fout;
            for(int k = 0; k < i; k++) {
                out_path = vars.path + "/" + to_string(i) + "/" + to_string(j) + "/" + to_string(k) + "_cipher_gcs";
                fout.open(out_path, ios::binary);
                fout.write(reinterpret_cast<char *> (ciphers[k]), uLens[k]);
                fout.close();
                delete[] uData[k];
                delete[] uKeys[k];
            }
        }
		printf("\n N_FILES: %5d | AVG_TIME: %10.4lf ms\n", i, (((double)isum * 100)/vars.m_batches)/CLOCKS_PER_SEC);
    }

    return 0;
}