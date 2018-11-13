#include <iostream>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <cuda.h>
#include <vector>
#include <numeric>
#include <chrono> 

#include "../include/aeslib.hpp"
#include "../include/genlib.hpp"
#include "../include/parallelcore.cuh"

using namespace std;

int sliceData(vector<byte *> &uData, vector<int> &uLens, vector<byte *> &slicedData, vector<int> &keyTable){

    //cut coalesced data into slices
    int n;
    int approx_n_slices = 0;
    int n_users = uData.size();

    /*for(int i = 0 ; i < uLens.size() ; ++i)
        n_slices += ceil((float)uLens[i]/(float)SLICELEN);*/
    
    long long tot_lens = accumulate(uLens.begin(), uLens.end(), 0);
    approx_n_slices = tot_lens/SLICELEN;

    slicedData.reserve(approx_n_slices);
    keyTable.reserve(approx_n_slices);

    for(int i = 0; i < n_users; i++){
        n = uLens[i];
        for(int j = 0; j < n; j += SLICELEN){
            byte* slice = new byte[SLICELEN]; 
            for(int k = 0; k < SLICELEN; k++) {
                if(j + k < n) {
                    slice[k] = uData[i][j+k];
                }
                else {
                    slice[k] = 0;
                }
            }
            keyTable.push_back(i);
            slicedData.push_back(move(slice));
        }
    }

    return slicedData.size();
}

long GCS(vector<byte *> &uData, vector<int> &uLens, vector<int> keyTable, vector<byte *> &uKeys, vector<byte *> &ciphers) {
    
    // The published algorithm copies the ciphers back to uData
    // But I'm gonna put them in a separate array in case I need the raw user data for something.

    // The following variables are stored in global memory
    // They will be further copied to shared memory in the kernel
    // The idea being to reduce memory latency 

    auto start = chrono::high_resolution_clock::now();
    vector<byte *> expandedKeys(uKeys.size());

    for(int i = 0; i < uKeys.size(); i++) {
        expandedKeys[i] = new byte[176];
        KeyExpansion(uKeys[i], expandedKeys[i]);
    }

    byte *d_sbox;
    byte *d_mul2;
    byte *d_mul3;
    // byte *d_rcon;

    CUDA_ERR_CHK(cudaMalloc((void **) &d_sbox, 256));
    CUDA_ERR_CHK(cudaMalloc((void **) &d_mul2, 256));
    CUDA_ERR_CHK(cudaMalloc((void **) &d_mul3, 256));
    // CUDA_ERR_CHK(cudaMalloc((void **) &d_rcon, 256));

    CUDA_ERR_CHK(cudaMemcpy(d_sbox, sbox, 256, cudaMemcpyHostToDevice));
    CUDA_ERR_CHK(cudaMemcpy(d_mul2, mul2, 256, cudaMemcpyHostToDevice));
    CUDA_ERR_CHK(cudaMemcpy(d_mul3, mul3, 256, cudaMemcpyHostToDevice));
    // CUDA_ERR_CHK(cudaMemcpy(d_rcon, rcon, 256, cudaMemcpyHostToDevice));

    int n_slices = uData.size();
    int n = uLens.size();
    byte *h_uData[n_slices];
    byte *h_uKeys[n];

    for(int i = 0; i < n_slices; i++) {
        CUDA_ERR_CHK(cudaMalloc((void **) &h_uData[i], SLICELEN));
        CUDA_ERR_CHK(cudaMemcpy(h_uData[i], uData[i], SLICELEN, cudaMemcpyHostToDevice))

        if(i < n) {
        CUDA_ERR_CHK(cudaMalloc((void **) &h_uKeys[i], 176));
        CUDA_ERR_CHK(cudaMemcpy(h_uKeys[i], expandedKeys[i], 176, cudaMemcpyHostToDevice));
        }
    }

    byte **d_uData;
    byte **d_uKeys;
    int *d_keyTable;
    CUDA_ERR_CHK(cudaMalloc((void **) &d_uData, n_slices*sizeof(byte*)));
    CUDA_ERR_CHK(cudaMalloc((void **) &d_uKeys, n*sizeof(byte*)));
    CUDA_ERR_CHK(cudaMalloc((void **) &d_keyTable, n_slices*sizeof(int)));
    CUDA_ERR_CHK(cudaMemcpy(d_uData, h_uData, n_slices*sizeof(byte*), cudaMemcpyHostToDevice));
    CUDA_ERR_CHK(cudaMemcpy(d_uKeys, h_uKeys, n*sizeof(byte*), cudaMemcpyHostToDevice));
    CUDA_ERR_CHK(cudaMemcpy(d_keyTable, &(keyTable[0]), n_slices*sizeof(int), cudaMemcpyHostToDevice));
    
    int gridsize, blocksize;
    blocksize = BLOCKSIZE;
    gridsize = n_slices; 

    GCS_Cipher <<< gridsize, blocksize>>> (d_uData, d_uKeys, d_keyTable, n_slices, d_sbox, d_mul2, d_mul3);
    CUDA_ERR_CHK(cudaPeekAtLastError()); // Checks for launch error
    CUDA_ERR_CHK(cudaThreadSynchronize()); // Checks for execution error
    auto end = chrono::high_resolution_clock::now();

    
    for(int i = 0; i < n_slices; i++) {
        byte *cipher = new byte[SLICELEN];
        CUDA_ERR_CHK(cudaMemcpy(cipher, h_uData[i], SLICELEN, cudaMemcpyDeviceToHost));
        ciphers.push_back(move(cipher));
        CUDA_ERR_CHK(cudaFree(h_uData[i]));
        if(i < n)
        CUDA_ERR_CHK(cudaFree(h_uKeys[i]));
    }

    CUDA_ERR_CHK(cudaFree(d_uData));
    CUDA_ERR_CHK(cudaFree(d_uKeys));
    CUDA_ERR_CHK(cudaFree(d_keyTable));

    CUDA_ERR_CHK(cudaFree(d_sbox));
    CUDA_ERR_CHK(cudaFree(d_mul2));
    CUDA_ERR_CHK(cudaFree(d_mul3));


    auto _time = chrono::duration_cast<chrono::milliseconds>(end - start);
    return _time.count();
}


long long get_data(opts vars, vector<byte*> &msgs, vector<int> &lens, vector<byte*> &keys, int i, int j) {

    if(i < vars.n_files_start || i > vars.n_files_end || j < 0 || j >= vars.m_batches ) {
        cout << "Invalid getdata params";
        return -1;
    }

	string msg_path, key_path;
    ifstream f_msg, f_key;
   
    int k, n;
    long long sum = 0;
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
    ofstream data_dump;
    data_dump.open(vars.datadump, fstream::app);

    int i, j;
    long time_val;
    for(i = vars.n_files_start; i <= vars.n_files_end; i += vars.step) {
        for(j = 0; j < vars.m_batches; j++) {
            vector<byte*> uData;
            vector<int> uLens;
            vector<int> keyTable;
            vector<byte*> uKeys;
            vector<byte*> slicedData;

            long long len = get_data(vars, uData, uLens, uKeys, i, j);
            int n_slices = sliceData(uData, uLens, slicedData, keyTable);
            vector<byte*> ciphers;
            ciphers.reserve(n_slices);
    
            auto start = chrono::high_resolution_clock::now();
            time_val = GCS(slicedData, uLens, keyTable, uKeys, ciphers);
            auto end = chrono::high_resolution_clock::now();
    
            string out_path;
            ofstream fout;

            int cur_slice = 0;
            for(int k = 0; k < i; k++) {
                out_path = vars.path + "/" + to_string(i) + "/" + to_string(j) + "/" + to_string(k) + "_cipher_gcs";
                fout.open(out_path, ios::binary);

                while(cur_slice < n_slices && keyTable[cur_slice] == k) {
                    fout.write(reinterpret_cast<char *> (ciphers[cur_slice]), SLICELEN);
                    delete[] slicedData[cur_slice];
                    delete[] ciphers[cur_slice];
                    cur_slice++;
                }

                fout.close();
                delete[] uData[k];
                delete[] uKeys[k];
            }

            auto _time = chrono::duration_cast<chrono::milliseconds>(end - start);
        	printf("\n N_FILES: %5d | BATCH: %2d | TIME: %10ld ms", i, j, time_val);
            data_dump << vars.path << ",GCS," << i << "," << j << "," << time_val << "," << len << endl;
        }
        cout << endl;
	}

    return 0;
}
