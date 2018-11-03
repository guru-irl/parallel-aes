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

void GNC(vector<byte *> &uData, vector<int> &uLens, vector<byte *> &uKeys, vector<byte *> &ciphers) {
    
    // The published algorithm copies the ciphers back to uData
    // But I'm gonna put them in a separate array in case I need the raw user data for something.

    // The following variables are stored in global memory
    // They will be further copied to shared memory in the kernel
    // The idea being to reduce memory latency 
    byte *d_sbox;
    byte *d_mul2;
    byte *d_mul3;

    gpuErrchk(cudaMalloc((void **) &d_sbox, 256));
    gpuErrchk(cudaMalloc((void **) &d_mul2, 256));
    gpuErrchk(cudaMalloc((void **) &d_mul3, 256));

    gpuErrchk(cudaMemcpy(d_sbox, sbox, 256, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_mul2, mul2, 256, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_mul3, mul3, 256, cudaMemcpyHostToDevice));


    int n;
    byte expandedKey[176];
    byte *d_expandedKey;
    gpuErrchk(cudaMalloc((void**) &d_expandedKey, 176));
    
    int gridsize, blocksize;
    for(int i = 0; i < uData.size(); i++) {
        n = uLens[i];
        byte *d_message;
        byte *cipher = new byte[n];
        gpuErrchk(cudaMalloc((void**) &d_message, n));
        
        KeyExpansion(uKeys[i], expandedKey);
        gpuErrchk(cudaMemcpy(d_expandedKey, expandedKey, 176, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_message, uData[i], n, cudaMemcpyHostToDevice));
        
        blocksize = BLOCKSIZE;
        gridsize = ceil (uLens[i]/(BLOCKSIZE*16));
        
        if(uLens[i] <= BLOCKSIZE) gridsize = 1;

        Cipher <<< gridsize, blocksize>>> (d_message, n, d_expandedKey, d_sbox, d_mul2, d_mul3);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaMemcpy(cipher, d_message, n, cudaMemcpyDeviceToHost));
        ciphers.push_back(move(cipher));

        gpuErrchk(cudaFree(d_message));
    }
    
    // cout << endl << endl;
    // cout << hex(uData[0], uLens[0]);
    // cout << endl << endl << endl << endl;
    // cout << hex(ciphers[0], uLens[0]);
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

            // if(k == 0) cout << endl << endl << hex(message, n) << endl << endl;

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

    // cout << msgs.size() << endl;
    // cout << hex(keys[i-1], 16) << endl;
    // cout << hex(msgs[0], lens[0]) << endl;
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

            get_data(vars, uData, uLens, uKeys, i, j);
            vector<byte*> ciphers;
            
            start = clock();
            GNC(uData, uLens, uKeys, ciphers);
            end = clock();
            batchtimes.push_back((end-start));
			sum += (end-start);
			printf("\n N_FILES: %5d | BATCH: %2d | TIME: %10.4lf ms", i, j, ((double)sum * 100)/CLOCKS_PER_SEC);
			isum += sum;

            string out_path;
            ofstream fout;
            for(int k = 0; k < i; k++) {
                out_path = vars.path + "/" + to_string(i) + "/" + to_string(j) + "/" + to_string(k) + "_cipher_gnc";
                fout.open(out_path, ios::binary);
                fout.write(reinterpret_cast<char *> (ciphers[k]), uLens[k]);
                fout.close();
                // free(uData[i]);
                // free(uKeys[i]);
            }
        }
		printf("\n N_FILES: %5d | AVG_TIME: %10.4lf ms\n", i, (((double)isum * 100)/vars.m_batches)/CLOCKS_PER_SEC);
    }

    return 0;
}

/*
    // VERIFICATION ANALYSIS
    byte *d_sbox;
    byte *d_mul2;
    byte *d_mul3;
    cudaMalloc((void **) &d_sbox, 256);
    cudaMalloc((void **) &d_mul2, 256);
    cudaMalloc((void **) &d_mul3, 256);
    
    cudaMemcpy(d_sbox, sbox, 256, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mul2, mul2, 256, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mul3, mul3, 256, cudaMemcpyHostToDevice);
    
    byte message[] = {0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34, 0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34, 0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34};
    byte key[] = {0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
    byte expandedKey[176];
    byte cipher[48];
    byte* d_message;
    byte* d_expandedKey;
    // byte* d_cipher;
    int n = 48;
            
    // byte cipher[n];
    cudaMalloc((void**) &d_message, n);
    // cudaMalloc((void**) &d_cipher, n);
    cudaMalloc((void**) &d_expandedKey, 176);
    
    KeyExpansion(key, expandedKey);
    cudaMemcpy(d_expandedKey, expandedKey, 176, cudaMemcpyHostToDevice);
    cudaMemcpy(d_message, message, n, cudaMemcpyHostToDevice);
    Cipher <<<1, 256>>> (d_message, n, d_expandedKey, d_sbox, d_mul2, d_mul3);

    cudaMemcpy(cipher, d_message, n, cudaMemcpyDeviceToHost);
    cout << hex(cipher, 48) << endl;  
*/