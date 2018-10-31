#include <iostream>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <cuda.h>
#include <vector>

#include "../include/aeslib.h"
#include "../include/genlib.hpp"
#include "../include/parallelcore.cuh"

using namespace std;

void get_data(opts vars, vector<byte*> &msgs, vector<int> &lens, vector<byte*> &keys, int i, int j) {

    if(i < vars.n_files_start || i > vars.n_files_end || j < 0 || j >= vars.m_batches ) {
        cout << "Invalid getdata params";
        return;
    }

	string msg_path, key_path;
    ifstream f_msg, f_key;

    for(k = 0; k < i; k++) {
        msg_path = vars.path + "/" + to_string(i) + "/" + to_string(j) + "/" + to_string(k);
        key_path = path+"_key";

        cout << msg_path << " ";
        f_msg.open(msg_path);
        f_key.open(key_path);

	    if(f_msg && f_key) {

		    f_msg.seekg(0, f_msg.end);
	        n = f_msg.tellg();
            cout << n << endl;
    		f_msg.seekg(0, f_msg.beg);

            byte message[n];
		    byte key[16];

            f_msg.read(reinterpret_cast<char *> (message), n);
		    f_key.read(reinterpret_cast<char *> (key), 16);

            msgs.push_back(message);
            lens.push_back(n);
            keys.push_back(key);

            f_msg.close();
            f_key.close();
        }
        else {
            cout << "read failed";
        }
    }
}

void GNC(vector<byte *> &uData, vector<int> &uLens vector<byte *> &uKeys, vector<byte *> &ciphers) {
    
    // The published algorithm copies the ciphers back to uData
    // But I'm gonna put them in a separate array in case I need the raw user data for something.

    
    // The following variables are stored in global memory
    // They will be further copied to shared memory in the kernel
    // The idea being to reduce memory latency 
    byte* d_sbox;
    byte* d_mul2;
    byte* d_mul3;
    load_boxes(d_sbox, d_mul2, d_mul3);

    int n;
    byte expandedKey[176];
    byte* d_expandedKey;
    cudaMalloc((void**) &d_expandedKey, 176);

    byte *message;
    byte *cipher;

    for(int i = 0; i < uData.length(); i++) {
        n = uLens[i];
        byte message[n];
        byte cipher[n];
        


        KeyExpansion(key, expandedKey);
        cudaMemcpy(d_expandedKey, expandedKey, 176, cudaMemcpyHostToDevice);

	    Cipher(message, n, expandedKey, cipher);
    }
	// cout << "MSG\n" << hex(message, n) << endl << endl; 
	// cout << "Cry\n" << hex(cipher, n) << endl << endl;	
}

