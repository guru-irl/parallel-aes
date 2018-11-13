#include <iostream>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <omp.h> 
#include <math.h>

#include <chrono>

#include "../include/aeslib.hpp"
#include "../include/genlib.hpp"
#include "../include/parallelcpu.hpp"

#define SLICE_LEN 16384

using namespace std;

void sliceData(vector<byte *> &uData, vector<int> &uLens, vector<byte *> &slicedData, vector<int> &key_table){

    //cut coalesced data into slices
    int n;
    int n_users = uData.size();

    for(int i = 0; i < n_users; i++){
        n = uLens[i];
        for(int j = 0; j < n; j += SLICE_LEN){
            byte* slice = new byte[SLICE_LEN]; 
            for(int k = 0; k < SLICE_LEN; k++) {
                if(j + k < n) {
                    slice[k] = uData[i][j+k];
                }
                else {
                    slice[k] = 0;
                }
            }
            key_table.push_back(i);
            slicedData.push_back(move(slice));
        }
    }
}

void CCS(vector<byte *> &slicedData, vector<int> &uLens, vector<byte *> &uKeys, vector<byte *> &ciphers, vector<int> key_table){

    int n;                      //variable to store the length of the message in the loop
    byte expandedKey[176];      //expanded key variable, differs for every user key -> 44*4 = 176

    //nested parallesim is being implemented
    //enables nested parallelism
    //omp_set_nested(1);

    //the total number of cores I have is 4
    //hence the parallelism is split as 2*2 giving a total of 4 threads 
    omp_set_num_threads(4);
    #pragma omp parallel for
    for(int i = 0; i < slicedData.size(); i++) {

        byte *cipher = new byte[SLICE_LEN];
        
        KeyExpansion(uKeys[key_table[i]], expandedKey);
        
        //omp_set_num_threads(2);
        //#pragma omp parallel for 
        for(int curr_index = 0 ; curr_index<SLICE_LEN ; curr_index+=16){

            AddRoundKey(slicedData[i] + curr_index , expandedKey);
            for(int n_rounds = 1 ; n_rounds<=10 ; ++n_rounds)
                Round(slicedData[i] + curr_index, expandedKey + (n_rounds*16), (n_rounds==10));
        }

        cipher = slicedData[i];
        ciphers.push_back(move(cipher));
    }
}

long long get_data(opts vars, vector<byte*> &msgs, vector<int> &lens, vector<byte*> &keys, int i, int j) {

    if(i < vars.n_files_start || i > vars.n_files_end || j < 0 || j >= vars.m_batches ) {
        cout << "Invalid getdata params";
        return -1;
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
}

int main() {
    opts vars = get_defaults();    
    ofstream data_dump;
    data_dump.open(vars.datadump, fstream::app);
    int i, j;
    for(i = vars.n_files_start; i <= vars.n_files_end; i += vars.step) {
        for(j = 0; j < vars.m_batches; j++) {

            vector<double> batchtimes;
			double sum = 0;
    
            vector<byte*> uData;
            vector<int> uLens;
            vector<byte*> uKeys;

            long long len = get_data(vars, uData, uLens, uKeys, i, j);

            // we have to slice the data and obtain the key table
            vector<byte *> slicedData;
            vector<int> key_table;
            int n_slices = 0;

            for(int i = 0 ; i < uLens.size() ; ++i)
                n_slices += ceil((float)uLens[i]/(float)SLICE_LEN);

            slicedData.reserve(n_slices);
            key_table.reserve(n_slices);

            sliceData(uData,uLens, slicedData, key_table);

            vector<byte*> ciphers;
            ciphers.reserve(n_slices);

            auto start = chrono::high_resolution_clock::now();
            CCS(slicedData, uLens, uKeys, ciphers, key_table);
            auto end = chrono::high_resolution_clock::now();
  
            string out_path;
            ofstream fout;
            for(int k = 0; k < i; k++) {
                out_path = vars.path + "/" + to_string(i) + "/" + to_string(j) + "/" + to_string(k) + "_cipher_ccs";
                fout.open(out_path, ios::binary);
                fout.write(reinterpret_cast<char *> (ciphers[k]), uLens[k]);
                fout.close();
                delete[] uData[k];
                delete[] uKeys[k];
            }
            auto _time = chrono::duration_cast<chrono::milliseconds>(end - start);
        	printf("\n N_FILES: %5d | BATCH: %2d | TIME: %10ld ms", i, j, _time.count());
            data_dump << vars.path << ",CCS," << i << "," << j << "," << _time.count() << "," << len << endl;
        }
        cout << endl;
    }
    return 0;
}



