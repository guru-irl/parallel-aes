#include <iostream>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <ctime>
#include <omp.h>
#include <sys/time.h> 
#include <math.h>

#include "../include/aeslib.hpp"
#include "../include/genlib.hpp"
#include "../include/parallelcpu.hpp"

#define SLICE_LEN 16384

using namespace std;

void CCS(vector<byte *> &uData, vector<int> &uLens, vector<byte *> &uKeys, vector<byte *> &ciphers){

    int n;                      //variable to store the length of the message in the loop
    byte expandedKey[176];      //expanded key variable, differs for every user key -> 44*4 = 176

    //cut coalesced data into slices

    vector<byte *> slicedData;
    vector<int> key_table;

    int n_slices = 0;
    int n_users = uData.size();

    for(int i = 0 ; i < uLens.size() ; ++i)
        n_slices += ceil((float)uLens[i]/(float)SLICE_LEN);

    slicedData.reserve(n_slices);
    key_table.reserve(n_slices);

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
    //nested parallesim is being implemented
    //enables nested parallelism
    omp_set_nested(1);

    //the total number of cores I have is 4
    //hence the parallelism is split as 2*2 giving a total of threads 
    omp_set_num_threads(2);
    #pragma omp parallel for
    for(int i = 0; i < uData.size(); i++) {

        n = uLens[i];
        byte *cipher = new byte[n];
        
        KeyExpansion(uKeys[i], expandedKey);
        
        omp_set_num_threads(2);
        #pragma omp parallel for 
        for(int curr_index = 0 ; curr_index<uLens[i] ; curr_index+=16){

            AddRoundKey(uData[i] + curr_index , expandedKey);
            for(int n_rounds = 1 ; n_rounds<=10 ; ++n_rounds)
                Round(uData[i] + curr_index, expandedKey + (n_rounds*16), (n_rounds==10));
        }

        cipher = uData[i];
        ciphers.push_back(move(cipher));
    }
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
    int i, j;
    for(i = vars.n_files_start; i <= vars.n_files_end; i += vars.step) {
        
        double isum = 0;
        for(j = 0; j < vars.m_batches; j++) {

            vector<double> batchtimes;
			double sum = 0;
    
            vector<byte*> uData;
            vector<int> uLens;
            vector<byte*> uKeys;

            get_data(vars, uData, uLens, uKeys, i, j);
            vector<byte*> ciphers;
            ciphers.reserve(i);

            struct timeval start, end; 
            gettimeofday(&start, NULL); 
            ios_base::sync_with_stdio(false); 
            CCS(uData, uLens, uKeys, ciphers);
            gettimeofday(&end, NULL); 

            double time_taken; 
  
            time_taken = (end.tv_sec - start.tv_sec) * 1e6; 
            time_taken = (time_taken + (end.tv_usec - start.tv_usec));

            batchtimes.push_back((time_taken));
			sum += (time_taken);
			printf("\n N_FILES: %5d | BATCH: %2d | TIME: %10.4lf ms", i, j, ((double)sum * 100)/CLOCKS_PER_SEC);
			isum += sum;

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
        }
		printf("\n N_FILES: %5d | AVG_TIME: %10.4lf ms\n", i, (((double)isum * 100)/vars.m_batches)/CLOCKS_PER_SEC);
    }

    return 0;
}



