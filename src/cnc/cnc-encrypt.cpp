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
            ciphers.reserve(i);
            start = clock();
            CNC(uData, uLens, uKeys, ciphers);
            end = clock();
            batchtimes.push_back((end-start));
			sum += (end-start);
			printf("\n N_FILES: %5d | BATCH: %2d | TIME: %10.4lf ms", i, j, ((double)sum * 100)/CLOCKS_PER_SEC);
			isum += sum;

            string out_path;
            ofstream fout;
            for(int k = 0; k < i; k++) {
                out_path = vars.path + "/" + to_string(i) + "/" + to_string(j) + "/" + to_string(k) + "_cipher_cnc";
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

/*int main(){

    vector<byte*> uData;
    vector<int> uLens;
    vector<byte*> uKeys;
    vector<byte*> ciphers;

    //get_data(vars, uData, uLens, uKeys, i, j);
    CNC(uData, uLens, uKeys, ciphers);
    

    return 0;
}*/

/*
    byte message[] = {0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34, 0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34, 0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34};
    byte key[] = {0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};

    uData.push_back(move(message));
    uLens.push_back(48);
    uKeys.push_back(move(key));

    cout << hex(ciphers[0], 48) << endl;
*/
/*
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

