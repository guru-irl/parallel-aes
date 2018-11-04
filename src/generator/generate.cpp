#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <string>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>

#include "../include/genlib.hpp"

#define SYSERROR()  errno
#define KEYLENGTH 16

using namespace std;

typedef unsigned char byte;

/*
    1 Step -> M Batches -> N files per batch
*/

void make_dir(string path) {
    const int dir_err = mkdir(path.c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (dir_err == -1)
    {
        cerr << "Error creating directory:" << path;
        exit(1);
    }
}

void check_make_dir(string path) {
    DIR* dir = opendir(path.c_str());

    if (dir) {
        closedir(dir);
    }
    else if(SYSERROR() = ENOENT) {
        make_dir(path);
    }
    else {
        cerr << "opendir failed";
        exit(1);
    }
}

void generate_file(string path, int n_bytes) {
    ofstream fout;
    fout.open(path, ios::binary);

    if(fout.is_open()){
        int i;
        byte b;
        for(i = 0; i < n_bytes; i++) {
            b = rand()%256;
            fout << b;
        }
        fout.close();
    }
    else {
        cerr<<"Failed to open file : "<< SYSERROR() << std::endl;
        exit(1);
    }

    fout.open(path + "_key", ios::binary);
    if(fout.is_open()){
        int i;
        byte b;
        for(i = 0; i < KEYLENGTH; i++) {
            b = rand()%256;
            fout << b;
        }
        fout.close();
    }
    else {
        cerr<<"Failed to open file : "<< SYSERROR() << std::endl;
        exit(1);
    }
}

void generate(opts vars) {
    int i, j, k, n_bytes;
    string path;
    for(i = vars.n_files_start; i <= vars.n_files_end; i+=vars.step) {
        for (j = 0; j < vars.m_batches; j++) {
            for(k = 0; k < i; k++) {
                path = vars.path + "/" + to_string(i);
                check_make_dir(path);
                path = path + "/" + to_string(j);  
                check_make_dir(path);
                path = path + "/" + to_string(k);
                cout << path << endl;
                n_bytes = vars.minlength + rand()%(vars.maxlength - vars.minlength + 1);
                n_bytes -= n_bytes%16;
                generate_file(path, n_bytes);
            }
        }
    }
}

int main(int argc, char const *argv[])
{
    opts vars = get_defaults();
    
    /*
    vars.path = argv[1];
    vars.n_files_start = stoi(argv[2]);
    vars.n_files_end = stoi(argv[3]);
    vars.step = stoi(argv[4]);
    vars.m_batches = stoi(argv[5]);
    */  
    
    /*vars.path = "../dataset";
    vars.n_files_start = 100;
    vars.n_files_end = 1000;
    vars.step = 100;
    vars.m_batches = 1;
    vars.minlength = 1024; //bytes
    vars.maxlength = 1024*32; //bytes*/

    srand(time(NULL));

    check_make_dir(vars.path);
    generate(vars);

    return 0;
}