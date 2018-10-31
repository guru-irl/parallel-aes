#ifndef GENLIB_H
#define GENLIB_H

#include <string>

struct opts {
    int n_files_start;
    int n_files_end;
    int step;
    int m_batches;
    std::string path;

    int minlength;
    int maxlength;
};

opts get_defaults(){
    opts default_vars;
    default_vars.path = "../dataset";
    default_vars.n_files_start = 100;
    default_vars.n_files_end = 1000;
    default_vars.step = 100;
    default_vars.m_batches = 1;
    default_vars.minlength = 1024*32; //bytes
    default_vars.maxlength = 1024*128; //bytes

    return default_vars;
}
#endif