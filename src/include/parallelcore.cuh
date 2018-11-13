#ifndef PARALLELCORE_H
#define PARALLELCORE_H

#include <cuda.h>
#include <vector>
#include "../include/aeslib.hpp"

// This needs to be calculated properly using the formulae in the paper. 
// Using a placeholder value for now.
#define BLOCKSIZE 256 
#define SLICELEN 4096

#define CUDA_ERR_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

using namespace std;

__device__ void AddRoundKey(byte *state, byte *RoundKey) {
    #pragma unroll
    for(int i = 0; i < 16; i++) {
        state[i] ^= RoundKey[i];
    }
}

__device__ void SubBytes(byte *state, byte* d_sbox) {
    #pragma unroll
    for(int i = 0; i < 16; i++) {
        state[i] = d_sbox[state[i]];
    }
}

__device__ void ShiftRows(byte *state) {
    byte tmp;

    tmp = state[1];
    state[1] = state[5];
    state[5] = state[9];
    state[9] = state[13];
    state[13] = tmp;

    tmp = state[2];
    state[2] = state[10];
    state[10] = tmp;

    tmp = state[6];
    state[6] = state[14];
    state[14] = tmp;

    tmp = state[15];
    state[15] = state[11];
    state[11] = state[7];
    state[7] = state[3];
    state[3] = tmp;
}


__device__ void MixColumns(byte *state, byte* d_mul2, byte* d_mul3) {
    byte tmp[16];

    tmp[0]  = (byte) d_mul2[state[0]] ^ d_mul3[state[1]] ^ state[2] ^ state[3];
    tmp[1]  = (byte) state[0] ^ d_mul2[state[1]] ^ d_mul3[state[2]] ^ state[3];
    tmp[2]  = (byte) state[0] ^ state[1] ^ d_mul2[state[2]] ^ d_mul3[state[3]];
    tmp[3]  = (byte) d_mul3[state[0]] ^ state[1] ^ state[2] ^ d_mul2[state[3]];

    tmp[4]  = (byte) d_mul2[state[4]] ^ d_mul3[state[5]] ^ state[6] ^ state[7];
    tmp[5]  = (byte) state[4] ^ d_mul2[state[5]] ^ d_mul3[state[6]] ^ state[7];
    tmp[6]  = (byte) state[4] ^ state[5] ^ d_mul2[state[6]] ^ d_mul3[state[7]];
    tmp[7]  = (byte) d_mul3[state[4]] ^ state[5] ^ state[6] ^ d_mul2[state[7]];

    tmp[8]  = (byte) d_mul2[state[8]] ^ d_mul3[state[9]] ^ state[10] ^ state[11];
    tmp[9]  = (byte) state[8] ^ d_mul2[state[9]] ^ d_mul3[state[10]] ^ state[11];
    tmp[10] = (byte) state[8] ^ state[9] ^ d_mul2[state[10]] ^ d_mul3[state[11]];
    tmp[11] = (byte) d_mul3[state[8]] ^ state[9] ^ state[10] ^ d_mul2[state[11]];

    tmp[12] = (byte) d_mul2[state[12]] ^ d_mul3[state[13]] ^ state[14] ^ state[15];
    tmp[13] = (byte) state[12] ^ d_mul2[state[13]] ^ d_mul3[state[14]] ^ state[15];
    tmp[14] = (byte) state[12] ^ state[13] ^ d_mul2[state[14]] ^ d_mul3[state[15]];
    tmp[15] = (byte) d_mul3[state[12]] ^ state[13] ^ state[14] ^ d_mul2[state[15]];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        state[i] = tmp[i];
    }
}

__device__ void Round(byte *state, byte *RoundKey, byte *d_sbox, byte *d_mul2, byte *d_mul3, bool isFinal=false) {
    SubBytes(state, d_sbox);
    ShiftRows(state);
    if(!isFinal) MixColumns(state, d_mul2, d_mul3);
    AddRoundKey(state, RoundKey);
}

__device__ void d_shift_sub_rcon(byte *in, byte i, byte *d_sbox, byte *d_rcon) {
    byte t = in[0];
    in[0] = in[1];
    in[1] = in[2];
    in[2] = in[3];
    in[3] = t;

    in[0] = d_sbox[in[0]];
    in[1] = d_sbox[in[1]];
    in[2] = d_sbox[in[2]];
    in[3] = d_sbox[in[3]];

    in[0] ^= d_rcon[i];
}

__device__ void d_KeyExpansion(byte* inputKey, byte* expandedKeys, byte *d_sbox, byte *d_rcon) {
    
    // byte a = 123456;
    for (int i = 0; i < 16; i++) {
        expandedKeys[i] = inputKey[i];
    }

    int bytesGenerated = 16; 
    int rconIteration = 1;
    byte tmpCore[4]; 
    while (bytesGenerated < 176) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            tmpCore[i] = expandedKeys[i + bytesGenerated - 4];
	    }
		
	    if (bytesGenerated % 16 == 0) {
            d_shift_sub_rcon(tmpCore, rconIteration++, d_sbox, d_rcon);
        }

        #pragma unroll
        for (int a = 0; a < 4; a++) {
            expandedKeys[bytesGenerated] = expandedKeys[bytesGenerated - 16] ^ tmpCore[a];
            bytesGenerated++;
        }
    }
}


__global__ void GNC_Cipher(byte *message, int msg_length, byte expandedKey[176], byte *sbox, byte *mul2, byte *mul3) {

    __shared__ byte d_sbox[256];
    __shared__ byte d_mul2[256];
    __shared__ byte d_mul3[256];
    __shared__ byte d_expandedKey[176];

    int id = (blockDim.x*blockIdx.x + threadIdx.x) * 16;
    if(id < 256) {
        d_sbox[id] = sbox[id];
        d_mul2[id] = mul2[id];
        d_mul3[id] = mul3[id];   
    }

    if(id < 176) {
        d_expandedKey[id] = expandedKey[id];
    }

    __syncthreads();

   
    if((id + 16) <= msg_length) {
        AddRoundKey(message + id, d_expandedKey);
        for(int n = 1; n <= N_ROUNDS; n++) {
            Round(message + id, d_expandedKey + (n)*16, d_sbox, d_mul2, d_mul3, n == 10);
        }
    }
}


__global__ void GCNS_Cipher(byte **uData, byte **keys, int *lens, int n_users, byte *sbox, byte *mul2, byte *mul3, byte *rcon) {

    if(blockIdx.x < n_users) {

        __shared__ byte d_sbox[256];
        __shared__ byte d_mul2[256];
        __shared__ byte d_mul3[256];
        __shared__ byte d_expandedKey[176];
    
        int user_id = blockIdx.x;
        int id = threadIdx.x;

        if(id == 0)
            d_KeyExpansion(keys[user_id], d_expandedKey, sbox, rcon);

        __syncthreads();

        if(id < 256) {
            d_sbox[id] = sbox[id];
            d_mul2[id] = mul2[id];
            d_mul3[id] = mul3[id];   
        }
        __syncthreads();

        int cur_index = (threadIdx.x)*16;
        if((cur_index + 16) <= lens[user_id]) {
            AddRoundKey(uData[user_id] + cur_index, d_expandedKey);
            for(int n = 1; n <= N_ROUNDS; n++) {
                Round(uData[user_id] + cur_index, d_expandedKey + (n)*16, d_sbox, d_mul2, d_mul3, n == 10);
            }
        }
    }
}

__global__ void GCS_Cipher(byte **slicedData, byte **expandedKeys, int *keyTable, int n_slices, byte *sbox, byte *mul2, byte *mul3) {

    if(blockIdx.x < n_slices) {

        __shared__ byte d_sbox[256];
        __shared__ byte d_mul2[256];
        __shared__ byte d_mul3[256];
        __shared__ byte d_expandedKey[176];
        __shared__ int user_id;

        int slice_id = blockIdx.x;
        int id = threadIdx.x;
    
        
        if(id == 0) {
            user_id = keyTable[slice_id];
        }
        __syncthreads();

        if(id < 256) {
            d_sbox[id] = sbox[id];
            d_mul2[id] = mul2[id];
            d_mul3[id] = mul3[id];   
        }

        if (id < 176) {
            d_expandedKey[id] = expandedKeys[user_id][id];
        }
        __syncthreads();
        
        
        int cur_index = (threadIdx.x)*16;
        if((cur_index + 16) <= SLICELEN) {

            byte local_4bytes[16];
            #pragma unroll
            for(int i = 0; i < 16; i++)
                local_4bytes[i] = (slicedData[slice_id] + cur_index)[i];

            AddRoundKey(local_4bytes, d_expandedKey);
            for(int n = 1; n <= N_ROUNDS; n++) {
                Round(local_4bytes, d_expandedKey + (n)*16, d_sbox, d_mul2, d_mul3, n == 10);
            }

            #pragma unroll
            for(int i = 0; i < 16; i++)
                (slicedData[slice_id] + cur_index)[i] = local_4bytes[i];
        }
    }
}

#endif
