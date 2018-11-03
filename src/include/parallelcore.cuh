#ifndef PARALLELCORE_H
#define PARALLELCORE_H

#include <cuda.h>
#include <iostream>
#include "../include/aeslib.hpp"

// This needs to be calculated properly using the formulae in the paper. 
// Using a placeholder value for now.
#define BLOCKSIZE 1024 

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
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

__global__ void GNC_Cipher(byte *message, int msg_length, byte expandedKey[176], byte *sbox, byte *mul2, byte *mul3) {

    __shared__ byte d_sbox[256];
    __shared__ byte d_mul2[256];
    __shared__ byte d_mul3[256];
    __shared__ byte d_expandedKey[176];

    if(threadIdx.x == 0) {
        for(int i = 0; i < 256; i++) {
            d_sbox[i] = sbox[i];
            d_mul2[i] = mul2[i];
            d_mul3[i] = mul3[i];
            if(i < 176) d_expandedKey[i] = expandedKey[i];
		}
    }

	__syncthreads();
	int id = (blockDim.x*blockIdx.x + threadIdx.x) * 16;
	// printf("%d %d %d \n", blockDim.x, blockIdx.x, threadIdx.x);
	// printf("Thread out %d \n", id/16);
	
    if((id + 16) <= msg_length) {
		// printf("Thread %d \n", id/16);
        AddRoundKey(message + id, d_expandedKey);
		for(int n = 1; n <= N_ROUNDS; n++) {
			Round(message + id, d_expandedKey + (n)*16, d_sbox, d_mul2, d_mul3, n == 10);
		}
    }
}

#endif