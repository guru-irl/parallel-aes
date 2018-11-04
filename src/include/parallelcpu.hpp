#ifndef PARALLELCPU_H
#define PARALLELCPU_H

#include <iostream>
#include "../include/aeslib.hpp"

using namespace std;

void AddRoundKey(byte *state, byte *RoundKey) {

    //#pragma unroll
    for(int i = 0; i < 16; i++) {
        state[i] ^= RoundKey[i];
    }
}

void SubBytes(byte *state, byte* d_sbox) {

    //#pragma unroll
    for(int i = 0; i < 16; i++) {
        state[i] = d_sbox[state[i]];
    }
}

void ShiftRows(byte *state) {

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

void MixColumns(byte *state, byte* d_mul2, byte* d_mul3) {

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

    //#pragma unroll
	for (int i = 0; i < 16; i++) {
		state[i] = tmp[i];
	}
}

void Round(byte *state, byte *RoundKey, bool isFinal=false){
    
	SubBytes(state, sbox);
	ShiftRows(state);
	if(!isFinal) MixColumns(state, mul2, mul3);
	AddRoundKey(state, RoundKey);
}

#endif