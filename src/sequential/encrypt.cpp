#include <iostream>
#include "aeslib.h"
using namespace std;


void AddRoundKey(byte *state, byte *RoundKey) {
    for(int i = 0; i < 16; i++) {
        state[i] ^= RoundKey[i];
    }
}

void SubBytes(byte *state) {
    for(int i = 0; i < 16; i++) {
        state[i] = sbox[state[i]];
    }
}

void ShiftRows(byte *state) {
  	byte tmp[16];

    tmp[0] = state[0];
	tmp[1] = state[5];
	tmp[2] = state[10];
	tmp[3] = state[15];
	
	tmp[4] = state[4];
	tmp[5] = state[9];
	tmp[6] = state[14];
	tmp[7] = state[3];

	tmp[8] = state[8];
	tmp[9] = state[13];
	tmp[10] = state[2];
	tmp[11] = state[7];
	
	tmp[12] = state[12];
	tmp[13] = state[1];
	tmp[14] = state[6];
	tmp[15] = state[11];

	for (int i = 0; i < 16; i++) {
		state[i] = tmp[i];
	}
}

void MixColumns(byte *state) {
	byte tmp[16];

	tmp[0]  = (byte) mul2[state[0]] ^ mul3[state[1]] ^ state[2] ^ state[3];
	tmp[1]  = (byte) state[0] ^ mul2[state[1]] ^ mul3[state[2]] ^ state[3];
	tmp[2]  = (byte) state[0] ^ state[1] ^ mul2[state[2]] ^ mul3[state[3]];
	tmp[3]  = (byte) mul3[state[0]] ^ state[1] ^ state[2] ^ mul2[state[3]];

	tmp[4]  = (byte) mul2[state[4]] ^ mul3[state[5]] ^ state[6] ^ state[7];
	tmp[5]  = (byte) state[4] ^ mul2[state[5]] ^ mul3[state[6]] ^ state[7];
	tmp[6]  = (byte) state[4] ^ state[5] ^ mul2[state[6]] ^ mul3[state[7]];
	tmp[7]  = (byte) mul3[state[4]] ^ state[5] ^ state[6] ^ mul2[state[7]];

	tmp[8]  = (byte) mul2[state[8]] ^ mul3[state[9]] ^ state[10] ^ state[11];
	tmp[9]  = (byte) state[8] ^ mul2[state[9]] ^ mul3[state[10]] ^ state[11];
	tmp[10] = (byte) state[8] ^ state[9] ^ mul2[state[10]] ^ mul3[state[11]];
	tmp[11] = (byte) mul3[state[8]] ^ state[9] ^ state[10] ^ mul2[state[11]];

	tmp[12] = (byte) mul2[state[12]] ^ mul3[state[13]] ^ state[14] ^ state[15];
	tmp[13] = (byte) state[12] ^ mul2[state[13]] ^ mul3[state[14]] ^ state[15];
	tmp[14] = (byte) state[12] ^ state[13] ^ mul2[state[14]] ^ mul3[state[15]];
	tmp[15] = (byte) mul3[state[12]] ^ state[13] ^ state[14] ^ mul2[state[15]];

	for (int i = 0; i < 16; i++) {
		state[i] = tmp[i];
	}
}

int main() {
    byte a = 0xb4;
    cout << a;
    return 0;
}