/*******************************************************************
* Header File Declarations
********************************************************************/

#include <iostream>
#include <string>
#include <stdlib.h>
#include <fstream>

#include "../include/aeslib.hpp"
#include "../include/genlib.hpp"

using namespace std;

/*******************************************************************
* Pretty Printing Hex Values
********************************************************************/
byte hexmap[] = {'0', '1', '2', '3', '4', '5', '6', '7',
                '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};

string hex(byte *data, int len)
{
  	std::string s(len * 3, ' ');
  	for (int i = 0; i < len; ++i) {
	    s[3 * i]     = hexmap[(data[i] & 0xF0) >> 4];
	    s[3 * i + 1] = hexmap[data[i] & 0x0F];
	    s[3 * i + 2] = ' ';
  	}
  	return s;
}

/********************************************************************
* AES Implementation
*********************************************************************/

#define N_ROUNDS 10

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

void Round(byte *state, byte *RoundKey, bool isFinal=false) {
	SubBytes(state);
	ShiftRows(state);
	if(!isFinal) MixColumns(state);
	AddRoundKey(state, RoundKey);
}

void Cipher(byte *message, int msg_length, byte expandedKey[176], byte *cipher) {
	for(int i = 0; i < msg_length; i++) {
		cipher[i] = message[i];
	}

	for(int i = 0; i < msg_length; i+=16) {
		AddRoundKey(cipher + i, expandedKey);
		for(int n = 1; n <= N_ROUNDS; n++) {
			Round(cipher + i, expandedKey + (n)*16, n == 10);
		}
	}
}

void encrypt(string msg_path, string key_path, string out_path) {
	ifstream f_msg(msg_path);
	ifstream f_key(key_path);
	ofstream fout(out_path);

	if(f_msg && f_key) {

		f_msg.seekg(0, f_msg.end);
		int n = f_msg.tellg();
		cout << n;
		f_msg.seekg(0, f_msg.beg);

		byte message[n];
		byte cipher[n];
		byte key[16];
		byte expandedKey[176];

		f_msg.read(reinterpret_cast<char *> (message), n);
		f_key.read(reinterpret_cast<char *> (key), 16);

		// cout << hex(key, 16) << endl;

		KeyExpansion(key, expandedKey);
		Cipher(message, n, expandedKey, cipher);
		fout.write(reinterpret_cast<char *> (cipher), n);

		// cout << "MSG\n" << hex(message, n) << endl << endl; 
		// cout << "Cry\n" << hex(cipher, n) << endl << endl;	
	}
}

int main(int argc, char const *argv[])
{
	opts vars = get_defaults();

	int i, j, k;
	string path;
	for(i = vars.n_files_start; i <= vars.n_files_end; i+=vars.step) {
        for (j = 0; j < vars.m_batches; j++) {
            for(k = 0; k < i; k++) {
                path = vars.path + "/" + to_string(i) + "/" + to_string(j) + "/" + to_string(k);
                cout << path << endl;
				encrypt(path, path+"_key", path+"_cipher");
            }
        }
    }
	
	return 0;
}

/*
	byte message[] = {0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34};
	byte key[] = {0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
*/

/*
	path = vars.path + "/" + to_string(100) + "/" + to_string(0) + "/" + to_string(0);
	cout << path << endl;
	encrypt(path, path+"_key", path+"_cipher");
*/