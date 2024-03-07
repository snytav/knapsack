/*
 * slice.h
 *
 *  Created on: Sep 1, 2015
 *      Author: snytav
 */

#include "find.h"


#ifndef SLICE_H_
#define SLICE_H_

//#define SET 6666
//#define OR  6667

//типы данных: slice - вектор, table - таблица, word - строка. и integer (других типов нет в принципе).


//Операции над слайсами:
class Slice{
 //  bool word_flag;
   unsigned long long int *d_v;
//   int *d_first_non_zero;

public:
   unsigned int length, NN;
Slice(){}
~Slice(){}
int Init(unsigned int l);
//void Print();
//void set_from_host_array (unsigned long long int *f_h_v);
//void set_from_device_array (unsigned long long int *f_h_v);

unsigned long long int *get_device_pointer(){return d_v;}
unsigned long long int ToDigit();
unsigned long long int FromDigit(unsigned long long dig);
//заполнить единичками,
void SET();

//заполнить нулями,
void CLR();

//заполняет нулями до i-1 позиции, остальные заполняет единичками
void MASK(int i);
//доступ к i-ой компоненте слайса, как на чтение, так и на запись,
unsigned char get(int i);
void set(int i,unsigned char n);

// - выдает номер старшей единичке в слайсе Y (то, что делал)
int FND();
int FND1();
// - то же самое, но эту единичку заменяет на ноль
int STEP();

// - конвертирует слайс в строку (используется крайне редко).
//void CONVERT();

//Побитовые X and Y, not X, X or Y, X xor Y
Slice operator & (const Slice & b);
Slice operator | (const Slice & b);
Slice operator ^ (const Slice & b);
void AND(const Slice *b);
void OR(const Slice *b);
void XOR(const Slice *b);

void print(char *label,int row){print_device_bit_row(label,d_v,length,row,NN);};

void convert_to_string(char *s);

Slice operator = (char *s);

Slice operator = (Slice *s);
void assign(const Slice *s);
//void trim(int i, int h, const Slice *s);
Slice operator ~();
void NOT();
// - true, если X ненулевой.
bool SOME();
bool SOME1();
bool ZERO();
int NUMB();
int NUMB1();//через __popcll() и reduce;
void trim(int i, int h, Slice *s);
void shift_up(int i,Slice *s);
void shift_down(int i,Slice *s);
};
void __global__ shiftup_kernel(unsigned long long int *d_v, unsigned long long int *d_v_in,int i);
void __global__ shiftdown_kernel(unsigned long long int *d_v, unsigned long long int *d_v_in,int i);

void __device__ trim_(unsigned long long int *d_v, unsigned long long int *d_v_in,int i, int h);
void __device__ shiftup(unsigned long long int *d_v, unsigned long long int *d_v_in,int i);
void __device__ shiftdown(unsigned long long int *d_v, unsigned long long int *d_v_in,int i);
#endif /* SLICE_H_ */
