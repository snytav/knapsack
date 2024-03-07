/*
 * table.h
 *
 *  Created on: Sep 1, 2015
 *      Author: snytav
 */

#ifndef TABLE_H_
#define TABLE_H_

#include "slice.h"
#include "param.h"
class Table{
//	LongPointer *d_table;
	Slice table[M];   //скорее всего приколы с памятью сдесь из-за таблицы меньшего размера
	LongPointer *slice_device_pointer_table,*d_slice_device_pointer_table;

	void InitDevicePointerTable();
public:
    unsigned int length,size;
	Table(){}
	~Table(){}
	int Init(unsigned int lg,unsigned int sz);

	// - доступ к i-му столбцу, как на чтение, так и на запись,
	Slice *row(int i);
	//доступ к i-ой строке, как на чтение, так и на запись.
	Slice *col(int i){return &(table[i-1]);}
	void SetRow(Slice *s,int i);
	void GetRow(Slice *s,int i);
	void GetRow_opt(Slice *s,int i);
	void SetCol(Slice *s,int i);
	void GetCol(Slice *s,int i);

	void readFromFile(char *fn);
	void readFromFileListAd_or(char *fn,int *eds);
	void readFromFileListAd_unor(char *fn,int *eds);
	void writeToFile(char *fn);
	LongPointer *get_device_pointer(){return d_slice_device_pointer_table;}
};

void readFromFileListLR(char *fn,int *eds, Table *left, Table *right);
__global__ void get_row(LongPointer *p,unsigned long long *d_v,int i,unsigned int size);

#endif /* TABLE_H_ */
