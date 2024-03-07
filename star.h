/*
 * star.h
 *
 *  Created on: Feb 1, 2016
 *      Author: snytav
 */

//#ifndef STAR_H_
#define STAR_H_

#include "slice.h"
#include "table.h"

void warshall(Table *T);
void warshall_o(Table *T);
void warshall_c(Table *tab);
void warshall_c2(Table *tab);
//__global__ void Warshal_dev(LongPoint *d_tab, int k, unsigned long long int *d_x);
void warshall_c_ogr(Table *tab);
void CountTriangles(Table *tab, int * count);
void CountTrianglesOPT(Table *tab, int * count);
void DFS(Table *left, Table *right, Table *code, Slice *root, Table *NV, Slice *T, Slice *X);

//void DFS_CUDA(Table *left, Table *right, Table *code, Slice *root, Table *NV, Slice *T, Slice *X);
void dijkstra(Table *T, int s, int h, Slice *inf, Table *D);  // only connected graph!!!
// T - matrix of wights, inf - code of infinity, h - count of bits for inf
//#endif /* ALGORITHMS_H_ */

void dijkstra1(Table *T, int s, Table *D);
//h=32, inf=0xFFFF
void dijkstra2(Table *T, int s, Table *D); // can work with unconnected graph
void dijkstra2_opt(Table *T, int s, Table *D);

