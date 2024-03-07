/*
 * knapsack.h
 *
 *  Created on: 05 янв. 2024 г.
 *      Author: tigra
 */

#ifndef KNAPSACK_H_
#define KNAPSACK_H_

#include "table.h"


// для заданного числа n генерирует масив весов от 1 до 100 и массив стоимости от 1 до 1000
void problem_generate(int n, int *w, int *c);

void branch_cut(int n,int *w, int W, Slice *T,Slice *B);
void branch_cut(int n,int *w, int W, int &k_t ,int &k_b);

void knapsack_exp();

#endif /* KNAPSACK_H_ */
