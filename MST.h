/*
 * MST.h
 *
 *  Created on: 28 дек. 2018 г.
 *      Author: tigra
 */

#ifndef MST_H_
#define MST_H_

#include "slice.h"
#include "table.h"




void MSTPaths(Table *left, Table *right, Table *weight, Table *code, Slice *S, Slice *T, Table *M1);



#endif /* MST_H_ */
