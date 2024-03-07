
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include "MST.h"
#include "star_kernel.h"
#include "iostar.h"
#include "basic.h"

void MSTPaths(Table *left, Table *right, Table *weight, Table *code, Slice *S, Slice *T, Table *M1)
{ unsigned int i,k,l;

  Slice *S1;
  Slice *N12;
  Slice *N2;
  Slice *X,*Y,*Z;
  Slice *node,*node1;

  i=left->length;
  k=code->length;
  S1= new Slice;
  S1->Init(i);
  N12= new Slice;
  N12->Init(i);
  N2= new Slice;
  N2->Init(i);
  X=new Slice;
  X->Init(i);
  Z=new Slice;
  Z->Init(i);
  Y= new Slice;
  Y->Init(k);
  l=code->size;
  node1= new Slice;
  node1->Init(l);
  node=new Slice;
  node->Init(l);
 //-----------------------------------------------------------------
   N12->CLR(); N2->CLR(); Y->SET();
   T->CLR(); M1->SetCol(N12,1);
   code->GetRow(node,1);
      S1->assign(S); Z->assign(S);
      while  (Z->SOME())
      {
    	  MATCH(left,S1,node,X);
          N12->OR(X);
          MATCH(right,S1,node,X);
          N2->OR(X);
            X->assign(N12); X->AND(N2); X->NOT();
            S1->AND(X);
 // Positions of edges forming a cycle are deleted from the slice S.
            Z->assign(N12); Z->OR(N2); Z->AND(S1);
// Positions of candidates for including into T(S) are selected by ones in the slice Z.
            if (Z->SOME())
            {
                MIN(weight,Z,X); i=X->FND();
                T->set(i,1); S1->set(i,0);
// The edge from the i-th position is added to T(S).
                  if (N12->get(i)==1)
                  {
                      right->GetRow(node,i);
                      left->GetRow(node1,i);
                  }
                  else
                  {
                	  right->GetRow(node1,i);
                	  left->GetRow(node,i);
                  }
// A new vertex is written in the variable node. }
                  MATCH(code,Y,node,X); 	k=X->FND();
                  MATCH(code,Y,node1,X); 	l=X->FND();
                  M1->GetCol(X,l); X->set(i,1);
                  M1->SetCol(X,k);
            }
      }
}
