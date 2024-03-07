#include "slice.h"
#include "table.h"

void initIO();
/*
 * ввод-вывод графов
 */
void readfromDimageA(char *fn, Table *T );
// read matrix of adjacency from dimage format ('a' l r w)
void readfromDimageW(char *fn, Table *T);
void readfromDimageC(char *fn, Table *T);// транспонированнаяs readfromDimageW
// read matrix of weights from dimage format ('a' l r w)
void readfromDimageWC(char *fn, Table *Weight, Table *Cost);
void readfromDimageL(char *fn, Table *L, Table *R);
// read list of unweighed arcs from dimage format ('a' l r w)
void readfromDimageL(char *fn, Table *L, Table *R, Table *W);
// read list of weighted arcs from dimage format ('a' l r w)
void writeStrip(char *fn, LongPointer *d_tab, int i); // для контроля одну полосу из матрицы весов или Left/Right/Weight
void printStrip(LongPointer *d_tab, int i);
void writetoDimageA(char *fn, Table *T );
// write matrix of adjacency to dimage format ('a' l r w)
void writetoDimageW(char *fn, Table *T);
// write matrix of weights to dimage format ('a' l r w)
void writetoDimageL(char *fn, Table *L, Table *R);
// write list of unweighed arcs to dimage format ('a' l r w)
void writetoDimageL(char *fn, Table *L, Table *R, Table *W);
// write list of weighted arcs to dimage format ('a' l r w)


/*
 *  ввод-вывод нуклиотидных последовательностей
 */
