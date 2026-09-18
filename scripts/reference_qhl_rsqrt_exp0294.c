/* SDK QHL rsqrt reference executed by Qualcomm ISA simulator, no device data. */
#include <stdio.h>
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>
#include "qhmath_hvx_vector.h"
int main(int argc,char **argv) {
 if(argc!=3)return 2;
 FILE *in=fopen(argv[1],"rb"),*out=fopen(argv[2],"wb");if(!in||!out)return 3;
 float values[32] __attribute__((aligned(128)));
 size_t n;
 while((n=fread(values,4,32,in))) {
  for(size_t j=n;j<32;j++)values[j]=1.f;
  *(HVX_Vector *)values=qhmath_hvx_rsqrt_vf(*(HVX_Vector *)values);
  if(fwrite(values,4,n,out)!=n)return 4;
 }
 fclose(in);fclose(out);return 0;
}
