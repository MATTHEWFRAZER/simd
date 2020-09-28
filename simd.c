#include <stdio.h>
#include <xmmintrin.h>
#include <stdint.h>
#include <string.h>

void print128_num(__m128 var)
{
    int val[4];
	_mm_store_pd((float *)val, var);
    printf("Numerical: %i %i %i %i \n", 
           val[0], val[1], val[2], val[3]);
}

int main(void) {
  int i;
  int ptr[4] = {5, 7, 3, 2};
  int ptr2[4] = {3, 3, 3, 3};
  int ptr3[4] = {6, 6, 6, 6};
  int ptr4[4] = {-1, -1, -1, -1};
  int ptr5[4] = {1, 1, 1, 1};
  int ptr6[4] = {0,0,0,0};

  __m128 x = _mm_load_ps((float *) ptr);
  __m128 y = _mm_load_ps((float *) ptr2);
  __m128 yy = _mm_load_ps((float *) ptr3);
  __m128 mask = _mm_load_ps((float *) ptr4);
  __m128 zeros = _mm_load_ps((float *) ptr5);
  __m128 mul = _mm_load_ps((float *) ptr5);
  __m128 z = _mm_cmplt_ps(x, y);
  print128_num(z);
  __m128 xx = _mm_cmplt_ps(x, yy);
  print128_num(xx);
  __m128 xx2 = _mm_xor_ps(z, mask);
  print128_num(xx2);
  __m128 xx3 = _mm_and_ps(xx2, xx);
  print128_num(xx3);
  __m128 result1 = _mm_and_ps(x, xx3);
  print128_num(result1);

__m128 result_last = _mm_xor_ps(xx3, mask);
__m128 a_and = _mm_and_ps(result1, xx3);
__m128 b_and = _mm_and_ps(result_last, mask);

__m128 selected = _mm_or_ps(a_and,b_and);
print128_num(selected);
  
  return 0;
}
