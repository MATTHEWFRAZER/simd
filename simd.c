#include <stdio.h>
#include <xmmintrin.h>
#include <stdint.h>
#include <string.h>

void print128_num(__m128 var)
{
  int val[4];
	_mm_store_ps((float *)val, var);
    printf("Numerical: %i %i %i %i \n",
           val[0], val[1], val[2], val[3]);
}

static int initialized;
static __m128 ginverter;
static int ginverter_ints[4] = {-1, -1, -1, -1};

void initialize_inverter()
{
    ginverter = _mm_load_ps((float *)ginverter_ints);
}

void initialize_globals()
{
    initialize_inverter();
    initialized = 1;
}

__m128 multiplex(__m128 a, __m128 b, __m128 selector)
{
    if (!initialized)
    {
        initialize_inverter();
    }
    __m128 not_selector = _mm_xor_ps(selector, ginverter);
    __m128 a_and = _mm_and_ps(a, selector);
    __m128 b_and = _mm_and_ps(b, not_selector);
    return _mm_or_ps(a_and, b_and);
}

int main(void) {
  int i;
  int ptr[4] = {5, 7, 3, 2};
  int ptr2[4] = {3, 3, 3, 3};
  int ptr3[4] = {6, 6, 6, 6};
  int ptr4[4] = {-1, -1, -1, -1};
  int ptr5[4] = {1, 1, 1, 1};
  int ptr6[4] = {0,0,0,0};
  printf("start");
  initialize_globals();

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

__m128 selected = multiplex(result1, mask, xx3);

print128_num(selected);

  return 0;
}
