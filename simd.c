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
static int ginverter_ptr[4] = {-1, -1, -1, -1};

void initialize_inverter()
{
    ginverter = _mm_load_ps((float*)ginverter_ptr);
}

void initialize_globals()
{
    initialize_inverter();
    initialized = 1;
}

__m128 multiplex(__m128 a, __m128 b, __m128 selector)
{
    __m128 not_selector = _mm_xor_ps(selector, ginverter);
    __m128 a_and = _mm_and_ps(a, selector);
    __m128 b_and = _mm_and_ps(b, not_selector);
    return _mm_or_ps(a_and, b_and);
}

__m128 are_values_in_between(__m128 lower_bound, __m128 upper_bound, __m128 values)
{
    __m128 cmp_lower_bound = _mm_cmpge_ps(values, lower_bound);
    __m128 cmp_upper_bound = _mm_cmple_ps(values, upper_bound);
    return _mm_and_ps(cmp_lower_bound, cmp_upper_bound);
}

__m128 get_values_in_between(__m128 lower_bound, __m128 upper_bound, __m128 values)
{
    __m128 in_between = are_values_in_between(lower_bound, upper_bound, values);
    return _mm_and_ps(values, in_between);
}

void swap(__m128 *a, __m128 *b)
{
    *a = _mm_xor_ps(*a, *b);
    *b = _mm_xor_ps(*a, *b);
    *a = _mm_xor_ps(*a, *b);
}

int main(void) {
  int i;
  int ptr[4]  = {5, 7, 3, 2};
  int ptr2[4] = {3, 3, 3, 3};
  int ptr3[4] = {6, 6, 6, 6};
  int ptr4[4] = {-1, -1, -1, -1};
  int ptr5[4] = {1, 1, 1, 1};

  initialize_globals();

  __m128 values      = _mm_load_ps((float *) ptr);
  __m128 lower_bound = _mm_load_ps((float *) ptr2);
  __m128 upper_bound = _mm_load_ps((float *) ptr3);
  __m128 mask        = _mm_load_ps((float *) ptr4);
  __m128 zeros       = _mm_load_ps((float *) ptr5);
  __m128 mul         = _mm_setzero_ps();
  __m128 a           = _mm_setzero_ps();
  __m128 b           = _mm_load_ps((float *)ptr3);

  __m128 result      = get_values_in_between(lower_bound, upper_bound, values);
  __m128 in_between  = are_values_in_between(lower_bound, upper_bound, values);
  __m128 selected    = multiplex(result, mask, in_between);

  swap(&a, &b);

  print128_num(result);
  print128_num(selected);
  print128_num(a);
  print128_num(b);

  return 0;
}
