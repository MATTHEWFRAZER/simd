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

void branchless_filter_example(void)
{
   int val[8] = {1, 2, 3, 4, 5, 6, 7, 8};
   int out[8] = {0};
   int ptr2[4] = {3, 3, 3, 3};
   int ptr3[4] = {6, 6, 6, 6};
   int i, j;
   int size = 8;
   int span = 1;

   __m128 lower_bound = _mm_load_ps((float *) ptr2);
   __m128 upper_bound = _mm_load_ps((float *) ptr3);
   ginverter = _mm_load_ps((float *)ginverter_ptr);


   for(i = 0; i < 8; i = i + 4)
   {
       __m128 toFilter = _mm_load_ps((float *)&val[i]);
       __m128 output;
       output = get_values_in_between(lower_bound, upper_bound, toFilter);
       _mm_store_ps((float *) &out[i], output);
   }

   i = 0;
   while(i + span < 8)
   {
      int isZero = (0 == out[i]);

      // out[i] = isZero ? out[i + span] : out[i];
      out[i] = (isZero * out[i + span]) + (!isZero * out[i]);

      // out[i + span] = isZero ?  0 : out[i + span];
      out[i + span] = !isZero * out[i + span];

      // i = isZero ? i : i + 1;
      i = i + !isZero;

      // we increase the span only if we have just encountered a
      // 0 at out[i] and at out[i + span], that is, if we have already used out[i + span], we can not use it again and must increment.
      // span points to what value we are considering to use as a replacement
      // int increment = 0;
      // if(0 == out[i] && 0 == out[i + span])
      // { increment = 1; }
      // span = span + increment;
      span = span + (0 == out[i]) * (0 == out[i + span]);
   }
   for(i = 0 ; i < size; ++i)
   {
     printf("%d\n", out[i]);
   }
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
