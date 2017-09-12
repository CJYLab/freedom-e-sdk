#include <stdio.h>

// The CSR encodings are in this header.
#include "encoding.h"

// The mcycle counter is 64-bit counter, but since
// Freedom E platforms use RV32, we must access it as
// 2 32-bit registers. At 256MHz, the lower bits will
// roll over approx. every 5 seconds, so we check for
// rollover with this routine as suggested by the
// RISC-V Priviledged Architecture Specification.

#if __riscv_xlen == 64
#define rdmcycle(x)  {					\
    uint64_t hi;					\
    __asm__ __volatile__ ("1:\n\t"			\
			  "csrr %0, mcycle\n\t"		\
			  : "=r" (hi)) ;		\
    *(x) = hi;		 				\
  }
#else
#define rdmcycle(x)  {				       \
    uint32_t lo, hi, hi2;			       \
    __asm__ __volatile__ ("1:\n\t"		       \
			  "csrr %0, mcycleh\n\t"       \
			  "csrr %1, mcycle\n\t"	       \
			  "csrr %2, mcycleh\n\t"       \
			  "bne  %0, %2, 1b\n\t"			\
			  : "=r" (hi), "=r" (lo), "=r" (hi2)) ;	\
    *(x) = lo | ((uint64_t) hi << 32); 				\
  }
#endif


// The minstret counter is 64-bit counter, but
// Freedom E platforms use RV32, we must access it as
// 2 32-bit registers, same as for mcycle.

#if __riscv_xlen == 64
#define rdminstret(x)  {				\
    uint64_t hi;					\
    __asm__ __volatile__ ("1:\n\t"			\
			  "csrr %0, minstret\n\t"	\
			  : "=r" (hi)) ;		\
    *(x) = hi;						\
  }
#else
#define rdminstret(x)  {			       \
    uint32_t lo, hi, hi2;			       \
    __asm__ __volatile__ ("1:\n\t"		       \
			  "csrr %0, minstreth\n\t"       \
			  "csrr %1, minstret\n\t"	       \
			  "csrr %2, minstreth\n\t"       \
			  "bne  %0, %2, 1b\n\t"			\
			  : "=r" (hi), "=r" (lo), "=r" (hi2)) ;	\
    *(x) = lo | ((uint64_t) hi << 32); 				\
  }
#endif

extern char _binary_t10k_images_idx3_ubyte_start[];
extern char _binary_t10k_images_idx3_ubyte_end[];

int main()
{

  uint64_t before_cycle;
  uint64_t before_instret;
  
  uint64_t after_cycle;
  uint64_t after_instret;

  volatile char *p_data = _binary_t10k_images_idx3_ubyte_start;
  
  printf ("Flash Memory Latency Measurement\n");
  
  for (int i = 0; i < 28*28; i++){
    rdmcycle(&before_cycle);
    rdminstret(&before_instret);

    volatile char result = p_data[i];
    
    rdmcycle(&after_cycle);
    rdminstret(&after_instret);

	if ((i % 100) == 0) {
	  printf ("Start:%ld, Stop:%ld, Cycle=%ld\n",
			  (uint32_t)(before_cycle),
			  (uint32_t)after_cycle,
			  (uint32_t)(after_cycle - before_cycle));
	}
  }

  printf ("Flash Memory Throughput Measurement\n");
  
  rdmcycle(&before_cycle);
  rdminstret(&before_instret);
  for (int i = 0; i < 28*28*10; i++){
    volatile int result = p_data[i];
  }
  rdmcycle(&after_cycle);
  rdminstret(&after_instret);
  printf ("Start:%ld, Stop:%ld, Cycle=%ld\n",
		  (uint32_t)(before_cycle),
		  (uint32_t)after_cycle,
		  (uint32_t)(after_cycle - before_cycle));

  return 0;
}
