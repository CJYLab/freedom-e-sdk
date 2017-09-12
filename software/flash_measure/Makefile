TARGET = flash_measure
C_SRCS += flash_measure.c
CFLAGS += -O2 -fno-builtin-printf

LINK_OBJS += t10k-images-idx3-ubyte.o

BSP_BASE = ../../bsp
include $(BSP_BASE)/env/common.mk

t10k-images-idx3-ubyte.o: t10k-images-idx3-ubyte
	riscv64-unknown-elf-objcopy -I binary -O elf32-littleriscv -B riscv --rename-section .data=.rodata $< $@ 

