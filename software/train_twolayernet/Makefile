TARGET = train_twolayernet
C_SRCS += $(TARGET).c

LINK_OBJS += $(abspath ./libfixmath/libfixmath/fix16.o)
LINK_OBJS += $(abspath ./libfixmath/libfixmath/fix16_exp.o)
# LINK_OBJS += $(abspath ./libfixmath/libfixmath/fix16_sqrt.o)
# LINK_OBJS += $(abspath ./libfixmath/libfixmath/fix16_str.o)
# LINK_OBJS += $(abspath ./libfixmath/libfixmath/fix16_trig.o)
# LINK_OBJS += $(abspath ./libfixmath/libfixmath/fract32.o)
LINK_OBJS += $(abspath ./libfixmath/libfixmath/uint32.o)

LINK_OBJS += $(abspath ./t10k-images-idx3-ubyte.o)
LINK_OBJS += $(abspath ./t10k-labels-idx1-ubyte.o)
LINK_OBJS += $(abspath ./train-images-idx3-ubyte.o)
LINK_OBJS += $(abspath ./train-labels-idx1-ubyte.o)

CFLAGS += -O3 -fno-builtin-printf

CFLAGS += -std=gnu11
CFLAGS += -O3
CFLAGS += -g
CFLAGS += -L$(abspath ./libfixmath/libfixmath)
CFLAGS += -I$(abspath ./libfixmath/libfixmath)
# CFLAGS += -lfixmath
CFLAGS += -lm

GPERF := off

DEFINES :=
DEFINES += FIXMATH_NO_CACHE

ifeq ($(GPERF), on)
	DEFINES += GPERF
	CFLAGS += -fno-inline
	CFLAGS += -g3
	LOPTS += -lprofiler
endif

CFLAGS += $(addprefix -D, $(DEFINES))

LOPTS += -lm

all: train_data t10k_data run

run: train_twolayernet_run train_twolayernet_float_run train_twolayernet_fix16_run

train_data: train-images-idx3-ubyte train-labels-idx1-ubyte
t10k_data: t10k-images-idx3-ubyte t10k-labels-idx1-ubyte

%-ubyte.o: %-ubyte
	$(OBJCOPY) -I binary -O elf32-littleriscv -B riscv --rename-section .data=.rodata $(notdir $<) $(notdir $@)

%-ubyte: %-ubyte.gz
	gunzip $^

%.gz:
	wget "http://yann.lecun.com/exdb/mnist/$@"

prof: train_twolayernet.prof train_twolayernet_float.prof train_twolayernet_fix16.prof

train_twolayernet.prof: train_twolayernet_run
	google-pprof --callgrind train_twolayernet $@ > train_twolayernet.cg
train_twolayernet_float.prof: train_twolayernet_float_run
	google-pprof --callgrind train_twolayernet_float $@ > train_twolayernet_float.cg
train_twolayernet_fix16.prof: train_twolayernet_fix16_run
	google-pprof --callgrind train_twolayernet_fix16 $@ > train_twolayernet_fix16.cg

#clean:
#	rm -rf train_twolayernet train_twolayernet_float train_twolayernet_fix16
#	rm -rf train_twolayernet_run train_twolayernet_float_run train_twolayernet_fix16_run
#	rm -rf *.prof *.cg

BSP_BASE = ../../bsp
include $(BSP_BASE)/env/common.mk

