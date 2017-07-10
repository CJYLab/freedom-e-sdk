TARGET = train_twolayernet_fix16
C_SRCS += train_twolayernet_fix16.c
CFLAGS += -O2 -fno-builtin-printf
CFLAGS += -I$(abspath libfixmath/libfixmath/)
CFLAGS += -DFIXMATH_NO_OVERFLOW

LDFLAGS += -L$(abspath libfixmath/libfixmath/)
LDFLAGS += -lfixmath

LINK_DEPS += libfixmath.a

all: $(TARGET)

libfixmath.a:
	$(MAKE) -C ./libfixmath/libfixmath CC=$(CC) CC_FLAGS="$(CFLAGS)" LD_FLAGS="$(LDFLAGS)" RISCV_ARCH=$(RISCV_ARCH) RISCV_ABI=$(RISCV_ABI) AR=$(AR)

BSP_BASE = ../../bsp
include $(BSP_BASE)/env/common.mk
