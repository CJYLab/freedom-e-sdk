#Project settings
PROJECT = libfixmath
LIB =
SRC = .
INC =

DEFINES += FIXMATH_NO_OVERFLOW
# DEFINES += FIXMATH_NO_64BIT

TARGET=riscv64-unknown-elf-

#Compiler settings
CPP = $(TARGET)gcc
CC 	= $(TARGET)gcc
AS 	= $(TARGET)gcc
LD 	= $(TARGET)gcc
AR 	= $(TARGET)ar

DEFINES += FIXMATH_NO_CACHE

CPP_FLAGS = -O2 $(INC) -Wall -Wextra -c -std=gnu11 -O3 -g  -g -march=rv32imac -mabi=ilp32 -mcmodel=medany $(addprefix -D, $(DEFINES))
CC_FLAGS  = -O2 $(INC) -Wall -Wextra -c -std=gnu11 -O3 -g  -g -march=rv32imac -mabi=ilp32 -mcmodel=medany $(addprefix -D, $(DEFINES))
AS_FLAGS  = $(CC_FLAGS) -D_ASSEMBLER_
LD_FLAGS = -Wall

# Find all source files
SRC_CPP = $(foreach dir, $(SRC), $(wildcard $(dir)/*.cpp))
SRC_C   = $(foreach dir, $(SRC), $(wildcard $(dir)/*.c))
SRC_S   = $(foreach dir, $(SRC), $(wildcard $(dir)/*.S))
OBJ_CPP = $(patsubst %.cpp, %.o, $(SRC_CPP))
OBJ_C   = $(patsubst %.c, %.o, $(SRC_C))
OBJ_S   = $(patsubst %.S, %.o, $(SRC_S))
OBJ     = $(OBJ_CPP) $(OBJ_C) $(OBJ_S)

# Compile rules.
.PHONY : all
all: $(PROJECT).a

$(PROJECT).a: $(OBJ)
	$(AR) rcs $(PROJECT).a $(OBJ)

$(OBJ_CPP) : %.o : %.cpp
	$(CPP) $(CPP_FLAGS) -o $@ $<

$(OBJ_C) : %.o : %.c
	$(CC) $(CC_FLAGS) -o $@ $<

$(OBJ_S) : %.o : %.S
	$(AS) $(AS_FLAGS) -o $@ $<



# Clean rules
.PHONY : clean
clean:
	rm -f $(PROJECT).a $(OBJ)
