
# Copyright (C) Xilinx .Ltd, 2019. All rights reserved.
#

CC=$(CROSS_COMPILE)g++

CFLAGS = -shared -O3 -fPIC -std=c++11

RM=rm -rf $(1)
CP=cp $(1) $(2)

all: $(subst .cpp,.so,$(wildcard *dpu*.cpp))

%.so: %.cpp
	$(CC) $(CFLAGS)  $< -o $@ 

clean:
	$(call RM,*.o)
	$(call RM,*.so)

