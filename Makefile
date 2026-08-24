CC      = cc
CFLAGS  = -std=c11 -O2 -Wall -Wextra -fPIC
AR      = ar

BEARSRC := $(filter-out bear/bear_cudnn.c bear/bear_cudnn_cublas.c bear/bear_cudnn_cuda.c bear/bear_opt_test.c,$(wildcard bear/*.c))
# CPU-only core for now; GPU backends opt-in via BEAR_GPU=1
ifeq ($(BEAR_GPU),1)
CFLAGS += -DBEAR_GPU
else
BEARSRC := $(filter-out bear/bear_vulkan_soft.c,$(BEARSRC))
endif

OBJS := $(BEARSRC:.c=.o)

libbear.a: $(OBJS)
	$(AR) rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

test_bear_opt: libbear.a bear/bear_opt_test.c
	$(CC) $(CFLAGS) bear/bear_opt.c bear/bear_arena.c bear/bear_opt_test.c -lm -o $@
	./$@

tests: test_bear_opt

clean:
	rm -f *.a bear/*.o test_bear_opt

.PHONY: tests clean
