CC = nvcc
CFLAGS = -O3 -arch=sm_60 -Xcompiler -fopenmp
LIBS = -lcublas

EXE = nn
SRC = nn.cu
OBJS = nn.o

all: $(EXE)

$(EXE): $(OBJS)
	$(CC) $(CFLAGS) -o $(EXE) $(OBJS) $(LIBS)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

run: $(EXE)
	./$(EXE)

clean:
	rm -f $(EXE)
	rm -f $(OBJS)

.PHONY: all run clean