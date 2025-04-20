CC = nvcc
CFLAGS = -O2 -arch=sm_60 -rdc=true

EXE = nn
SRC = nn.cu
OBJS = nn.o

all: $(EXE)

$(EXE): $(OBJS)
	$(CC) $(CFLAGS) -o $(EXE) $(OBJS)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

run: $(EXE)
	./$(EXE)

clean:
	rm -f $(EXE)
	rm -f $(OBJS)

.PHONY: all run clean