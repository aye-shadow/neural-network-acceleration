CC = nvcc
CFLAGS = -O3 -lcublas -pg

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
	gprof -b $(EXE) gmon.out > profile_results.txt

clean:
	rm -f $(EXE)
	rm -f $(OBJS)
	rm -f gmon.out
	rm -f profile_results.txt

.PHONY: all run clean