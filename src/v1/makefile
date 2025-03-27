CC = gcc
CFLAGS = -Wall -O2 -pg

EXE = nn.exe
SRC = nn.c

all: $(EXE) 

OBJS = nn.o

$(EXE): $(OBJS)
	$(CC) $(CFLAGS) -o $(EXE) $(OBJS) -lm

%.o: %.c
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
