# Compiler to use
CXX = clang++
MPICXX = mpicxx

# Compilation flags
# -O3 for optimization
CXXFLAGS = -Wall -Wextra -std=c++17 -O3
# Flags for OpenMP
OMPFLAGS = -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp

# Source files
TASK1_SRC = task_1.cc
TASK1_CSV_SRC = task_1_csv.cc
TASK2_SRC = task_2.cc

# Executables
EXECS = task_1 task_2 task_1_csv

all: $(EXECS)

task_1: $(TASK1_SRC)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

task_1_csv: $(TASK1_CSV_SRC)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

task_2: $(TASK2_SRC)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

clean:
	rm -f task_1 task_2 *.o

.PHONY: all clean
