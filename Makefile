# Specify the default compiler
COMPILER ?= CLANG

# Specify the compilers
ifeq ($(COMPILER),CLANG)
  CXX = clang++
  CC = clang
else
  CXX = g++
  CC = gcc
endif

# Specify compiler flags
CXXFLAGS = -Wall -std=c++17
CFLAGS = -Wall

# Specify the linker flags
# Add `-lsqlite3` to link the SQLite3 library
LDFLAGS = -lm

# Specify the target file
TARGET = nn

# Separate C and C++ source files
CPP_SOURCES = src/server.cpp 
C_SOURCES = main.c

# Generate object files from sources
OBJECTS = $(CPP_SOURCES:.cpp=.o) $(C_SOURCES:.c=.o)

all: $(TARGET)

# Test files
TEST_SOURCES = tests/run-tests.cpp 
TEST_C_SOURCES =  
TEST_OBJECTS = $(TEST_SOURCES:.cpp=.o) $(TEST_C_SOURCES:.c=.o)
TEST_TARGET = run_tests

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

# A pattern rule to create ".o" object files from ".cpp" source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# A pattern rule to create ".o" object files from ".c" source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST_TARGET): $(TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TEST_TARGET) $(TEST_OBJECTS) $(LDFLAGS)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

# Add a clean target
clean:
	rm -f $(TARGET) $(OBJECTS) $(TEST_TARGET) $(TEST_OBJECTS)
