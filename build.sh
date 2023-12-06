#!/bin/sh

set -xe

# clang -Wall -Wextra -o twice twice.c
# clang -Wall -Wextra -o gates gates.c -lm
# clang -Wall -Wextra -o xor xor.c -lm

clang -Wall -Wextra -o nn main.c -lm
clang++ -Wall -Wextra -o test ./tests/run-tests.cpp 
