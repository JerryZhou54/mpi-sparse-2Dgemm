# Makefile for Hello World C program

# Compiler
CC = mpicxx

# Source file
SOURCE1 = spmat.c
SOURCE2 = spmat_ec.c

# Executable name
TARGET1 = spmat
TARGET2 = spmat_ec

all: $(TARGET1) $(TARGET2)

$(TARGET1): $(SOURCE1)
	$(CC) -o $(TARGET1) $(SOURCE1) -O3

$(TARGET2): $(SOURCE2)
	$(CC) -o $(TARGET2) $(SOURCE2) -O3

clean:
	rm -f $(TARGET1)
	rm -f $(TARGET2)