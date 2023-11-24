# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -std=c++11

# Target executable name
TARGET = main

# Source files
SRC = main.c

# Compile and link
$(TARGET): $(SRC)
	@$(CXX) $(CXXFLAGS) -o $@ $^ $(shell pkg-config --cflags --libs opencv4)

# Clean
clean:
	rm -f $(TARGET)
