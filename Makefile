# Author: Drew Hans (github.com/DrewHans555)
# Description: This Makefile is for Neuronmancer (github.com/DrewHans555/Neuronmancer)
#
# How to use: 
#     type 'make', no quotes, to build Neuronmancer
#     type 'make clean', no quotes, to remove any compiled files
#
# Misc. Notes: 
#     @ is a built-in MAKE variable containing the target of each rule (neuronmancer in this case)
#     ^ is a built-in MAKE variable containing all dependencies of each rule (src variable in this case)
#     clean rule is marked as phony because it's target is not an actual file that will be generated

# define COMPILER variable: holds our choosen compiler
COMPILER=nvcc

# define src variable: collects the source files we want compiled (use $(wildcard *.c) when more than one src file)
src = main.cu

# define rule for building neuronmancer program
neuronmancer: $(src)
	$(COMPILER) $(^) -o $(@)

# define rule for cleaning up every target, in order to rebuild the whole program from scratch
.PHONY: clean
clean:
	rm -f ./neuronmancer

# end Neuronmancer Makefile
