#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

python3 fftGen.py 
mpicc -O2 -lm fft2.c -o fft
mpirun -np 8 fft < fft.txt
