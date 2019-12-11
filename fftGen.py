import random
max_lines = 33554432

random.seed(1)
f = open("fft.txt", "w+")
f.write(str(max_lines) + "\n")

for i in range(max_lines):
    real = random.randint(0,200000)
    im = random.randint(0, 200000)

    f.write(str(real) + " " + str(im) + "\n")

f.close()
