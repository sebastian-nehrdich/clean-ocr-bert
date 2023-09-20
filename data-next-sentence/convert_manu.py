import os
import sys
import random 

def create_pseudo_ocr(path):
    lines = []
    result = ""
    with open(path, 'r') as f:
        for line in f:
            tokens = line.split()
            windowsize = random.randint(10, 15)
            for i in range(0, len(tokens), windowsize):
                result += " ".join(tokens[i:i+windowsize]) + "\n"
    with open(path + "_processed", 'w') as f:
        f.write(result)

if __name__ == "__main__":
    create_pseudo_ocr(sys.argv[1])
    
    
