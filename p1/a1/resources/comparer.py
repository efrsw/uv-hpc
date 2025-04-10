import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

def read_float_array(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()[2:]
            
            data_string = ' '.join(lines).strip()
            data_array = np.array([float(x) for x in data_string.split()])
            
        return data_array
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        sys.exit(1)

def diff_arrays(array1, array2):
    min_length = min(len(array1), len(array2))
    
    diff_array = np.ones(min_length, dtype=bool)
    
    diff_array = np.abs(array1 - array2)
    
    return diff_array

def parse_arguments():
    parser = argparse.ArgumentParser(description='Compare two data files and visualize differences.')
    parser.add_argument('file1', help='Path to the first data file')
    parser.add_argument('file2', help='Path to the second data file')
    parser.add_argument('-t', '--tolerance', type=float, default=1e-6, 
                        help='Tolerance for floating-point comparison (default: 1e-6)')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    for file_path in [args.file1, args.file2]:
        if not os.path.isfile(file_path):
            print(f"Error: File '{file_path}' does not exist")
            sys.exit(1)
    
    print(f"Reading data from {args.file1}...")
    array1 = read_float_array(args.file1)
    
    print(f"Reading data from {args.file2}...")
    array2 = read_float_array(args.file2)
    
    print(f"File 1 contains {len(array1)} values")
    print(f"File 2 contains {len(array2)} values")
    
    diff = diff_arrays(array1, array2)
    print(f"Difference value with tolerance {args.tolerance}: {(diff > args.tolerance).sum()}")

if __name__ == "__main__":
    main()
