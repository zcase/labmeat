from VascularGenerator import VascularGenerator
import numpy as np

if __name__ == "__main__":

    # TODO: Make this (num_of_nodes) a command line argument
    vas_structure = VascularGenerator(max_range=100, num_of_nodes=3)
    vas_structure.print_images()