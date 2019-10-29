from VascularGenerator import VascularGenerator

if __name__ == "__main__":
    vas_structure = VascularGenerator(max_range=100, num_of_nodes=3)
    vas_structure.print_images()