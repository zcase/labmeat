import numpy as np
import freud
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import voronoi_plot_2d

# The code below is from the following link:
# https://freud.readthedocs.io/en/stable/examples/module_intros/Voronoi-Voronoi.html#

def draw_voronoi(box, points, cells, nlist=None, color_by_sides=False):
    ax = plt.gca()
    # Draw Voronoi cells
    patches = [plt.Polygon(cell[:, :2]) for cell in cells]
    patch_collection = matplotlib.collections.PatchCollection(patches, edgecolors='black', alpha=0.4)
    cmap = plt.cm.Set1

    if color_by_sides:
        colors = [len(cell) for cell in voro.polytopes]
    else:
        colors = np.random.permutation(np.arange(len(patches)))

    cmap = plt.cm.get_cmap('Set1', np.unique(colors).size)
    bounds = np.array(range(min(colors), max(colors)+2))

    patch_collection.set_array(np.array(colors))
    patch_collection.set_cmap(cmap)
    patch_collection.set_clim(bounds[0], bounds[-1])
    ax.add_collection(patch_collection)

    # Draw points
    plt.scatter(points[:,0], points[:,1], c=colors)
    plt.title('Voronoi Diagram')
    plt.xlim((-box.Lx/2, box.Lx/2))
    plt.ylim((-box.Ly/2, box.Ly/2))

    # Set equal aspect and draw box
    ax.set_aspect('equal', 'datalim')
    box_patch = plt.Rectangle([-box.Lx/2, -box.Ly/2], box.Lx, box.Ly, alpha=1, fill=None)
    ax.add_patch(box_patch)

    # Draw neighbor lines
    if nlist is not None:
        bonds = np.asarray([points[j] - points[i] for i, j in zip(nlist.index_i, nlist.index_j)])
        box.wrap(bonds)
        line_data = np.asarray([[points[nlist.index_i[i]],
                                 points[nlist.index_i[i]]+bonds[i]] for i in range(len(nlist.index_i))])
        line_data = line_data[:, :, :2]
        line_collection = matplotlib.collections.LineCollection(line_data, alpha=0.3)
        ax.add_collection(line_collection)

    # Show colorbar for number of sides
    if color_by_sides:
        cb = plt.colorbar(patch_collection, ax=ax, ticks=bounds, boundaries=bounds)
        cb.set_ticks(cb.formatter.locs + 0.5)
        cb.set_ticklabels((cb.formatter.locs - 0.5).astype('int'))
        cb.set_label("Number of sides", fontsize=12)
    plt.show()


def hexagonal_lattice(rows=3, cols=3, noise=0):
    # Assemble a hexagonal lattice
    points = []
    for row in range(rows*2):
        for col in range(cols):
            x = (col + (0.5 * (row % 2)))*np.sqrt(3)
            y = row*0.5
            points.append((x, y, 0))
    points = np.asarray(points)
    points += np.random.multivariate_normal(mean=np.zeros(3), cov=np.eye(3)*noise, size=points.shape[0])
    # Set z=0 again for all points after adding Gaussian noise
    points[:, 2] = 0

    # Wrap the points into the box
    box = freud.box.Box(Lx=cols*np.sqrt(3), Ly=rows, is2D=True)
    points = box.wrap(points)
    return box, points

if __name__ == "__main__":
    points = np.array([
        [-0.5, -0.5],
        [0.5, -0.5],
        [-0.5, 0.5],
        [0.5, 0.5]])
    plt.scatter(points[:,0], points[:,1])
    plt.title('Points')
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.show()

    # We must add a z=0 component to this array for freud
    points = np.hstack((points, np.zeros((points.shape[0], 1))))

    L = 6
    box = freud.box.Box.square(L)
    voro = freud.voronoi.Voronoi(box, L/2)

    cells = voro.compute(box=box, positions=points).polytopes
    print(cells)

    draw_voronoi(box, points, voro.polytopes)

    # Compute the Voronoi diagram and plot
    box, points = hexagonal_lattice()
    voro = freud.voronoi.Voronoi(box, np.max(box.L)/2)
    voro.compute(box=box, positions=points)
    draw_voronoi(box, points, voro.polytopes)

    box, points = hexagonal_lattice(rows=4, cols=4, noise=0.04)
    voro = freud.voronoi.Voronoi(box, np.max(box.L)/2)
    voro.compute(box=box, positions=points)
    draw_voronoi(box, points, voro.polytopes)

    draw_voronoi(box, points, voro.polytopes, color_by_sides=True)

    voro.computeVolumes()
    plt.hist(voro.volumes)
    plt.title('Voronoi cell volumes')
    plt.show()

    voro.computeNeighbors(box=box, positions=points)
    nlist = voro.nlist
    draw_voronoi(box, points, voro.polytopes, nlist=nlist)


    print(type(voro.voronoi))
    voronoi_plot_2d(voro.voronoi)
    plt.title('Raw Voronoi diagram including buffer points')
    plt.show()

    voro.voronoi