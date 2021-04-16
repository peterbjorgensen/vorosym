import sys
sys.path.append("../vorosym/") # TODO delete
import ase
import numpy as np
from vorosym import voro_tessellate
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def atoms_to_graph_voronoi(atoms: ase.atoms, atom_to_node_fn, min_solid_angle=None):

    edges = []
    connections = []
    nodes = []

    atom_numbers = atoms.get_atomic_numbers()
    for ii in range(len(atoms)):
        nodes.append(atom_to_node_fn(atom_numbers[ii]))

    voronoi_cells = voro_tessellate(atoms)
    assert np.all(np.array([v.atom_idx for v in voronoi_cells]) == np.arange(len(atom_numbers)))

    for cell in voronoi_cells:
        total_area = sum(face.area for face in cell.faces)
        total_weighted_bond_length = sum(face.distance*face.solid_angle/(4*np.pi) for face in cell.faces)
        for face in cell.faces:
            dist = face.distance
            area = face.area
            normed_area = face.area/total_area
            sangle = face.solid_angle
            if min_solid_angle and (sangle < min_solid_angle):
                continue
            connections.append([face.neighbor, cell.atom_idx]) # [from, to]
            edges.append([dist, dist/total_weighted_bond_length, area, normed_area, sangle]+list(face.symmetries))

    return np.array(nodes), np.array(edges), np.array(connections)

def plot_faces(atoms, atomnumber=0, threshold=0.2):
    limits = np.diag(atoms.cell)
    points = atoms.get_positions()
    cells = voro_tessellate(atoms)
    cell = cells[atomnumber]
    verts = [fv.vertices for fv in cell.faces if fv.solid_angle > threshold]
    #print(4*np.pi*np.asarray(cell.face_areas())/cell.surface_area())
    print(atoms)
    print([fv.solid_angle for fv in cell.faces])

    colorscale = 100*['k']
    colorscale[ase.atoms.atomic_numbers["Au"]] = 'b'
    colorscale[ase.atoms.atomic_numbers["Ca"]] = 'r'
    colorscale[ase.atoms.atomic_numbers["Si"]] = 'g'
    colorscale = np.asarray(colorscale)
    facecolors = colorscale[atoms.numbers[[face.neighbor for face in cell.faces if face.solid_angle>threshold]]]


    fig = plt.figure()
    ax = Axes3D(fig)
    xmax = 4
    ax.set_xlim3d(-xmax,xmax)
    ax.set_ylim3d(-xmax,xmax)
    ax.set_zlim3d(-xmax,xmax)
    #ax.set_zlim3d(*lim[2])
    ax.set_aspect('equal','box')
    ax.add_collection3d(Poly3DCollection(verts, linewidths=1, edgecolors='k', facecolors=facecolors))
    ax.set_axis_off()
    #import pdb
    #pdb.set_trace()

def plot_individual_faces(atoms, atomnumber=0, threshold=0.2):
    limits = np.diag(atoms.cell)
    points = atoms.get_positions()
    cells = voro_tessellate(atoms)
    cell = cells[atomnumber]
    for fa in cell.faces:
        #if np.any(fa.symmetries>0.99) and fa.normal.dot(fa.normal) > 0.001:
        #if np.any(fa.symmetries[1:4]>0.99) and fa.normal.dot(fa.normal) > 0.001:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        verts = fa.vertices
        rotmat = rotation_matrix(fa.normal, [0,0,1])
        #print(np.linalg, rotmat)
        aligned = verts.dot(rotmat.T)[:,0:2]
        aligned -= aligned[0]
        polygon = matplotlib.patches.Polygon(aligned)
        ax.add_patch(polygon)
        rotnames=["C2", "C3", "C4", "C6", "D1", "D2", "D3", "D4", "D6"]
        label = " ".join("%s:%.2f" % (l,v) for (l,v) in zip(rotnames,list(fa.symmetries)))
        ax.set_title(label)
        ax.set_xlim([np.min(aligned[:,0]),np.max(aligned[:,0])])
        ax.set_ylim([np.min(aligned[:,1]),np.max(aligned[:,1])])
        ax.set_aspect('equal','box')
        ax.set_axis_off()

def rotation_matrix(a,b):
    if abs(-1 - a.dot(b) ) < 1e-7:
        R = -np.eye(len(a))
    else:
        # Return rotation matrix from vector a to b
        u = a/np.linalg.norm(a)
        v = b/np.linalg.norm(b)
        N = len(a)
        S = reflection(np.eye(N), v+u)
        R = reflection(S, v)
    return R

def reflection(u, n):
    # Reflection of u on hyperplane with normal n
    # Reshape n from array to column vector
    nc = np.reshape(n, (-1,1))
    v = u - 2*nc*(nc.T.dot(u))/(n.dot(n))
    return v

if __name__ == "__main__":
    import sys
    from ase.build import bulk
    from mpl_toolkits.mplot3d import Axes3D
    atoms = bulk('Si')
    atoms.set_chemical_symbols(['Si', 'F'])
    atoms = atoms.repeat(3)
    from mpl_toolkits.mplot3d import proj3d
    def orthogonal_proj(zfront, zback):
        a = (zfront+zback)/(zfront-zback)
        b = -2*(zfront*zback)/(zfront-zback)
        return np.array([[1,0,0,0],
                            [0,1,0,0],
                            [0,0,a,b],
                            [0,0,-0.0001,zback]])
    proj3d.persp_transformation = orthogonal_proj

    atomnumber=6
    plot_individual_faces(atoms, atomnumber=atomnumber)
    nodes, edges, connections = atoms_to_graph_voronoi(atoms, lambda x: x, min_solid_angle=0.2)

    N = len(nodes)
    apos = atoms.get_positions()
    Xn=[pos[0] for pos in apos]# x-coordinates of nodes
    Yn=[pos[1] for pos in apos]# y-coordinates
    Zn=[pos[2] for pos in apos]# z-coordinates
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xn, Yn, Zn)

    volume = atoms.get_volume()
    for con in connections:
        source = apos[con[1]]
        dest = apos[con[0]]
        x = (source[0], dest[0])
        y = (source[1], dest[1])
        z = (source[2], dest[2])
        if np.linalg.norm(source-dest) > (0.5 * volume**(1./3.)):
            pass
            #ax.plot(x,y,z, 'r', linewidth=0.1)
        else:
            ax.plot(x,y,z, 'k', linewidth=0.5)
    plt.show()

