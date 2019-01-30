import math
import ase
import ase.geometry
import numpy as np
import vorosym
np.seterr(all='raise')

def solid_angle(coords):
    """solid_angle

    Compute solid angle of triangle using [1]

    [1] A. Van Oosterom and J. Strackee, "The Solid Angle of a Plane Triangle,"
    in IEEE Transactions on Biomedical Engineering, vol. BME-30, no. 2, pp. 125-126, Feb. 1983.

    :param coords: 3x3 ndarray where each row is xyz of one vertex
    :return: solid angle
    """
    numerator = np.linalg.det(coords)

    norms = np.linalg.norm(coords, axis=1)

    r0r1 = np.dot(coords[0], coords[1])
    r1r2 = np.dot(coords[1], coords[2])
    r2r0 = np.dot(coords[2], coords[0])

    denominator = np.prod(norms) + r0r1*norms[2] + r2r0*norms[1] + r1r2*norms[0]

    return abs(2 * math.atan2(numerator, denominator))

def polygon_solid_angle(verts):
    """polygon_solid_angle
    Compute solid angle of convex polygon by splitting the
    polygon into triangles

    :param verts: Nx3 array, where each row is xyz of one vertex
    :return: scalar solid angle of polygon
    """
    n = verts.shape[0]
    return sum(solid_angle(verts[[0, i, i+1]]) for i in range(n-1))

def axis_align_cell(atm):
    """axis_align_cell
    Rotate atoms and unit cell such that the unit cell
    matrix is lower triangular

    :param atm: ase.Atoms object to align
    """

    # Use QR-decomposition:
    # A^T = QR
    # A = R^T Q^T
    # A*Q = R^T
    # I.e. using R^T as unit cell corresponds
    # to rotating A using Q

    AT = atm.get_cell().T
    Q,R = np.linalg.qr(AT)
    fractional_coords = atm.get_scaled_positions()
    atm.set_cell(R.T)
    atm.set_scaled_positions(fractional_coords)

def ase_axis_align_cell(atm):
    fractional_coords = atm.get_scaled_positions()
    cell_par = ase.geometry.cell_to_cellpar(atm.get_cell())
    new_cell = ase.geometry.cellpar_to_cell(cell_par)
    atm.set_cell(new_cell)
    atm.set_scaled_positions(fractional_coords)
    atm.wrap()


class VoronoiFace():
    def __init__(self, neighbor, offset, area, normal, vertices, distance=None, solid_angle=None, symmetries=None):
        self.neighbor = neighbor
        self.neighbor_offset = offset
        self.area = area
        self.normal = normal
        self.vertices = vertices
        self.distance = distance
        self.solid_angle = solid_angle
        self.symmetries = symmetries

    def get_distance(self, point):
        return np.abs(np.dot(self.normal, self.vertices[0]-point))

    def get_solid_angle(self, viewpoint):
        return polygon_solid_angle(self.vertices - viewpoint)

    def __repr__(self):
        return "%s(%r,%r,%r,%d vertices)" % (self.__class__, self.neighbor, self.area, self.normal, self.vertices.shape[0])

class VoronoiCell():
    def __init__(self, atom_idx, atom_pos, cell_offset, volume):
        self.atom_idx = atom_idx
        self.atom_pos = atom_pos
        self.cell_offset = cell_offset
        self.volume = volume
        self.faces = []

    def add_face(self, face):
        self.faces.append(face)

    def __repr__(self):
        return "%s(%r,%r,%r,%d faces)" % (self.__class__, self.atom_idx, self.atom_pos, self.volume, len(self.faces))


def voro_tessellate(atm):
    ase_axis_align_cell(atm)
    a, b, c = atm.get_cell()
    basis = (a[0], b[0], b[1], c[0], c[1], c[2])
    res = vorosym.tessellate(basis, atm.get_scaled_positions())
    voro_cells = []
    for at in res:
        aidx, xyz, offset, volume, vert_coords, faces = at
        #assert np.all(abs(atm.get_positions(wrap=True)[aidx]-(xyz-np.dot(offset, atm.get_cell()))) < 1e-5)
        cell = VoronoiCell(aidx, xyz, offset, volume)
        for fa in faces:
            neigh_idx, neigh_dist, area, solid_angle, normal_vec, vert_inds, symmetries, neigh_offset = fa
            # Normal vector should be normalised to 1
            # if it is very small, it means that the face is degenerate
            if normal_vec.dot(normal_vec) > 1e-5:
                face = VoronoiFace(neigh_idx, neigh_offset, area, normal_vec, vert_coords[vert_inds], neigh_dist, solid_angle, symmetries)
                cell.add_face(face)
                #dist = (atm.get_positions(wrap=True)[aidx]-(atm.get_positions(wrap=True)[neigh_idx]+np.dot(neigh_offset-offset, atm.get_cell())))
        voro_cells.append(cell)

    return voro_cells
