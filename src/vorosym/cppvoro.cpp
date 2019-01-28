#include <vector>
#include <container_prd.hh>
#include <c_loops.hh>
#include <cell.hh>
#include <cmath>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <numpy/numpyconfig.h>
#include <numpy/arrayobject.h>
#include "rotationsym.h"

#define DISABLE_POLYDISPERSE

#define TARGET_GRID_NUM (6.0)

#define VORO_TESSELLATE_FAILURE     \
do                                  \
{                                   \
    Py_XDECREF(PyObj_arr_basis);    \
    Py_XDECREF(PyObj_arr_atoms);    \
    Py_XDECREF(PyObj_tup_return);   \
    Py_XDECREF(PyObj_arr_xyz);      \
    Py_XDECREF(PyObj_arr_vert);     \
    Py_XDECREF(PyObj_arr_normalvec);\
    Py_XDECREF(PyObj_arr_vertinds); \
    Py_XDECREF(PyObj_tup_faces);    \
    Py_XDECREF(PyObj_arr_symmetries);\
    Py_XDECREF(PyObj_arr_neighoffset);\
    Py_XDECREF(PyObj_arr_celloffset);\
    return NULL;                    \
} while(false)

#define GRAPH_DIST_FAILURE       \
do                                  \
{                                   \
    Py_XDECREF(PyObj_arr_edgelist); \
    Py_XDECREF(PyObj_arr_distancematrix);\
    return NULL;                    \
} while(false)

// Compute solid angle of triangle using [1]
//
// [1] A. Van Oosterom and J. Strackee, "The Solid Angle of a Plane Triangle,"
// in IEEE Transactions on Biomedical Engineering, vol. BME-30, no. 2, pp. 125-126, Feb. 1983.
static inline double solid_angle_triangle(
    double ax, double ay, double az,
    double bx, double by, double bz,
    double cx, double cy, double cz)
{
    double det = ax*(by*cz-bz*cy)-ay*(bx*cz-bz*cx)+az*(bx*cy-by*cx);
    double an = sqrt(pow(ax,2)+pow(ay,2)+pow(az,2));
    double bn = sqrt(pow(bx,2)+pow(by,2)+pow(bz,2));
    double cn = sqrt(pow(cx,2)+pow(cy,2)+pow(cz,2));
    double ab = ax*bx+ay*by+az*bz;
    double bc = bx*cx+by*cy+bz*cz;
    double ca = cx*ax+cy*ay+cz*az;
    double denom = an*bn*cn + ab*cn + ca*bn + bc*an;
    return fabs(2*atan2(det, denom));
}

// Compute solid angle of polygon by splitting it into smaller triangles
static double solid_angle_polygon(
        const std::vector<double> &verts // Vector of vertices coordinates
        )
{
    double total = 0;
    for (unsigned int i=1; i<(verts.size()/3)-1; i++)
    {
        total +=
            solid_angle_triangle(
                    verts[    0], verts[    1], verts[    2],
                    verts[i*3+0], verts[i*3+1], verts[i*3+2],
                    verts[i*3+3], verts[i*3+4], verts[i*3+5]
                    );
    }
    return total;
}

static PyObject *voro_tessellate(PyObject *self, PyObject *args) {
    PyObject *PyObj_arg_basis, *PyObj_arg_atoms;
    PyArrayObject *PyObj_arr_basis=NULL, *PyObj_arr_atoms=NULL;
    PyObject *PyObj_tup_return=NULL;

    // Variables used in per_atom loop,
    // but must be defined here to free on failure
    PyObject *PyObj_arr_xyz=NULL;
    PyObject *PyObj_arr_vert=NULL;
    PyObject *PyObj_arr_normalvec=NULL;
    PyObject *PyObj_arr_vertinds=NULL;
    PyObject *PyObj_tup_faces=NULL;
    PyObject *PyObj_arr_symmetries=NULL;
    PyObject *PyObj_arr_neighoffset=NULL;
    PyObject *PyObj_arr_celloffset=NULL;

    double ax, bx, by, cx, cy, cz;
    double cinv[6];
    npy_double *dptr;
    bool polydisperse;
    int natoms;
    std::vector<double> atomX, atomY, atomZ, atomR;
    std::vector<double> face_vert_rel;

    // Parse python input args
    if (!PyArg_ParseTuple(
            args, "OO", &PyObj_arg_basis, &PyObj_arg_atoms))
    {
        return NULL;
    }

    //
    // Convert input arguments to numpy arrays
    //
    PyObj_arr_basis = (PyArrayObject*) PyArray_FROMANY(
        PyObj_arg_basis,
        NPY_DOUBLE,
        1,
        1,
        NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);

    if (PyObj_arr_basis == NULL) return NULL;
    if (PyArray_SHAPE(PyObj_arr_basis)[0] != 6)
    {
        PyErr_SetString(PyExc_ValueError, "First argument must be an array of length 6");
        VORO_TESSELLATE_FAILURE;
    }

    PyObj_arr_atoms = (PyArrayObject*) PyArray_FROMANY(
        PyObj_arg_atoms,
        NPY_DOUBLE,
        2,
        2,
        NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (PyObj_arr_atoms == NULL) VORO_TESSELLATE_FAILURE;
    natoms = PyArray_SHAPE(PyObj_arr_atoms)[0];
    polydisperse = (PyArray_SHAPE(PyObj_arr_atoms)[1] == 4);
#ifdef DISABLE_POLYDISPERSE
    if ((PyArray_SHAPE(PyObj_arr_atoms)[1] != 3) && (PyArray_SHAPE(PyObj_arr_atoms)[1] != 4))
    {
        PyErr_SetString(PyExc_ValueError, "Coordinate rows must be x,y,z or x,y,z,r (polydisperse)");
        VORO_TESSELLATE_FAILURE;
    }
#else
    if (PyArray_SHAPE(PyObj_arr_atoms)[1] != 3)
    {
        PyErr_SetString(PyExc_ValueError, "Coordinate rows must be x,y,z ");
        VORO_TESSELLATE_FAILURE;
    }
#endif

    // Reserve number of atoms
    atomX.reserve(natoms);
    atomY.reserve(natoms);
    atomZ.reserve(natoms);
    atomR.reserve(natoms);

    // Get basis function parameters
    ax = ((npy_double *) PyArray_DATA(PyObj_arr_basis))[0];
    bx = ((npy_double *) PyArray_DATA(PyObj_arr_basis))[1];
    by = ((npy_double *) PyArray_DATA(PyObj_arr_basis))[2];
    cx = ((npy_double *) PyArray_DATA(PyObj_arr_basis))[3];
    cy = ((npy_double *) PyArray_DATA(PyObj_arr_basis))[4];
    cz = ((npy_double *) PyArray_DATA(PyObj_arr_basis))[5];

    // Compute inverse of basis matrix which is also lower triangular
    // This matrix is used to convert from cartesian coordinates to
    // fractional coordinates
    // | 0     |
    // | 1 2   |
    // | 3 4 5 |
    cinv[0] = 1/ax;
    cinv[1] = -bx/(ax*by);
    cinv[2] = 1/by;
    cinv[3] = (bx*cy/by-cx)/(cz*ax);
    cinv[4] = -cy/(by*cz);
    cinv[5] = 1/cz;

    dptr = (npy_double *) PyArray_DATA(PyObj_arr_atoms);
    for (int a=0; a<natoms; a++)
    {
        double x, y, z, r;
        x = *dptr++;
        y = *dptr++;
        z = *dptr++;
        if (polydisperse) {
            r = *dptr++;
        } else {
            r = 1;
        }
        // Store results
        atomX.push_back(x * ax + y * bx + z * cx);
        atomY.push_back(y * by + z * cy);
        atomZ.push_back(z * cz);
        atomR.push_back(r);
    }

    int griddim = (static_cast<int>(floor(pow(natoms/TARGET_GRID_NUM, 1.0/3.0)))) + 1;
#ifdef DISABLE_POLYDISPERSE
    // Create the cell
    voro::container_periodic_poly box(ax, bx, by, cx, cy, cz,
        griddim, griddim, griddim, 10);
    for (int a=0; a<natoms; a++) {
        box.put(a, atomX[a], atomY[a], atomZ[a], atomR[a]);
    }
#else
    voro::container_periodic box(ax, bx, by, cx, cy, cz,
        griddim, griddim, griddim, 10);
    for (int a=0; a<natoms; a++) {
        box.put(a, atomX[a], atomY[a], atomZ[a]);
    }
#endif


    PyObj_tup_return = PyTuple_New(natoms);
    if (PyObj_tup_return == NULL) VORO_TESSELLATE_FAILURE;

    // Output cell information
    voro::c_loop_all_periodic loop(box);
    loop.start();
    for (int a=0; a<natoms; a++) {

        int atom;
        double x, y, z, r;
        double ca, cb, cc;
        double volume;
        npy_intp dims[2];

        // Get current position
        loop.pos(atom, x, y, z, r);
        if (atom < 0 || atom >= natoms)
        {
            PyErr_SetString(PyExc_RuntimeError, "Unexpected Voro++ output");
            VORO_TESSELLATE_FAILURE;
        }

        // Save position in new xyz array
        dims[0]=3;
        PyObj_arr_xyz = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if(PyObj_arr_xyz == NULL) VORO_TESSELLATE_FAILURE;
        *((npy_double*) PyArray_GETPTR1((PyArrayObject *)PyObj_arr_xyz, 0)) = x;
        *((npy_double*) PyArray_GETPTR1((PyArrayObject *)PyObj_arr_xyz, 1)) = y;
        *((npy_double*) PyArray_GETPTR1((PyArrayObject *)PyObj_arr_xyz, 2)) = z;

        // The xyz position is not guarenteed to be within the unit cell,
        // Calculate the offset (i.e. which image) the position is in.
        // First convert to fractional coordinates
        ca = cinv[0]*x+cinv[1]*y+cinv[3]*z;
        cb = cinv[2]*y+cinv[4]*z;
        cc = cinv[5]*z;

        dptr = (npy_double *) PyArray_DATA(PyObj_arr_atoms);
        // Save offset in another array
        PyObj_arr_celloffset = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if(PyObj_arr_celloffset == NULL) VORO_TESSELLATE_FAILURE;
        *((npy_double*) PyArray_GETPTR1((PyArrayObject *)PyObj_arr_celloffset, 0)) = round(ca - dptr[3*atom+0]);
        *((npy_double*) PyArray_GETPTR1((PyArrayObject *)PyObj_arr_celloffset, 1)) = round(cb - dptr[3*atom+1]);
        *((npy_double*) PyArray_GETPTR1((PyArrayObject *)PyObj_arr_celloffset, 2)) = round(cc - dptr[3*atom+2]);

        // Compute the Voronoi cell
        voro::voronoicell_neighbor cell;
        if (!box.compute_cell(cell, loop)) {
            PyErr_SetString(PyExc_RuntimeError, "Voro++ tessellation failed");
            VORO_TESSELLATE_FAILURE;
        }

        // Get the volume
        volume = cell.volume();

        // Get the vertices coordinates
        std::vector<double> verts;
        cell.vertices(x, y, z, verts);
        // Create numpy array
        dims[0] = verts.size()/3;
        dims[1] = 3;
        PyObj_arr_vert = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        if(PyObj_arr_vert == NULL) VORO_TESSELLATE_FAILURE;
        for (unsigned int v=0; v<verts.size() / 3; v++) {
            *((npy_double*) PyArray_GETPTR2((PyArrayObject *)PyObj_arr_vert, v, 0)) = verts[v*3+0];
            *((npy_double*) PyArray_GETPTR2((PyArrayObject *)PyObj_arr_vert, v, 1)) = verts[v*3+1];
            *((npy_double*) PyArray_GETPTR2((PyArrayObject *)PyObj_arr_vert, v, 2)) = verts[v*3+2];
        }

        // Create tuple of faces
        PyObj_tup_faces = PyTuple_New(cell.number_of_faces());
        if(PyObj_tup_faces == NULL) VORO_TESSELLATE_FAILURE;

        // Gather face information
        std::vector<int> neigh, faceVerts;
        cell.neighbors(neigh);
        cell.face_vertices(faceVerts);
        std::vector<double> area, normal;
        cell.face_areas(area);
        cell.normals(normal);

        // Get face information
        int fvPos = 0;
        for (int f=0; f<cell.number_of_faces(); f++) {
            // Create normal vector
            dims[0] = 3;
            PyObj_arr_normalvec = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
            if(PyObj_arr_normalvec == NULL) VORO_TESSELLATE_FAILURE;
            *((npy_double*) PyArray_GETPTR1((PyArrayObject *)PyObj_arr_normalvec, 0)) = normal[3*f+0];
            *((npy_double*) PyArray_GETPTR1((PyArrayObject *)PyObj_arr_normalvec, 1)) = normal[3*f+1];
            *((npy_double*) PyArray_GETPTR1((PyArrayObject *)PyObj_arr_normalvec, 2)) = normal[3*f+2];

            int f_order = faceVerts[fvPos++];
            // Create vector of relative coordinates
            face_vert_rel.reserve(f_order*3);
            face_vert_rel.clear();
            // Create vector of vertex indices
            dims[0] = f_order;
            PyObj_arr_vertinds = PyArray_SimpleNew(1, dims, NPY_INT);
            if(PyObj_arr_vertinds == NULL) VORO_TESSELLATE_FAILURE;
            // Calculate distance to neighbor using first vertex
            // By projecting the vector from the cell center to
            // the vertex onto the normal vector.
            double neigh_distance = 2.0 * fabs(
                normal[3*f+0]*(verts[faceVerts[fvPos]*3+0]-x)+
                normal[3*f+1]*(verts[faceVerts[fvPos]*3+1]-y)+
                normal[3*f+2]*(verts[faceVerts[fvPos]*3+2]-z));

            // Determine which unit cell the neigbor is in (neighbor offset)
            //
            // Calculate coordinates of neighbor
            double nex, ney, nez, a, b, c;
            nex = normal[3*f+0]*neigh_distance+x;
            ney = normal[3*f+1]*neigh_distance+y;
            nez = normal[3*f+2]*neigh_distance+z;
            a = cinv[0]*nex+cinv[1]*ney+cinv[3]*nez;
            b = cinv[2]*ney+cinv[4]*nez;
            c = cinv[5]*nez;
            // Create vector for saving neighbor offset, i.e. which image is the neighbor in
            dims[0]=3;
            PyObj_arr_neighoffset = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
            if(PyObj_arr_neighoffset == NULL) VORO_TESSELLATE_FAILURE;

            dptr = (npy_double *) PyArray_DATA(PyObj_arr_atoms);
            *((npy_double*) PyArray_GETPTR1((PyArrayObject *)PyObj_arr_neighoffset, 0)) = round(a - dptr[3*neigh[f]+0]);
            *((npy_double*) PyArray_GETPTR1((PyArrayObject *)PyObj_arr_neighoffset, 1)) = round(b - dptr[3*neigh[f]+1]);
            *((npy_double*) PyArray_GETPTR1((PyArrayObject *)PyObj_arr_neighoffset, 2)) = round(c - dptr[3*neigh[f]+2]);

            for (int fv=0; fv<f_order; fv++) {
                // Write relative coordinates to vector
                face_vert_rel.push_back(verts[faceVerts[fvPos]*3+0] - x);
                face_vert_rel.push_back(verts[faceVerts[fvPos]*3+1] - y);
                face_vert_rel.push_back(verts[faceVerts[fvPos]*3+2] - z);
                // Write indices to numpy array
                *((npy_int*) PyArray_GETPTR1((PyArrayObject *)PyObj_arr_vertinds, fv)) = faceVerts[fvPos++];
            }
            // Compute rotational symmetries for face
            dims[0] = NUM_ROTATION_SYMMETRIES;
            PyObj_arr_symmetries = NULL;
            PyObj_arr_symmetries = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
            if(PyObj_arr_symmetries == NULL) VORO_TESSELLATE_FAILURE;
            if((normal[3*f]*normal[3*f]+normal[3*f+1]*normal[3*f+1]+normal[3*f+2]*normal[3*f+2] > 0.5))
            {
                get_symmetry_measures(
                    face_vert_rel,
                    &normal[3*f],
                    (npy_double *) PyArray_GETPTR1((PyArrayObject *) PyObj_arr_symmetries, 0));
            }
            else // Normal vector is degenerate, set all symmetries to 0
            {
                for (int si=0; si<NUM_ROTATION_SYMMETRIES; si++)
                {
                    *((npy_double *) PyArray_GETPTR1((PyArrayObject *) PyObj_arr_symmetries, si)) = 0;
                }
            }

            // Create (neigh, dist, area, solid_angle, normal_vec, vert_ind, symmetries, neighoffset) tuple
            PyObject *PyObj_tup_oneface = NULL;
            PyObj_tup_oneface = Py_BuildValue(
                "idddNNNN",
                neigh[f],
                neigh_distance,
                area[f],
                solid_angle_polygon(face_vert_rel),
                PyObj_arr_normalvec,
                PyObj_arr_vertinds,
                PyObj_arr_symmetries,
                PyObj_arr_neighoffset);
            if(PyObj_tup_oneface == NULL) VORO_TESSELLATE_FAILURE;

            // Insert tuple into tuple of faces
            PyTuple_SET_ITEM(PyObj_tup_faces, f, PyObj_tup_oneface);
        }
        PyObject *PyObj_tup_oneatom = NULL;
        // Create (atom_idx, xyz_arr, cell_offset_arr, volume, vert_cords_arr, faces) tuple
        PyObj_tup_oneatom = Py_BuildValue("iNNdNN", atom, PyObj_arr_xyz, PyObj_arr_celloffset, volume, PyObj_arr_vert, PyObj_tup_faces);
        if(PyObj_tup_oneatom == NULL) VORO_TESSELLATE_FAILURE;
        // Insert tuple into tuple of atoms
        PyTuple_SET_ITEM(PyObj_tup_return, atom, PyObj_tup_oneatom);

        // Increment loop counter
        loop.inc();
    }

    // Decrease reference count
    Py_DECREF(PyObj_arr_basis);
    Py_DECREF(PyObj_arr_atoms);

    //return PyLong_FromLong(42);
    return PyObj_tup_return;
}

static PyObject *graph_distance(PyObject *self, PyObject *args)
{
    PyObject *PyObj_arg_edgelist;
    PyArrayObject *PyObj_arr_edgelist = NULL;
    PyArrayObject *PyObj_arr_distancematrix = NULL;
    npy_intp output_dims[2] = {0, 0};
    npy_int *int_ptr, *int_ptr_distmat;
    int num_nodes;
    npy_intp k, i, j;
    npy_intp num_edges;
    if (!PyArg_ParseTuple(args, "Oi", &PyObj_arg_edgelist, &num_nodes))
    {
        return NULL;
    }
    //
    // Convert input arguments to numpy arrays
    //
    PyObj_arr_edgelist = (PyArrayObject*) PyArray_FROMANY(
        PyObj_arg_edgelist,
        NPY_INT,
        2,
        2,
        NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);

    if (PyObj_arr_edgelist == NULL) return NULL;
    if (PyArray_SHAPE(PyObj_arr_edgelist)[1] != 2)
    {
        PyErr_SetString(PyExc_ValueError, "First argument must be an Nx2 array");
        GRAPH_DIST_FAILURE;
    }
    num_edges = PyArray_SHAPE(PyObj_arr_edgelist)[0];

    // Create output array
    output_dims[0] = num_nodes;
    output_dims[1] = num_nodes;
    PyObj_arr_distancematrix = (PyArrayObject*) PyArray_SimpleNew(2, output_dims, NPY_INT);
    if(PyObj_arr_distancematrix == NULL) GRAPH_DIST_FAILURE;

    // Set all distances to infinity (MAX_INT)
    int_ptr = (npy_int *) PyArray_DATA(PyObj_arr_distancematrix);
    for (i=0; i<num_nodes*num_nodes; i++)
    {
        *int_ptr++ = NPY_MAX_INT;
    }

    // Initialize all the present edges to 1
    int_ptr = (npy_int *) PyArray_DATA(PyObj_arr_edgelist);
    for (i=0; i<num_edges; i++)
    {
        npy_intp to = *int_ptr++;
        npy_intp from = *int_ptr++;
        int_ptr_distmat = (npy_int *) PyArray_DATA(PyObj_arr_distancematrix);
        if ((to >= num_nodes) || (from >= num_nodes) || (to < 0) || (from < 0))
        {
            PyErr_SetString(PyExc_ValueError, "Invalid indices found in edge list");
            GRAPH_DIST_FAILURE;
        }
        int_ptr_distmat[to*num_nodes+from] = 1;
    }

    // Set all diagional elements to zero
    //int_ptr = (npy_int *) PyArray_DATA(PyObj_arr_distancematrix);
    //for (i=0; i<num_nodes; i++)
    //{
    //*int_ptr = 0;
    //int_ptr += num_nodes+1;
    //}

    for (k=0; k<num_nodes; k++)
    {
        for (i=0; i<num_nodes; i++)
        {
            for (j=0; j<num_nodes; j++)
            {
                int_ptr_distmat = (npy_int *) PyArray_DATA(PyObj_arr_distancematrix);
                npy_int sum = (int_ptr_distmat[i*num_nodes+k]
                        + int_ptr_distmat[k*num_nodes+j]);
                // Check for overflow
                if ((sum < int_ptr_distmat[i*num_nodes+k])
                        || (sum < int_ptr_distmat[k*num_nodes+j]))
                {
                    sum = NPY_MAX_INT;
                }
                if (int_ptr_distmat[i*num_nodes+j] > sum)
                {
                    int_ptr_distmat[i*num_nodes+j] = sum;
                }
            }
        }
    }
    return (PyObject *) PyObj_arr_distancematrix;
}

static PyMethodDef VoroMethods[] = {
    {"tessellate",  voro_tessellate, METH_VARARGS,
     "Tessellate."},
    {"graphdistance",  graph_distance, METH_VARARGS,
     "Graph Distance"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef cppvoro = {
    PyModuleDef_HEAD_INIT,
    "cppvoro",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    VoroMethods
};

PyMODINIT_FUNC
PyInit_cppvoro(void)
{
    PyObject *m;
    m = PyModule_Create(&cppvoro);
    if (m == NULL)
        return NULL;
    import_array();
    return m;
}
