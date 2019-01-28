#include <iostream>
#include <vector>
#include <stack>
#include <tgmath.h>
#include <limits>
#include "rotationsym.h"

#define EPSILON (std::numeric_limits<double>::epsilon()*1000)
#define LEN(arr) ((sizeof(arr) / sizeof(arr)[0]))

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/ch_graham_andrew.h>
#include <CGAL/Polygon_2.h>
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Polygon_2<K> Polygon_2;
typedef K::Point_2 Point_2;

// Private function prototypes
static void apply_transforms(
    const std::vector<double> &verts,
    std::vector<double> &outpoints,
    const double transmat[][4],
    unsigned int num_transforms,
    bool include_identity
    );

static void print2d(const std::vector<double> &numbers, const char *prefix);
static void print3d(const std::vector<double> &numbers, const char *prefix);
static double get_convex_area2d(const double points[], int num_points);
static double get_signed_area2d(const std::vector<double> &verts);
static void axis_align_mirror2d(std::vector<double> &verts);

static const double matrot90[1][4]
{
    {0, -1, 1, 0}
};

static const double cmat2[1][4] = {
    {-1, 0, 0, -1},
};
static const double cmat3[2][4] = {
    {-0.5, -0.866025403784439, 0.866025403784439, -0.5},
    {-0.5, 0.866025403784438, -0.866025403784438, -0.5},
};
static const double cmat4[3][4] = {
    {0, -1, 1, 0},
    {-1, 0, 0, -1},
    {0, 1, -1, 0},
};
static const double cmat6[5][4] = {
    {0.5, -0.866025403784439, 0.866025403784439, 0.5},
    {-0.5, -0.866025403784439, 0.866025403784439, -0.5},
    {-1, 0, 0, -1},
    {-0.5, 0.866025403784438, -0.866025403784438, -0.5},
    {0.5, 0.866025403784439, -0.866025403784439, 0.5},
};
static const double dmat1[1][4] = {
    {1, 0, 0, -1},
};
static const double dmat2[2][4] = {
    {1, 0, 0, -1},
    {-1, 0, 0, 1},
};
static const double dmat3[3][4] = {
    {1, 0, 0, -1},
    {-0.5, 0.866025403784439, 0.866025403784439, 0.5},
    {-0.5, -0.866025403784438, -0.866025403784438, 0.5},
};
static const double dmat4[4][4] = {
    {1, 0, 0, -1},
    {0, 1, 1, 0},
    {-1, 0, 0, 1},
    {0, -1, -1, 0},
};
//static const double dmat5[5][4] = {
    //{1, 0, 0, -1},
    //{0.309016994374947, 0.951056516295154, 0.951056516295154, -0.309016994374947},
    //{-0.809016994374947, 0.587785252292473, 0.587785252292473, 0.809016994374947},
    //{-0.809016994374948, -0.587785252292473, -0.587785252292473, 0.809016994374948},
    //{0.309016994374947, -0.951056516295154, -0.951056516295154, -0.309016994374947},
//};
static const double dmat6[6][4] = {
    {1, 0, 0, -1},
    {0.5, 0.866025403784439, 0.866025403784439, -0.5},
    {-0.5, 0.866025403784439, 0.866025403784439, 0.5},
    {-1, 0, 0, 1},
    {-0.5, -0.866025403784438, -0.866025403784438, 0.5},
    {0.5, -0.866025403784439, -0.866025403784439, -0.5},
};

static void mat3x3mulvec(
    const double mat[9],
    const double vec[3],
    double outvec[3])
{
    outvec[0] = mat[0]*vec[0]+mat[1]*vec[1]+mat[2]*vec[2];
    outvec[1] = mat[3]*vec[0]+mat[4]*vec[1]+mat[5]*vec[2];
    outvec[2] = mat[6]*vec[0]+mat[7]*vec[1]+mat[8]*vec[2];
}

static void mat2x2mulvec(
    const double mat[4],
    const double vec[2],
    double outvec[2])
{
    outvec[0] = mat[0]*vec[0]+mat[1]*vec[1];
    outvec[1] = mat[2]*vec[0]+mat[3]*vec[1];
}

static double vec3dotp(
    const double veca[3],
    const double vecb[3])
{
    return veca[0]*vecb[0]+veca[1]*vecb[1]+veca[2]*vecb[2];
}
static double vec2dotp(
    const double veca[2],
    const double vecb[2])
{
    return veca[0]*vecb[0]+veca[1]*vecb[1];
}

//https://math.stackexchange.com/questions/432057/how-can-i-calculate-a-4-times-4-rotation-matrix-to-match-a-4d-direction-vector/433611#433611
static void reflection3d(
    const double A[9],
    const double n[3],
    double out[9])
{
    double nTA[3];
    double rhs[9];
    int i;
    double nTn = n[0]*n[0]+n[1]*n[1]+n[2]*n[2];
    nTA[0] = n[0]*A[0] + n[1]*A[3] + n[2]*A[6];
    nTA[1] = n[0]*A[1] + n[1]*A[4] + n[2]*A[7];
    nTA[2] = n[0]*A[2] + n[1]*A[5] + n[2]*A[8];

    rhs[0] = n[0]*nTA[0];
    rhs[1] = n[0]*nTA[1];
    rhs[2] = n[0]*nTA[2];
    rhs[3] = n[1]*nTA[0];
    rhs[4] = n[1]*nTA[1];
    rhs[5] = n[1]*nTA[2];
    rhs[6] = n[2]*nTA[0];
    rhs[7] = n[2]*nTA[1];
    rhs[8] = n[2]*nTA[2];

    double c = -2.0/nTn;

    for (i=0; i<9; i++)
    {
        out[i] = A[i] + rhs[i]*c;
    }

    return;
}

static void reflection2d(
    const double A[4],
    const double n[2],
    double out[4])
{
    double nTA[2];
    double rhs[4];
    int i;
    double nTn = n[0]*n[0]+n[1]*n[1];
    nTA[0] = n[0]*A[0] + n[1]*A[2];
    nTA[1] = n[0]*A[1] + n[1]*A[3];

    rhs[0] = n[0]*nTA[0];
    rhs[1] = n[0]*nTA[1];
    rhs[2] = n[1]*nTA[0];
    rhs[3] = n[1]*nTA[1];

    double c = -2.0/nTn;

    for (i=0; i<4; i++)
    {
        out[i] = A[i] + rhs[i]*c;
    }

    return;
}

/* Get rotation matrix that moves the vector "from" into "to"
 * where "from" and "to" must be unit vectors */
static void get_rotation3d(
    const double from[3],
    const double to[3],
    double out[9])
{
    const double eye[9] = {1,0,0, 0,1,0, 0,0,1};
    double S[9];
    double fpt[3];
    double scaling;
    if (fabs(-1 - vec3dotp(from, to)) < EPSILON)
    {
        // The two unit vectors are parallel but opposite
        // Find orthogonal rotation axis and rotate 180 degrees around it
        fpt[0] = 1;
        fpt[1] = 1;
        fpt[2] = 1;
        // Solve ax+by+cz = 0 to find rotation axis
        if ((fabs(to[0]) > fabs(to[1])) && (fabs(to[0]) > fabs(to[2])))
        {
            fpt[0] = (-to[1]-to[2])/to[0];
        }
        else if ((fabs(to[1]) > fabs(to[0])) && (fabs(to[1]) > fabs(to[2])))
        {
            fpt[1] = (-to[0]-to[2])/to[1];
        }
        else
        {
            fpt[2] = (-to[0]-to[1])/to[2];
        }
        // Normalise rotation axis to unit length
        scaling = 1/sqrt(fpt[0]*fpt[0]+fpt[1]*fpt[1]+fpt[2]*fpt[2]);
        fpt[0] *= scaling;
        fpt[1] *= scaling;
        fpt[2] *= scaling;
        // https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
        out[0] = 2*fpt[0]*fpt[0]-1;
        out[1] = 2*fpt[0]*fpt[1];
        out[2] = 2*fpt[0]*fpt[2];
        out[3] = 2*fpt[1]*fpt[0];
        out[4] = 2*fpt[1]*fpt[1]-1;
        out[5] = 2*fpt[1]*fpt[2];
        out[6] = 2*fpt[2]*fpt[0];
        out[7] = 2*fpt[2]*fpt[1];
        out[8] = 2*fpt[2]*fpt[2]-1;
    }
    else
    {
        fpt[0] = from[0]+to[0];
        fpt[1] = from[1]+to[1];
        fpt[2] = from[2]+to[2];
        reflection3d(eye, fpt, S);
        reflection3d(S, to, out);
    }
    return;
}

/* Get rotation matrix that moves the vector "from" into "to"
 * where "from" and "to" must be unit vectors */
static void get_rotation2d(
    const double from[2],
    const double to[2],
    double out[4])
{
    const double eye[4] = {1,0, 0,1};
    double S[4];
    double fpt[2];
    if (fabs(-1-vec2dotp(from, to)) < EPSILON)
    {
        // From and to are parallel but opposite
        // Rotation matrix flips all axes
        out[0] = -1;
        out[1] = 0;
        out[2] = 0;
        out[3] = -1;
    }
    else
    {
        fpt[0] = from[0]+to[0];
        fpt[1] = from[1]+to[1];
        reflection2d(eye, fpt, S);
        reflection2d(S, to, out);
    }
    return;
}

static void axis_align_3to2(
    const double normalvec[3],
    const std::vector<double> &inp,
    std::vector<double> &outp)
{
    const double zaxis[3] = {0, 0, 1};
    double R[9];
    double vec[3];
    unsigned int i;
    outp.reserve((inp.size()/3)*2);
    outp.clear();
    /* Get rotation matrix */
    get_rotation3d(normalvec, zaxis, R);
    /* Apply matrix to all points */
    for(i=0; i<inp.size()/3; i++)
    {
        mat3x3mulvec(R, &inp[i*3], vec);
        outp.push_back(vec[0]);
        outp.push_back(vec[1]);
        // Ignore last coordinate because it is constant
    }
}

// http://mathworld.wolfram.com/PolygonArea.html
static double get_signed_area2d(
    const std::vector<double> &verts
    )
{
    unsigned int i;
    double area = 0;
    for (i=0; i<verts.size()/2; i++)
    {
        // x_i*y_{i+1}-x_{i+1}*y_i
        area += verts[i*2+0]*verts[(i*2+3) % verts.size()];
        area -= verts[(i*2+2) % verts.size()]*verts[i*2+1];
    }
    area *= 0.5;
    return area;
}

//https://en.wikipedia.org/wiki/Centroid#Centroid_of_a_polygon
static void get_centroid2d(
    std::vector<double> verts,
    double centroid[2])
{
    unsigned int i;
    double x0, x1, y0, y1, a;
    double area = 0;
    centroid[0] = 0;
    centroid[1] = 0;
    for (i=0; i<verts.size()/2; i++)
    {
        x0 = verts[i*2];
        x1 = verts[(i*2+2) % verts.size()];
        y0 = verts[i*2+1];
        y1 = verts[(i*2+3) % verts.size()];
        a = x0*y1 - x1*y0;
        area += a;
        centroid[0] += (x0 + x1)*a;
        centroid[1] += (y0 + y1)*a;
    }
    area *= 0.5;
    centroid[0] /= (6*area);
    centroid[1] /= (6*area);
}

//https://en.wikipedia.org/wiki/Second_moment_of_area#Any_polygon
static void get_inertia(
    const std::vector<double> &verts,
    double inertia[4])
{
    unsigned int i;
    double x0, x1, y0, y1, a;
    double Ix, Ixy, Iyx, Iy;
    Ix = 0;
    Ixy = 0;
    Iyx = 0;
    Iy = 0;
    for (i=0; i<verts.size()/2; i++)
    {
        x0 = verts[i*2];
        x1 = verts[(i*2+2) % verts.size()];
        y0 = verts[i*2+1];
        y1 = verts[(i*2+3) % verts.size()];
        a = x0*y1 - x1*y0;
        Ix += (y0*y0+y0*y1+y1*y1)*a;
        Iy += (x0*x0+x0*x1+x1*x1)*a;
        Ixy += (x0*y1+2*x0*y0+2*x1*y1+x1*y0)*a;
        Iyx += (y0*x1+2*y0*x0+2*y1*x1+y1*x0)*a;
    }
    inertia[0] = fabs(Ix/12.);
    inertia[1] = fabs(Ixy/24.);
    inertia[2] = fabs(Iyx/24.);
    inertia[3] = fabs(Iy/12.);
}

//http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/
static void eig2x2(
    const double mat[4],
    double eigval[2],
    double eigvec[4])
{
    double a, b, c, d;
    double L1;
    double L2;
    double T;
    double D;
    double lterm;
    double scale1, scale2;

    a = mat[0];
    b = mat[1];
    c = mat[2];
    d = mat[3];

    T = a+d;
    D = a*d-b*c;

    lterm = (T*T/4-D);
    if (lterm <= 0)
    {
        eigvec[0] = 1;
        eigvec[1] = 0;
        eigvec[2] = 0;
        eigvec[3] = 1;
        eigval[0] = T/2;
        eigval[1] = T/2;
        return;
    }
    lterm = sqrt(lterm);
    L1 = T/2 + lterm;
    L2 = T/2 - lterm;

    if (fabs(c) > EPSILON)
    {
        scale1 = 1/sqrt((L1-d)*(L1-d)+c*c);
        scale2 = 1/sqrt((L2-d)*(L2-d)+c*c);
        eigvec[0] = scale1*(L1-d);
        eigvec[1] = scale2*(L2-d);
        eigvec[2] = scale1*c;
        eigvec[3] = scale2*c;
        eigval[0] = L1;
        eigval[1] = L2;
    }
    else if(fabs(b) > EPSILON)
    {
        scale1 = 1/sqrt(b*b+(L1-a)*(L1-a));
        scale2 = 1/sqrt(b*b+(L1-a)*(L1-a));
        eigvec[0] = scale1*b;
        eigvec[1] = scale2*b;
        eigvec[2] = scale1*(L1-a);
        eigvec[3] = scale2*(L2-a);
        eigval[0] = L1;
        eigval[1] = L2;
    }
    else
    {
        eigvec[0] = 1;
        eigvec[1] = 0;
        eigvec[2] = 0;
        eigvec[3] = 1;
        eigval[0] = L1;
        eigval[1] = L2;
    }
}

static void center2d(
    std::vector<double> &verts
    )
{
    double centroid[2];
    unsigned int i;
    get_centroid2d(verts, centroid);
    for (i=0; i<verts.size()/2; i++)
    {
        verts[i*2  ] -= centroid[0];
        verts[i*2+1] -= centroid[1];
    }
}

static bool AlmostEqualRelative(double A, double B, double epsilon)
{
    // Calculate the difference.
    double diff = fabs(A - B);
    A = fabs(A);
    B = fabs(B);
    // Find the largest
    double largest = (B > A) ? B : A;

    if (diff <= largest * epsilon)
        return true;
    return false;
}

static void axis_align_mirror2d(
    std::vector<double> &verts
    )
{
    std::vector<double> rotated_verts;
    std::vector<double> mirrored_verts;
    double orig_area;
    double new_area;
    double area_ratio = 0;
    double best_area_ratio = 0;
    double x0, x1, y0, y1, midx, midy, scale;
    double vec[2];
    //double segment[2];
    const double xaxis[2] = {1, 0};
    double R[4] = {1, 0, 0, 1};
    double bestR[4] = {1, 0, 0, 1};
    bool found_axis = false;
    unsigned int i, j;
    orig_area = fabs(get_signed_area2d(verts));
    // Loop over all line segments
    for (i=0; (i<verts.size()/2) and (not found_axis); i++)
    {
        x0 = verts[i*2];
        x1 = verts[(i*2+2) % verts.size()];
        y0 = verts[i*2+1];
        y1 = verts[(i*2+3) % verts.size()];
        midx = (x0+x1)/2;
        midy = (y0+y1)/2;
        scale = 1./sqrt(midx*midx+midy*midy);
        // Vec is the unit vector from origin to midpoint of segment
        vec[0] = midx*scale;
        vec[1] = midy*scale;
        //segment[0] = x1-x0;
        //segment[1] = y1-y0;
        // Get rotation matrix that aligns the axis with the
        // xaxis
        get_rotation2d(vec, xaxis, R);
        // Apply the rotation and save to rotated_verts
        rotated_verts.clear();
        rotated_verts.reserve(verts.size());
        for(j=0; j<verts.size()/2; j++)
        {
            mat2x2mulvec(R, &verts[j*2], vec);
            rotated_verts.push_back(vec[0]);
            rotated_verts.push_back(vec[1]);
        }
        // Mirror the vertices in the x axis
        apply_transforms(rotated_verts, mirrored_verts, dmat1, LEN(dmat1), true);
        new_area = get_convex_area2d(&mirrored_verts[0], mirrored_verts.size()/2);
        area_ratio = orig_area/new_area;
        if (area_ratio > best_area_ratio)
        {
            bestR[0] = R[0];
            bestR[1] = R[1];
            bestR[2] = R[2];
            bestR[3] = R[3];
            best_area_ratio = area_ratio;
        }
        if (best_area_ratio > 0.9999)
        {
            // Symmetry axis found
            found_axis = true;
        }
    }

    // Loop over all vertices
    for (i=0; (i<verts.size()/2) and (not found_axis); i++)
    {
        x0 = verts[i*2];
        y0 = verts[i*2+1];
        scale = 1./sqrt(x0*x0+y0*y0);
        // vec is the unit vector from origin to vertex
        vec[0] = x0*scale;
        vec[1] = y0*scale;
        // Get rotation matrix that aligns the axis with the
        // xaxis
        get_rotation2d(vec, xaxis, R);
        // Apply the rotation and save to rotated_verts
        rotated_verts.clear();
        rotated_verts.reserve(verts.size());
        for(j=0; j<verts.size()/2; j++)
        {
            mat2x2mulvec(R, &verts[j*2], vec);
            rotated_verts.push_back(vec[0]);
            rotated_verts.push_back(vec[1]);
        }
        // Mirror the vertices in the x axis
        apply_transforms(rotated_verts, mirrored_verts, dmat1, LEN(dmat1), true);
        new_area = get_convex_area2d(&mirrored_verts[0], mirrored_verts.size()/2);
        if (area_ratio > best_area_ratio)
        {
            bestR[0] = R[0];
            bestR[1] = R[1];
            bestR[2] = R[2];
            bestR[3] = R[3];
            best_area_ratio = area_ratio;
        }
        if (best_area_ratio > 0.9999)
        {
            // Symmetry axis found
            found_axis = true;
        }
    }

    if (not found_axis)
    {
        //printf("Failed to find mirror axis! Best we can do has area ratio %g\n", best_area_ratio);
    }

    for(i=0; i<verts.size()/2; i++)
    {
        mat2x2mulvec(bestR, &verts[i*2], vec);
        verts[i*2  ] = vec[0];
        verts[i*2+1] = vec[1];
    }
}

// Rotate polygon such that its principal
// axis of inertia is aligned with the chosen axis
// If the inertia tensor is degenerate the function will fail and do nothing.
// Returns true on failure, and false on success
static bool axis_align_pai2d(
    std::vector<double> &verts,
    bool align_yaxis)
{
    unsigned int i;
    double eigval[2];
    double eigvec[4];
    double inertia[4];
    double R[4];
    double refvec[2];
    double vec[2];
    const double xaxis[2] = {1, 0};
    const double yaxis[2] = {0, 1};
    get_inertia(verts, inertia);
    eig2x2(inertia, eigval, eigvec);
    if (std::isnan(eigval[0]))
    {
        printf("eigval %g, %g\n", eigval[0], eigval[1]);
        printf("eigvec %g, %g, %g, %g\n", eigvec[0], eigvec[1], eigvec[2], eigvec[3]);
        printf("inertia %g, %g\n%g, %g\n", inertia[0], inertia[1], inertia[2], inertia[3]);
    }

    if (AlmostEqualRelative(eigval[0], eigval[1], 1e-2))
    {
        return true;
    }

    if (eigval[0] > eigval[1])
    {
        refvec[0] = eigvec[0];
        refvec[1] = eigvec[2];
    }
    else
    {
        refvec[0] = eigvec[1];
        refvec[1] = eigvec[3];
    }

    if (align_yaxis)
    {
        get_rotation2d(refvec, yaxis, R);
    }
    else
    {
        get_rotation2d(refvec, xaxis, R);
    }

    if (std::isnan(R[0]))
    {
        printf("refvec %f, %f\n", refvec[0], refvec[1]);
        printf("aligny=%d\n", align_yaxis);
        printf("R %f, %f\n%f, %f\n", R[0], R[1], R[2], R[3]);
    }

    for(i=0; i<verts.size()/2; i++)
    {
        mat2x2mulvec(R, &verts[i*2], vec);
        verts[i*2  ] = vec[0];
        verts[i*2+1] = vec[1];
    }

    return false;
}

static void apply_transforms(
    const std::vector<double> &verts,
    std::vector<double> &outpoints,
    const double transmat[][4],
    unsigned int num_transforms,
    bool include_identity
    )
{
    unsigned int tidx;
    unsigned int vidx;
    double newpoint[2];
    outpoints.reserve(verts.size()*num_transforms);
    outpoints.clear();

    if (include_identity)
    {
        // Apply identity operation
        outpoints = verts;
    }
    // Apply other transforms
    for (tidx=0; tidx<num_transforms; tidx++)
    {
        for(vidx=0; vidx<verts.size()/2; vidx++)
        {
            mat2x2mulvec(transmat[tidx], &verts[vidx*2], newpoint);
            outpoints.push_back(newpoint[0]);
            outpoints.push_back(newpoint[1]);
        }
    }
}

struct TabEntry
{
    const double (*mat)[4];
    const char *name;
    unsigned int num_mats;
    bool try_second_axis;
};
static const TabEntry symmetry_table[NUM_ROTATION_SYMMETRIES] =
{
    {cmat2, "C2", LEN(cmat2), false},
    {cmat3, "C3", LEN(cmat3), false},
    {cmat4, "C4", LEN(cmat4), false},
    {cmat6, "C6", LEN(cmat6), false},
    {dmat1, "D1", LEN(dmat1), true},
    {dmat2, "D2", LEN(dmat2), true},
    {dmat3, "D3", LEN(dmat3), true},
    {dmat4, "D4", LEN(dmat4), true},
    {dmat6, "D6", LEN(dmat6), true},
};

bool vecnan(const std::vector<double> &numbers)
{
    unsigned int i;
    for(i=0; i<numbers.size(); i++)
    {
        if (std::isnan(numbers[i]))
        {
            return true;
        }
    }
    return false;
}
static void print2d(const std::vector<double> &numbers, const char *prefix)
{
    unsigned int i;
    printf("%s", prefix);
    for(i=0; i<numbers.size()/2; i++)
    {
        printf("(%f, %f),\n", numbers[i*2], numbers[i*2+1]);
    }
}
static void print3d(const std::vector<double> &numbers, const char *prefix)
{
    unsigned int i;
    printf("%s", prefix);
    for(i=0; i<numbers.size()/3; i++)
    {
        printf("(%f, %f, %f),\n", numbers[i*3], numbers[i*3+1], numbers[i*3+2]);
    }
}

static double get_convex_area2d(const double points[], int num_points)
{
    std::vector<Point_2> inp;
    std::vector<Point_2> hull;
    int i;
    for(i=0; i<num_points; i++)
    {
        inp.push_back({points[i*2], points[i*2+1]});
    }
    CGAL::ch_graham_andrew( inp.begin(), inp.end(), std::back_inserter(hull));
    double area = CGAL::polygon_area_2(hull.begin(), hull.end(), K());
    return fabs(area);
}

void get_symmetry_measures(
    const std::vector<double> &vertices3d,
    const double normal[3],
    double symmetry_measures[NUM_ROTATION_SYMMETRIES])
{
    std::vector<double> vertices2d;
    std::vector<double> vertices2d_yaligned;
    std::vector<double> vertices_transformed;
    std::vector<double> chull;
    double orig_area, new_area;
    unsigned int symid;
    bool degenerate_inertia;

    axis_align_3to2(normal, vertices3d, vertices2d);
    center2d(vertices2d);
    vertices2d_yaligned = (const std::vector<double>) vertices2d;
    if(vecnan(vertices2d))
    {
        print2d(vertices2d, "align_and_center\n");
        print3d(vertices3d, "original\n");
        printf("normal %f, %f, %f\n", normal[0], normal[1], normal[2]);
    }

    degenerate_inertia = axis_align_pai2d(vertices2d, false);
    if(vecnan(vertices2d))
    {
        print2d(vertices2d, "nan after xalign\n");
        print3d(vertices3d, "original\n");
        printf("normal %f, %f, %f\n", normal[0], normal[1], normal[2]);
    }
    if (degenerate_inertia)
    {
        // We are in degenerate case
        axis_align_mirror2d(vertices2d);
    }
    else
    {
        apply_transforms(vertices2d, vertices2d_yaligned, matrot90, 1, false);
    }
    orig_area = fabs(get_signed_area2d(vertices2d));

    for(symid=0; symid<NUM_ROTATION_SYMMETRIES; symid++)
    {
        apply_transforms(vertices2d, vertices_transformed, symmetry_table[symid].mat, symmetry_table[symid].num_mats, true);
        new_area = get_convex_area2d(&vertices_transformed[0], vertices_transformed.size()/2);
        if (std::isnan(new_area))
        {
            printf("Convex hull calculation failed!\n");
            symmetry_measures[symid] = 0;
        }
        else
        {
            symmetry_measures[symid] = orig_area/new_area;
        }

        if (symmetry_table[symid].try_second_axis and (not degenerate_inertia))
        {
            apply_transforms(vertices2d_yaligned, vertices_transformed, symmetry_table[symid].mat, symmetry_table[symid].num_mats, true);
            new_area = get_convex_area2d(&vertices_transformed[0], vertices_transformed.size()/2);
            if (std::isnan(new_area))
            {
                printf("Convex hull calculation failed!\n");
            }
            else
            {
                symmetry_measures[symid] = std::max(orig_area/new_area, symmetry_measures[symid]);
            }
        }
    }
}


int main(int argc, char **argv)
{
    unsigned int i, j;
    std::vector<double> vertices3d;
    double symmetry_measures[NUM_ROTATION_SYMMETRIES];
    //const double normal[3] = {0,  0, 1};
    //const double test_verts[] = {
    //-2, -2, 1,
    //-1.0, 2.0, 1,
    //4.0, 2, 1,
    //3, -2, 1,
    //};
#if 1
    //const double test_verts[] =
    //{
        //0.28054782, 0.16196131, 0.178509,
        //1.727357  , 2.667963  , 0.178509,
        //3.17417372, 0.16195695, 0.178509,
    //};
    //const double test_verts[] =
    //{
        //0.44388013,  0.03554383, -2.165855,
        //-0.03554583,  0.44388009, -2.165855,
        //-0.44388154, -0.03554578, -2.165855,
         //0.03554378, -0.44388158, -2.165855,
    //};
    const double test_verts[] =
    {
    -0.81792205,  1.52876621,  1.865395,
    -0.56659765,  1.65128095,  1.865395,
     1.52876745,  0.81791629,  1.865395,
     1.65127999,  0.56659524,  1.865395,
     0.81791704, -1.52876555,  1.865395,
     0.56658968, -1.65128129,  1.865395,
    -1.52876431, -0.8179228 ,  1.865395,
    -1.65128225, -0.56659209,  1.865395,
    };

    //const double test_verts[] =
    //{
        //-1.2, 0, 0.178509,
        //-0.2  , -1  , 0.178509,
        //0.8, 0, 0.178509,
    //};
    //const double test_verts[] =
    //{
    //-1, 0, 0.178509,
    //0  , 1  , 0.178509,
    //1, 0, 0.178509,
    //};
    //const double test_verts[] =
    //{
    //3.17417372+0.16195695,-3.17417372+0.16195695, 0.178509,
    //1.727357  +2.667963  ,-1.727357  +2.667963  , 0.178509,
    //0.28054782+0.16196131,-0.28054782+0.16196131, 0.178509,
    //};
    //const double normal[3] = {-7.67355677e-17,  1.32907463e-16, -1.00000000e+00};
    //const double normal[3] = {1.29213073e-15,  2.60464402e-15, -1.00000000e+00};
    const double normal[3] = {-5.12763349e-11, -5.12758926e-11,  1.00000000e+00};

#else

    const double test_verts[] =
    {
    -1, -1, 0.178509,
    0  , 1  , 0.178509,
    1, -1, 0.178509,
    };
    const double normal[3] = {-7.67355677e-17,  1.32907463e-16, -1.00000000e+00};
#endif

    for (i=0; i<LEN(test_verts); i++)
    {
        vertices3d.push_back(test_verts[i]);
    }
    for(j=0; j<1; j++)
    {

        get_symmetry_measures(vertices3d, normal, symmetry_measures);
        for (i=0; i<NUM_ROTATION_SYMMETRIES; i++)
        {
            printf("%u, %s, %f\n", i, symmetry_table[i].name, symmetry_measures[i]);
        }
    };

}
