# vorosym
Python library (numpy extension) for Voronoi diagrams with periodic boundary conditions and symmetry measures

The library implements the construction of symmetry labeled graphs from atomic positions as described in our paper:
*Materials property prediction using symmetry-labeled graphs as atomic-position independent descriptors* [arXiv:1905.06048](https://arxiv.org/abs/1905.06048)

# install
Make sure the following dependencies are installed on the system
 - [Voro++](http://math.lbl.gov/voro++/) - Computes the Voronoi diagram
 - [CGAL](https://www.cgal.org/) - Used for convex hull computations
 - [gmp](http://gmplib.org/) - Required by CGAL

run
`python setup.py install`
or
`python setup.py install --user`

Take a look in the examples directory for an example on how to use the library
