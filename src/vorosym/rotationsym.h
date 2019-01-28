#ifndef ROTATIONSYM_H
#define ROTATIONSYM_H

#define NUM_ROTATION_SYMMETRIES (9)

void get_symmetry_measures(
    const std::vector<double> &vertices3d,
    const double normal[3],
    double symmetry_measures[NUM_ROTATION_SYMMETRIES]);

#endif
