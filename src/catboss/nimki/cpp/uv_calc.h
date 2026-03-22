#ifndef UV_CALC_H
#define UV_CALC_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>

// Conditional OpenMP support
#ifdef _OPENMP
#include <omp.h>
#define NAMI_PARALLEL_FOR _Pragma("omp parallel for")
#else
#define NAMI_PARALLEL_FOR
#endif

namespace py = pybind11;

/**
 * Calculate UV distances for all rows and channels
 * 
 * UV distance = sqrt(u^2 + v^2) / wavelength
 *
 * @param uvw: UVW coordinates (n_rows, 3)
 * @param wavelengths: Wavelengths in meters (n_channels,)
 * @return UV distances in wavelengths (n_rows, n_channels)
 */
py::array_t<double> calculate_uv_distances(
    py::array_t<double> uvw,
    py::array_t<double> wavelengths
);

#endif // UV_CALC_H
