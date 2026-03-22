#ifndef DATA_COLLECTION_H
#define DATA_COLLECTION_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <complex>
#include <vector>
#include <cmath>

namespace py = pybind11;

/**
 * Collected data for a single correlation
 * Amplitude only, no weights
 */
struct CollectedData {
    std::vector<double> uv_dists;
    std::vector<double> amplitudes;
    std::vector<int> row_indices;
    std::vector<int> chan_indices;
};

/**
 * Collect amplitude data for a SINGLE correlation
 * 
 * @param data: Complex visibility data (n_rows, n_channels, n_corr)
 * @param flags: Flag array (n_rows, n_channels, n_corr)
 * @param uv_distances: UV distances (n_rows, n_channels)
 * @param spw_row_indices: Which rows belong to this SPW
 * @param corr_index: Which single correlation to process
 */
CollectedData collect_data_single_corr(
    py::array_t<std::complex<float>> data,
    py::array_t<bool> flags,
    py::array_t<double> uv_distances,
    const std::vector<int>& spw_row_indices,
    int corr_index
);

#endif // DATA_COLLECTION_H
