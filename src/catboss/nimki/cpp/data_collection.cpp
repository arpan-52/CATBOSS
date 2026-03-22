#include "data_collection.h"
#include <stdexcept>

CollectedData collect_data_single_corr(
    py::array_t<std::complex<float>> data,
    py::array_t<bool> flags,
    py::array_t<double> uv_distances,
    const std::vector<int>& spw_row_indices,
    int corr_index
) {
    // Get array buffers
    auto data_buf = data.request();
    auto flags_buf = flags.request();
    auto uv_buf = uv_distances.request();

    // Validate shapes
    if (data_buf.ndim != 3 || flags_buf.ndim != 3) {
        throw std::runtime_error("Data and flags must be 3D arrays");
    }
    if (uv_buf.ndim != 2) {
        throw std::runtime_error("UV distances must be 2D array");
    }

    const size_t n_rows_data = data_buf.shape[0];
    const size_t n_channels = data_buf.shape[1];
    const size_t n_corr = data_buf.shape[2];

    // Validate correlation index
    if (corr_index < 0 || static_cast<size_t>(corr_index) >= n_corr) {
        throw std::runtime_error("Invalid correlation index");
    }

    // Access data pointers
    auto data_ptr = static_cast<std::complex<float>*>(data_buf.ptr);
    auto flags_ptr = static_cast<bool*>(flags_buf.ptr);
    auto uv_ptr = static_cast<double*>(uv_buf.ptr);

    // Result storage
    CollectedData result;
    
    // Reserve approximate space (reduces reallocations)
    size_t est_size = spw_row_indices.size() * n_channels / 2;
    result.uv_dists.reserve(est_size);
    result.amplitudes.reserve(est_size);
    result.row_indices.reserve(est_size);
    result.chan_indices.reserve(est_size);

    // Process each row in this SPW
    for (int row_idx : spw_row_indices) {
        // Bounds check
        if (row_idx < 0 || static_cast<size_t>(row_idx) >= n_rows_data) {
            continue;
        }

        for (size_t chan_idx = 0; chan_idx < n_channels; ++chan_idx) {
            // Calculate indices
            const size_t idx = row_idx * n_channels * n_corr + chan_idx * n_corr + corr_index;
            
            // Skip if flagged
            if (flags_ptr[idx]) {
                continue;
            }

            // Get UV distance
            const size_t uv_idx = row_idx * n_channels + chan_idx;
            const double uv_dist = uv_ptr[uv_idx];

            // Skip invalid UV points
            if (uv_dist <= 0.0 || !std::isfinite(uv_dist)) {
                continue;
            }

            // Get visibility and compute amplitude
            const std::complex<float> vis = data_ptr[idx];
            const double amplitude = std::abs(vis);

            // Skip zero or invalid amplitudes
            if (amplitude <= 0.0 || !std::isfinite(amplitude)) {
                continue;
            }

            // Store result
            result.uv_dists.push_back(uv_dist);
            result.amplitudes.push_back(amplitude);
            result.row_indices.push_back(row_idx);
            result.chan_indices.push_back(static_cast<int>(chan_idx));
        }
    }

    return result;
}
