#include "uv_calc.h"
#include <stdexcept>

py::array_t<double> calculate_uv_distances(
    py::array_t<double> uvw,
    py::array_t<double> wavelengths
) {
    auto uvw_buf = uvw.request();
    auto wave_buf = wavelengths.request();

    // Validate inputs
    if (uvw_buf.ndim != 2 || uvw_buf.shape[1] != 3) {
        throw std::runtime_error("UVW must be shape (n_rows, 3)");
    }
    if (wave_buf.ndim != 1) {
        throw std::runtime_error("Wavelengths must be 1D array");
    }

    const size_t n_rows = uvw_buf.shape[0];
    const size_t n_channels = wave_buf.shape[0];

    if (n_rows == 0 || n_channels == 0) {
        throw std::runtime_error("Empty input arrays");
    }

    // Access data pointers
    const double* uvw_ptr = static_cast<double*>(uvw_buf.ptr);
    const double* wave_ptr = static_cast<double*>(wave_buf.ptr);

    // Create output array
    auto result = py::array_t<double>({n_rows, n_channels});
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);

    // Calculate UV distances
    // UV_dist[row, chan] = sqrt(u^2 + v^2) / wavelength[chan]
    NAMI_PARALLEL_FOR
    for (size_t row = 0; row < n_rows; ++row) {
        const double u = uvw_ptr[row * 3 + 0];
        const double v = uvw_ptr[row * 3 + 1];
        const double uv_magnitude = std::sqrt(u * u + v * v);

        for (size_t chan = 0; chan < n_channels; ++chan) {
            const double wavelength = wave_ptr[chan];
            if (wavelength > 0.0) {
                result_ptr[row * n_channels + chan] = uv_magnitude / wavelength;
            } else {
                result_ptr[row * n_channels + chan] = 0.0;
            }
        }
    }

    return result;
}
