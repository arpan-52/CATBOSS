#ifndef OUTLIER_DETECTION_H
#define OUTLIER_DETECTION_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>

namespace py = pybind11;

/**
 * Result of outlier detection
 */
struct OutlierResult {
    std::vector<bool> outliers;
    std::vector<double> residuals;
    double mad;           // Median Absolute Deviation
    double mad_sigma;     // MAD * 1.4826 (Gaussian-equivalent sigma)
    double median_resid;  // Median of residuals
};

/**
 * Flag outliers using MAD (Median Absolute Deviation)
 * No weights - simple and robust
 *
 * @param amplitudes: Actual amplitudes
 * @param predicted: Predicted amplitudes from model
 * @param sigma_threshold: Flagging threshold in sigmas
 * @return OutlierResult with outlier flags and statistics
 */
OutlierResult flag_outliers_mad(
    const std::vector<double>& amplitudes,
    const std::vector<double>& predicted,
    double sigma_threshold
);

/**
 * Python interface for flag_outliers
 */
py::dict flag_outliers_py(
    py::array_t<double> amplitudes,
    py::array_t<double> predicted,
    double sigma_threshold
);

#endif // OUTLIER_DETECTION_H
