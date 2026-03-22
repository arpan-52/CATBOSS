#include "outlier_detection.h"
#include <algorithm>
#include <stdexcept>

namespace {

/**
 * Compute median of a vector (modifies input by sorting)
 */
double compute_median(std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    
    if (n % 2 == 0) {
        return (values[n/2 - 1] + values[n/2]) / 2.0;
    } else {
        return values[n/2];
    }
}

} // anonymous namespace


OutlierResult flag_outliers_mad(
    const std::vector<double>& amplitudes,
    const std::vector<double>& predicted,
    double sigma_threshold
) {
    const size_t n = amplitudes.size();
    
    if (n != predicted.size()) {
        throw std::runtime_error("Amplitude and predicted arrays must have same size");
    }
    
    OutlierResult result;
    result.residuals.resize(n);
    result.outliers.resize(n, false);
    
    if (n == 0) {
        result.mad = 0.0;
        result.mad_sigma = 0.0;
        result.median_resid = 0.0;
        return result;
    }
    
    // Calculate residuals: actual - predicted
    for (size_t i = 0; i < n; ++i) {
        result.residuals[i] = amplitudes[i] - predicted[i];
    }
    
    // Compute median of residuals
    std::vector<double> resid_copy = result.residuals;
    result.median_resid = compute_median(resid_copy);
    
    // Compute MAD: median(|residual - median_residual|)
    std::vector<double> abs_dev(n);
    for (size_t i = 0; i < n; ++i) {
        abs_dev[i] = std::abs(result.residuals[i] - result.median_resid);
    }
    result.mad = compute_median(abs_dev);
    
    // Convert MAD to Gaussian-equivalent sigma
    // For Gaussian distribution: sigma = 1.4826 * MAD
    result.mad_sigma = 1.4826 * result.mad;
    
    // Avoid division by zero
    if (result.mad_sigma < 1e-10) {
        // All residuals are essentially identical - no outliers
        return result;
    }
    
    // Flag outliers: |residual - median| > sigma_threshold * mad_sigma
    for (size_t i = 0; i < n; ++i) {
        const double deviation = std::abs(result.residuals[i] - result.median_resid);
        result.outliers[i] = (deviation > sigma_threshold * result.mad_sigma);
    }
    
    return result;
}


py::dict flag_outliers_py(
    py::array_t<double> amplitudes,
    py::array_t<double> predicted,
    double sigma_threshold
) {
    auto amp_buf = amplitudes.request();
    auto pred_buf = predicted.request();
    
    const size_t n = amp_buf.shape[0];
    
    if (pred_buf.shape[0] != static_cast<py::ssize_t>(n)) {
        throw std::runtime_error("Arrays must have same size");
    }
    
    // Convert to vectors
    const double* amp_ptr = static_cast<double*>(amp_buf.ptr);
    const double* pred_ptr = static_cast<double*>(pred_buf.ptr);
    
    std::vector<double> amp_vec(amp_ptr, amp_ptr + n);
    std::vector<double> pred_vec(pred_ptr, pred_ptr + n);
    
    // Run outlier detection
    OutlierResult result = flag_outliers_mad(amp_vec, pred_vec, sigma_threshold);
    
    // Convert to numpy arrays
    auto outliers_arr = py::array_t<bool>(n);
    auto residuals_arr = py::array_t<double>(n);
    
    auto outliers_buf = outliers_arr.request();
    auto residuals_buf = residuals_arr.request();
    
    bool* out_ptr = static_cast<bool*>(outliers_buf.ptr);
    double* res_ptr = static_cast<double*>(residuals_buf.ptr);
    
    for (size_t i = 0; i < n; ++i) {
        out_ptr[i] = result.outliers[i];
        res_ptr[i] = result.residuals[i];
    }
    
    py::dict dict;
    dict["outliers"] = outliers_arr;
    dict["residuals"] = residuals_arr;
    dict["mad"] = result.mad;
    dict["mad_sigma"] = result.mad_sigma;
    dict["median_resid"] = result.median_resid;
    
    return dict;
}
