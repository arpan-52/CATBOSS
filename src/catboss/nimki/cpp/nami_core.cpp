/**
 * NAMI C++ Core Module
 *
 * Fast implementations for RFI flagging using Gabor basis fitting:
 * - UV distance calculation
 * - Data collection (amplitude only, per correlation)
 * - Gabor basis fitting (replaces spline)
 * - Adaptive Gabor fitting with roam_around
 * - Outlier detection using MAD
 * 
 * Author: Arpan Pal
 * Institution: NCRA-TIFR
 * Version: 1.0.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "uv_calc.h"
#include "data_collection.h"
#include "gabor_fit.h"
#include "outlier_detection.h"

namespace py = pybind11;

/**
 * Python wrapper for collect_data_single_corr
 * Returns dict with numpy arrays
 */
py::dict collect_data_py(
    py::array_t<std::complex<float>> data,
    py::array_t<bool> flags,
    py::array_t<double> uv_distances,
    py::array_t<int> spw_row_indices,
    int corr_index
) {
    // Convert numpy array to vector
    auto spw_buf = spw_row_indices.request();
    const int* spw_ptr = static_cast<int*>(spw_buf.ptr);
    std::vector<int> spw_vec(spw_ptr, spw_ptr + spw_buf.shape[0]);

    // Call C++ function
    CollectedData result = collect_data_single_corr(
        data, flags, uv_distances, spw_vec, corr_index
    );

    const size_t n = result.uv_dists.size();

    // Convert to numpy arrays
    auto uv_arr = py::array_t<double>(n);
    auto amp_arr = py::array_t<double>(n);
    auto row_arr = py::array_t<int>(n);
    auto chan_arr = py::array_t<int>(n);

    auto uv_buf = uv_arr.request();
    auto amp_buf = amp_arr.request();
    auto row_buf = row_arr.request();
    auto chan_buf = chan_arr.request();

    double* uv_ptr = static_cast<double*>(uv_buf.ptr);
    double* amp_ptr = static_cast<double*>(amp_buf.ptr);
    int* row_ptr = static_cast<int*>(row_buf.ptr);
    int* chan_ptr = static_cast<int*>(chan_buf.ptr);

    for (size_t i = 0; i < n; ++i) {
        uv_ptr[i] = result.uv_dists[i];
        amp_ptr[i] = result.amplitudes[i];
        row_ptr[i] = result.row_indices[i];
        chan_ptr[i] = result.chan_indices[i];
    }

    py::dict dict;
    dict["uv_dists"] = uv_arr;
    dict["amplitudes"] = amp_arr;
    dict["row_indices"] = row_arr;
    dict["chan_indices"] = chan_arr;

    return dict;
}


PYBIND11_MODULE(_nami_core, m) {
    m.doc() = R"doc(
        NAMI C++ Core - Fast RFI flagging for radio astronomy
        
        Uses Gabor basis functions for UV-domain visibility fitting:
        V(r) = Σ Aᵢ · exp(-(r/σᵢ)²/2) · cos(ωᵢ·r + φᵢ)
        
        This captures both smooth decay AND oscillations from source structure.
        
        Author: Arpan Pal (NCRA-TIFR)
        Version: 1.0.0
    )doc";

    // UV distance calculation
    m.def("calculate_uv_distances", &calculate_uv_distances,
          py::arg("uvw"),
          py::arg("wavelengths"),
          R"doc(
            Calculate UV distances for all rows and channels.

            Parameters
            ----------
            uvw : ndarray, shape (n_rows, 3)
                UVW coordinates in meters
            wavelengths : ndarray, shape (n_channels,)
                Wavelengths in meters

            Returns
            -------
            ndarray, shape (n_rows, n_channels)
                UV distances in wavelengths
          )doc");

    // Data collection
    m.def("collect_data", &collect_data_py,
          py::arg("data"),
          py::arg("flags"),
          py::arg("uv_distances"),
          py::arg("spw_row_indices"),
          py::arg("corr_index"),
          R"doc(
            Collect amplitude data for a single correlation.

            Only collects unflagged data points.

            Parameters
            ----------
            data : ndarray, shape (n_rows, n_channels, n_corr), complex64
                Visibility data
            flags : ndarray, shape (n_rows, n_channels, n_corr), bool
                Flag array (True = flagged)
            uv_distances : ndarray, shape (n_rows, n_channels)
                UV distances
            spw_row_indices : ndarray, int
                Row indices for this spectral window
            corr_index : int
                Which correlation to process

            Returns
            -------
            dict with keys:
                uv_dists : ndarray - UV distances
                amplitudes : ndarray - Visibility amplitudes
                row_indices : ndarray - Row indices
                chan_indices : ndarray - Channel indices
          )doc");

    // Gabor fitting
    m.def("fit_gabor", &fit_gabor_py,
          py::arg("x"),
          py::arg("y"),
          py::arg("n_components") = 5,
          py::arg("max_iter") = 500,
          py::arg("tol") = 1e-8,
          py::arg("n_restarts") = 5,
          R"doc(
            Fit Gabor basis model to visibility amplitudes.
            
            Model: V(r) = Σ Aᵢ · exp(-(r/σᵢ)²/2) · cos(ωᵢ·r + φᵢ)
            
            Each component has 4 parameters:
              - amplitude (A): strength
              - sigma (σ): Gaussian width (decay rate)
              - omega (ω): oscillation frequency
              - phi (φ): phase offset

            Parameters
            ----------
            x : ndarray
                UV distances
            y : ndarray
                Visibility amplitudes
            n_components : int, optional
                Number of Gabor components (default: 5)
            max_iter : int, optional
                Maximum Levenberg-Marquardt iterations (default: 500)
            tol : float, optional
                Convergence tolerance (default: 1e-8)
            n_restarts : int, optional
                Number of random restarts to avoid local minima (default: 5)

            Returns
            -------
            dict with keys:
                predicted : ndarray - Fitted values at input x
                residuals : ndarray - y - predicted
                components : list of dict - Fitted parameters for each component
                rms : float - RMS of residuals
                mad_sigma : float - MAD * 1.4826 (robust sigma estimate)
                converged : bool - Whether optimization converged
                n_iterations : int - Number of iterations used
                n_components : int - Number of components
          )doc");

    // Adaptive Gabor fitting (roam_around)
    m.def("fit_gabor_adaptive", &fit_gabor_adaptive_py,
          py::arg("x"),
          py::arg("y"),
          py::arg("n_components") = 5,
          py::arg("max_components") = 12,
          py::arg("min_improvement") = 0.05,
          py::arg("max_iter") = 500,
          py::arg("tol") = 1e-8,
          R"doc(
            Adaptive Gabor fitting that automatically determines optimal components.
            
            Starts with n_components and adds more until improvement drops below
            min_improvement threshold (diminishing returns).

            Parameters
            ----------
            x : ndarray
                UV distances
            y : ndarray
                Visibility amplitudes
            n_components : int, optional
                Starting number of components (default: 5)
            max_components : int, optional
                Maximum components to try (default: 12)
            min_improvement : float, optional
                Stop if MAD_sigma improvement < this fraction (default: 0.05 = 5%)
            max_iter : int, optional
                Max LM iterations per fit (default: 500)
            tol : float, optional
                Convergence tolerance (default: 1e-8)

            Returns
            -------
            dict with keys:
                predicted : ndarray - Best fitted values
                residuals : ndarray - Best residuals
                components : list - Best component parameters
                rms : float - Best RMS
                mad_sigma : float - Best MAD sigma
                converged : bool - Whether final fit converged
                n_iterations : int - Iterations for final fit
                n_components : int - Optimal number of components found
          )doc");

    // Outlier detection
    m.def("flag_outliers", &flag_outliers_py,
          py::arg("amplitudes"),
          py::arg("predicted"),
          py::arg("sigma_threshold"),
          R"doc(
            Flag outliers using MAD (Median Absolute Deviation).

            Parameters
            ----------
            amplitudes : ndarray
                Actual amplitudes
            predicted : ndarray
                Predicted amplitudes from Gabor model
            sigma_threshold : float
                Flagging threshold in sigma units

            Returns
            -------
            dict with keys:
                outliers : ndarray, bool - True where outlier
                residuals : ndarray - amplitude - predicted
                mad : float - Median Absolute Deviation
                mad_sigma : float - MAD * 1.4826 (Gaussian sigma)
                median_resid : float - Median of residuals
          )doc");

    m.attr("__version__") = "1.0.0";
}
