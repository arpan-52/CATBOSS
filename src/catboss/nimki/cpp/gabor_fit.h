#ifndef GABOR_FIT_H
#define GABOR_FIT_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>

namespace py = pybind11;

/**
 * Gabor Basis Function Fitter for UV-domain visibility fitting
 * 
 * Model: V(r) = Σ Aᵢ · exp(-(r/σᵢ)²/2) · cos(ωᵢ·r + φᵢ)
 * 
 * Each component has 4 parameters:
 *   - A (amplitude): strength of component
 *   - σ (sigma): Gaussian envelope width (controls decay rate)
 *   - ω (omega): oscillation frequency
 *   - φ (phi): phase offset
 * 
 * Fitting uses Levenberg-Marquardt algorithm with multiple restarts
 * to avoid local minima.
 * 
 * Author: Arpan Pal
 * Institution: NCRA-TIFR
 */

/**
 * Single Gabor component parameters
 */
struct GaborComponent {
    double amplitude;  // A
    double sigma;      // σ (width/decay)
    double omega;      // ω (frequency)
    double phi;        // φ (phase)
    
    GaborComponent() : amplitude(0.0), sigma(1.0), omega(0.0), phi(0.0) {}
    GaborComponent(double a, double s, double w, double p) 
        : amplitude(a), sigma(s), omega(w), phi(p) {}
};

/**
 * Result from Gabor fitting
 */
struct GaborFitResult {
    std::vector<GaborComponent> components;
    std::vector<double> predicted;
    std::vector<double> residuals;
    double rms_residual;
    double mad_sigma;
    int n_iterations;
    bool converged;
};

/**
 * Gabor Fitter class with Levenberg-Marquardt optimization
 */
class GaborFitter {
public:
    GaborFitter();
    
    /**
     * Fit Gabor model to data
     * 
     * @param x: UV distances (input)
     * @param y: Amplitudes (input)
     * @param n_components: Number of Gabor components to fit
     * @param max_iter: Maximum LM iterations
     * @param tol: Convergence tolerance
     * @param n_restarts: Number of random restarts to avoid local minima
     * @return GaborFitResult with fitted parameters and predictions
     */
    GaborFitResult fit(
        const std::vector<double>& x,
        const std::vector<double>& y,
        int n_components,
        int max_iter = 500,
        double tol = 1e-8,
        int n_restarts = 5
    );
    
    /**
     * Evaluate Gabor model at given points
     */
    std::vector<double> evaluate(
        const std::vector<double>& x,
        const std::vector<GaborComponent>& components
    ) const;
    
    /**
     * Evaluate single Gabor component
     */
    double evaluate_component(double x, const GaborComponent& comp) const;

private:
    std::mt19937 rng_;
    
    /**
     * Initialize components with smart starting values
     */
    std::vector<GaborComponent> initialize_components(
        const std::vector<double>& x,
        const std::vector<double>& y,
        int n_components,
        bool randomize = false
    );
    
    /**
     * Single Levenberg-Marquardt optimization run
     */
    GaborFitResult fit_lm(
        const std::vector<double>& x,
        const std::vector<double>& y,
        std::vector<GaborComponent>& components,
        int max_iter,
        double tol
    );
    
    /**
     * Compute Jacobian matrix for LM
     */
    void compute_jacobian(
        const std::vector<double>& x,
        const std::vector<GaborComponent>& components,
        std::vector<std::vector<double>>& J
    ) const;
    
    /**
     * Solve linear system (J^T J + λI) δ = J^T r using Cholesky
     */
    bool solve_normal_equations(
        const std::vector<std::vector<double>>& JtJ,
        const std::vector<double>& Jtr,
        double lambda,
        std::vector<double>& delta
    ) const;
    
    /**
     * Flatten components to parameter vector
     */
    std::vector<double> flatten_params(const std::vector<GaborComponent>& components) const;
    
    /**
     * Unflatten parameter vector to components
     */
    void unflatten_params(const std::vector<double>& params, 
                          std::vector<GaborComponent>& components) const;
    
    /**
     * Compute RMS residual
     */
    double compute_rms(const std::vector<double>& residuals) const;
    
    /**
     * Compute MAD sigma
     */
    double compute_mad_sigma(const std::vector<double>& residuals) const;
};

/**
 * Python interface for Gabor fitting
 * Returns dict with: predicted, components, rms, mad_sigma, converged, n_iter
 */
py::dict fit_gabor_py(
    py::array_t<double> x,
    py::array_t<double> y,
    int n_components,
    int max_iter = 500,
    double tol = 1e-8,
    int n_restarts = 5
);

/**
 * Adaptive Gabor fitting with roam_around
 * Starts with n_components and adds more until improvement < min_improvement
 * 
 * @param x: UV distances
 * @param y: Amplitudes
 * @param n_components: Starting number of components
 * @param max_components: Maximum components to try
 * @param min_improvement: Stop if improvement < this fraction (e.g., 0.05 = 5%)
 * @param max_iter: Max LM iterations per fit
 * @param tol: Convergence tolerance
 * @return dict with best fit results
 */
py::dict fit_gabor_adaptive_py(
    py::array_t<double> x,
    py::array_t<double> y,
    int n_components,
    int max_components,
    double min_improvement,
    int max_iter = 500,
    double tol = 1e-8
);

#endif // GABOR_FIT_H
