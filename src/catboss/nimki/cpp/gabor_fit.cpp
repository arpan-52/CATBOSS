#include "gabor_fit.h"
#include <numeric>
#include <stdexcept>
#include <iostream>

// Conditional OpenMP
#ifdef _OPENMP
#include <omp.h>
#define NAMI_PARALLEL_FOR _Pragma("omp parallel for")
#else
#define NAMI_PARALLEL_FOR
#endif

namespace {

/**
 * Clamp value to range
 */
inline double clamp(double val, double lo, double hi) {
    return std::max(lo, std::min(hi, val));
}

/**
 * Wrap angle to [-π, π]
 */
inline double wrap_angle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

/**
 * Compute median (modifies input by partial sort)
 */
double compute_median(std::vector<double>& values) {
    if (values.empty()) return 0.0;
    
    const size_t n = values.size();
    const size_t mid = n / 2;
    
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    
    if (n % 2 == 0) {
        auto max_left = std::max_element(values.begin(), values.begin() + mid);
        return (*max_left + values[mid]) / 2.0;
    }
    return values[mid];
}

} // anonymous namespace


GaborFitter::GaborFitter() : rng_(std::random_device{}()) {}


double GaborFitter::evaluate_component(double x, const GaborComponent& comp) const {
    // V(x) = A · exp(-(x/σ)²/2) · cos(ωx + φ)
    const double gaussian = std::exp(-0.5 * (x / comp.sigma) * (x / comp.sigma));
    const double oscillation = std::cos(comp.omega * x + comp.phi);
    return comp.amplitude * gaussian * oscillation;
}


std::vector<double> GaborFitter::evaluate(
    const std::vector<double>& x,
    const std::vector<GaborComponent>& components
) const {
    const size_t n = x.size();
    std::vector<double> result(n, 0.0);
    
    NAMI_PARALLEL_FOR
    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (const auto& comp : components) {
            sum += evaluate_component(x[i], comp);
        }
        result[i] = sum;
    }
    
    return result;
}


std::vector<GaborComponent> GaborFitter::initialize_components(
    const std::vector<double>& x,
    const std::vector<double>& y,
    int n_components,
    bool randomize
) {
    std::vector<GaborComponent> components(n_components);
    
    // Find data range
    double x_min = *std::min_element(x.begin(), x.end());
    double x_max = *std::max_element(x.begin(), x.end());
    double y_max = *std::max_element(y.begin(), y.end());
    double y_mean = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    
    double x_range = x_max - x_min;
    if (x_range < 1e-10) x_range = 1.0;
    
    // Estimate typical frequency from data spacing
    double dx_typical = x_range / std::sqrt(static_cast<double>(x.size()));
    double omega_max = M_PI / dx_typical;  // Nyquist-ish
    
    std::uniform_real_distribution<double> amp_dist(-0.3, 0.3);
    std::uniform_real_distribution<double> sigma_dist(0.8, 1.2);
    std::uniform_real_distribution<double> omega_dist(0.8, 1.2);
    std::uniform_real_distribution<double> phi_dist(-M_PI, M_PI);
    
    for (int i = 0; i < n_components; ++i) {
        // Base initialization: spread components across parameter space
        double base_amp = y_mean / n_components;
        double base_sigma = x_range / (i + 1);
        double base_omega = (i + 1) * omega_max / (2.0 * n_components);
        double base_phi = 0.0;
        
        if (randomize) {
            // Add randomization for multiple restarts
            components[i].amplitude = base_amp * (1.0 + amp_dist(rng_));
            components[i].sigma = base_sigma * sigma_dist(rng_);
            components[i].omega = base_omega * omega_dist(rng_);
            components[i].phi = phi_dist(rng_);
        } else {
            // Deterministic smart initialization
            components[i].amplitude = base_amp * (1.0 - 0.1 * i);
            components[i].sigma = base_sigma;
            components[i].omega = base_omega;
            components[i].phi = 0.0;
        }
        
        // Ensure valid ranges
        components[i].sigma = std::max(components[i].sigma, x_range * 0.01);
        components[i].omega = std::max(components[i].omega, 0.0);
    }
    
    return components;
}


std::vector<double> GaborFitter::flatten_params(
    const std::vector<GaborComponent>& components
) const {
    std::vector<double> params;
    params.reserve(components.size() * 4);
    
    for (const auto& comp : components) {
        params.push_back(comp.amplitude);
        params.push_back(comp.sigma);
        params.push_back(comp.omega);
        params.push_back(comp.phi);
    }
    
    return params;
}


void GaborFitter::unflatten_params(
    const std::vector<double>& params,
    std::vector<GaborComponent>& components
) const {
    const size_t n_comp = components.size();
    
    for (size_t i = 0; i < n_comp; ++i) {
        components[i].amplitude = params[4*i + 0];
        components[i].sigma = std::max(params[4*i + 1], 1e-6);  // Prevent zero sigma
        components[i].omega = std::max(params[4*i + 2], 0.0);   // Non-negative frequency
        components[i].phi = wrap_angle(params[4*i + 3]);
    }
}


void GaborFitter::compute_jacobian(
    const std::vector<double>& x,
    const std::vector<GaborComponent>& components,
    std::vector<std::vector<double>>& J
) const {
    const size_t n = x.size();
    const size_t n_comp = components.size();
    const size_t n_params = n_comp * 4;
    
    // Resize Jacobian
    J.resize(n);
    for (auto& row : J) {
        row.resize(n_params);
    }
    
    // Compute partial derivatives
    // V = A · exp(-(x/σ)²/2) · cos(ωx + φ)
    // ∂V/∂A = exp(-(x/σ)²/2) · cos(ωx + φ)
    // ∂V/∂σ = A · exp(-(x/σ)²/2) · cos(ωx + φ) · (x²/σ³)
    // ∂V/∂ω = A · exp(-(x/σ)²/2) · (-sin(ωx + φ)) · x
    // ∂V/∂φ = A · exp(-(x/σ)²/2) · (-sin(ωx + φ))
    
    NAMI_PARALLEL_FOR
    for (size_t i = 0; i < n; ++i) {
        const double xi = x[i];
        
        for (size_t c = 0; c < n_comp; ++c) {
            const auto& comp = components[c];
            
            const double x_over_sigma = xi / comp.sigma;
            const double gaussian = std::exp(-0.5 * x_over_sigma * x_over_sigma);
            const double angle = comp.omega * xi + comp.phi;
            const double cos_term = std::cos(angle);
            const double sin_term = std::sin(angle);
            
            // ∂V/∂A
            J[i][4*c + 0] = gaussian * cos_term;
            
            // ∂V/∂σ
            J[i][4*c + 1] = comp.amplitude * gaussian * cos_term * 
                           (xi * xi) / (comp.sigma * comp.sigma * comp.sigma);
            
            // ∂V/∂ω
            J[i][4*c + 2] = comp.amplitude * gaussian * (-sin_term) * xi;
            
            // ∂V/∂φ
            J[i][4*c + 3] = comp.amplitude * gaussian * (-sin_term);
        }
    }
}


bool GaborFitter::solve_normal_equations(
    const std::vector<std::vector<double>>& JtJ,
    const std::vector<double>& Jtr,
    double lambda,
    std::vector<double>& delta
) const {
    const size_t n = Jtr.size();
    delta.resize(n);
    
    // Copy and add damping: (J^T J + λI)
    std::vector<std::vector<double>> A = JtJ;
    for (size_t i = 0; i < n; ++i) {
        A[i][i] += lambda;
    }
    
    // Solve using Gaussian elimination with partial pivoting
    std::vector<double> b = Jtr;
    
    // Forward elimination
    for (size_t i = 0; i < n; ++i) {
        // Find pivot
        size_t max_row = i;
        double max_val = std::abs(A[i][i]);
        for (size_t k = i + 1; k < n; ++k) {
            if (std::abs(A[k][i]) > max_val) {
                max_val = std::abs(A[k][i]);
                max_row = k;
            }
        }
        
        // Check for singularity
        if (max_val < 1e-14) {
            return false;
        }
        
        // Swap rows
        if (max_row != i) {
            std::swap(A[i], A[max_row]);
            std::swap(b[i], b[max_row]);
        }
        
        // Eliminate
        for (size_t k = i + 1; k < n; ++k) {
            const double factor = A[k][i] / A[i][i];
            for (size_t j = i; j < n; ++j) {
                A[k][j] -= factor * A[i][j];
            }
            b[k] -= factor * b[i];
        }
    }
    
    // Back substitution
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        delta[i] = b[i];
        for (size_t j = i + 1; j < n; ++j) {
            delta[i] -= A[i][j] * delta[j];
        }
        delta[i] /= A[i][i];
        
        // Check for NaN
        if (!std::isfinite(delta[i])) {
            return false;
        }
    }
    
    return true;
}


double GaborFitter::compute_rms(const std::vector<double>& residuals) const {
    if (residuals.empty()) return 0.0;
    
    double sum_sq = 0.0;
    for (double r : residuals) {
        sum_sq += r * r;
    }
    return std::sqrt(sum_sq / residuals.size());
}


double GaborFitter::compute_mad_sigma(const std::vector<double>& residuals) const {
    if (residuals.empty()) return 0.0;
    
    // Compute median of residuals
    std::vector<double> r_copy = residuals;
    double median = compute_median(r_copy);
    
    // Compute MAD
    std::vector<double> abs_dev(residuals.size());
    for (size_t i = 0; i < residuals.size(); ++i) {
        abs_dev[i] = std::abs(residuals[i] - median);
    }
    double mad = compute_median(abs_dev);
    
    // Convert to Gaussian sigma
    return 1.4826 * mad;
}


GaborFitResult GaborFitter::fit_lm(
    const std::vector<double>& x,
    const std::vector<double>& y,
    std::vector<GaborComponent>& components,
    int max_iter,
    double tol
) {
    const size_t n = x.size();
    const size_t n_params = components.size() * 4;
    
    GaborFitResult result;
    result.converged = false;
    result.n_iterations = 0;
    
    // Initial prediction and cost
    std::vector<double> predicted = evaluate(x, components);
    std::vector<double> residuals(n);
    for (size_t i = 0; i < n; ++i) {
        residuals[i] = y[i] - predicted[i];
    }
    
    double cost = 0.0;
    for (double r : residuals) {
        cost += r * r;
    }
    
    // LM parameters
    double lambda = 1e-3;
    const double lambda_up = 10.0;
    const double lambda_down = 0.1;
    const double lambda_min = 1e-10;
    const double lambda_max = 1e10;
    
    std::vector<std::vector<double>> J;
    std::vector<std::vector<double>> JtJ(n_params, std::vector<double>(n_params));
    std::vector<double> Jtr(n_params);
    std::vector<double> delta(n_params);
    
    for (int iter = 0; iter < max_iter; ++iter) {
        result.n_iterations = iter + 1;
        
        // Compute Jacobian
        compute_jacobian(x, components, J);
        
        // Compute J^T J and J^T r
        std::fill(Jtr.begin(), Jtr.end(), 0.0);
        for (auto& row : JtJ) {
            std::fill(row.begin(), row.end(), 0.0);
        }
        
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n_params; ++j) {
                Jtr[j] += J[i][j] * residuals[i];
                for (size_t k = 0; k < n_params; ++k) {
                    JtJ[j][k] += J[i][j] * J[i][k];
                }
            }
        }
        
        // Check gradient convergence
        double grad_norm = 0.0;
        for (double g : Jtr) {
            grad_norm += g * g;
        }
        grad_norm = std::sqrt(grad_norm);
        
        if (grad_norm < tol * n) {
            result.converged = true;
            break;
        }
        
        // Solve for update
        bool solved = false;
        int lm_tries = 0;
        const int max_lm_tries = 20;
        
        while (!solved && lm_tries < max_lm_tries) {
            if (!solve_normal_equations(JtJ, Jtr, lambda, delta)) {
                lambda *= lambda_up;
                lm_tries++;
                continue;
            }
            
            // Try update
            std::vector<GaborComponent> new_components = components;
            std::vector<double> params = flatten_params(components);
            
            for (size_t i = 0; i < n_params; ++i) {
                params[i] += delta[i];
            }
            unflatten_params(params, new_components);
            
            // Compute new cost
            std::vector<double> new_predicted = evaluate(x, new_components);
            double new_cost = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double r = y[i] - new_predicted[i];
                new_cost += r * r;
            }
            
            if (new_cost < cost) {
                // Accept update
                components = new_components;
                predicted = new_predicted;
                for (size_t i = 0; i < n; ++i) {
                    residuals[i] = y[i] - predicted[i];
                }
                
                // Check cost convergence
                double rel_improvement = (cost - new_cost) / (cost + 1e-10);
                cost = new_cost;
                
                lambda = std::max(lambda * lambda_down, lambda_min);
                solved = true;
                
                if (rel_improvement < tol) {
                    result.converged = true;
                }
            } else {
                // Reject, increase damping
                lambda = std::min(lambda * lambda_up, lambda_max);
                lm_tries++;
            }
        }
        
        if (!solved || result.converged) {
            break;
        }
    }
    
    // Final results
    result.components = components;
    result.predicted = predicted;
    result.residuals = residuals;
    result.rms_residual = compute_rms(residuals);
    result.mad_sigma = compute_mad_sigma(residuals);
    
    return result;
}


GaborFitResult GaborFitter::fit(
    const std::vector<double>& x,
    const std::vector<double>& y,
    int n_components,
    int max_iter,
    double tol,
    int n_restarts
) {
    if (x.size() != y.size()) {
        throw std::runtime_error("x and y must have same size");
    }
    if (x.size() < static_cast<size_t>(n_components * 4)) {
        throw std::runtime_error("Not enough data points for requested components");
    }
    if (n_components < 1) {
        throw std::runtime_error("Need at least 1 component");
    }
    
    // Check for NaN/Inf
    for (size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) {
            throw std::runtime_error("Data contains NaN or Inf");
        }
    }
    
    GaborFitResult best_result;
    best_result.rms_residual = std::numeric_limits<double>::max();
    best_result.converged = false;
    
    // Try multiple restarts
    for (int restart = 0; restart < n_restarts; ++restart) {
        // Initialize components (first try deterministic, rest randomized)
        std::vector<GaborComponent> components = initialize_components(
            x, y, n_components, restart > 0
        );
        
        // Run LM optimization
        GaborFitResult result = fit_lm(x, y, components, max_iter, tol);
        
        // Keep best
        if (result.rms_residual < best_result.rms_residual) {
            best_result = result;
        }
        
        // Early exit if converged well
        if (result.converged && result.rms_residual < tol * 100) {
            break;
        }
    }
    
    return best_result;
}


// =============================================================================
// Python Interface
// =============================================================================

py::dict fit_gabor_py(
    py::array_t<double> x,
    py::array_t<double> y,
    int n_components,
    int max_iter,
    double tol,
    int n_restarts
) {
    auto x_buf = x.request();
    auto y_buf = y.request();
    
    if (x_buf.ndim != 1 || y_buf.ndim != 1) {
        throw std::runtime_error("x and y must be 1D arrays");
    }
    
    const size_t n = x_buf.shape[0];
    if (y_buf.shape[0] != static_cast<py::ssize_t>(n)) {
        throw std::runtime_error("x and y must have same size");
    }
    
    const double* x_ptr = static_cast<double*>(x_buf.ptr);
    const double* y_ptr = static_cast<double*>(y_buf.ptr);
    
    std::vector<double> x_vec(x_ptr, x_ptr + n);
    std::vector<double> y_vec(y_ptr, y_ptr + n);
    
    // Fit
    GaborFitter fitter;
    GaborFitResult result = fitter.fit(x_vec, y_vec, n_components, max_iter, tol, n_restarts);
    
    // Convert to numpy arrays
    auto predicted_arr = py::array_t<double>(n);
    auto residuals_arr = py::array_t<double>(n);
    
    double* pred_ptr = static_cast<double*>(predicted_arr.request().ptr);
    double* res_ptr = static_cast<double*>(residuals_arr.request().ptr);
    
    std::copy(result.predicted.begin(), result.predicted.end(), pred_ptr);
    std::copy(result.residuals.begin(), result.residuals.end(), res_ptr);
    
    // Convert components to list of dicts
    py::list comp_list;
    for (const auto& comp : result.components) {
        py::dict comp_dict;
        comp_dict["amplitude"] = comp.amplitude;
        comp_dict["sigma"] = comp.sigma;
        comp_dict["omega"] = comp.omega;
        comp_dict["phi"] = comp.phi;
        comp_list.append(comp_dict);
    }
    
    py::dict out;
    out["predicted"] = predicted_arr;
    out["residuals"] = residuals_arr;
    out["components"] = comp_list;
    out["rms"] = result.rms_residual;
    out["mad_sigma"] = result.mad_sigma;
    out["converged"] = result.converged;
    out["n_iterations"] = result.n_iterations;
    out["n_components"] = static_cast<int>(result.components.size());
    
    return out;
}


py::dict fit_gabor_adaptive_py(
    py::array_t<double> x,
    py::array_t<double> y,
    int n_components,
    int max_components,
    double min_improvement,
    int max_iter,
    double tol
) {
    auto x_buf = x.request();
    auto y_buf = y.request();
    
    const size_t n = x_buf.shape[0];
    const double* x_ptr = static_cast<double*>(x_buf.ptr);
    const double* y_ptr = static_cast<double*>(y_buf.ptr);
    
    std::vector<double> x_vec(x_ptr, x_ptr + n);
    std::vector<double> y_vec(y_ptr, y_ptr + n);
    
    GaborFitter fitter;
    
    // Start with initial components
    GaborFitResult best_result = fitter.fit(x_vec, y_vec, n_components, max_iter, tol, 5);
    double prev_mad_sigma = best_result.mad_sigma;
    int best_n_comp = n_components;
    
    // Roam around: try adding components
    for (int nc = n_components + 1; nc <= max_components; ++nc) {
        // Check if we have enough data
        if (n < static_cast<size_t>(nc * 4 + 10)) {
            break;
        }
        
        GaborFitResult result = fitter.fit(x_vec, y_vec, nc, max_iter, tol, 3);
        
        // Compute improvement
        double improvement = (prev_mad_sigma - result.mad_sigma) / (prev_mad_sigma + 1e-10);
        
        if (improvement > min_improvement && result.mad_sigma < prev_mad_sigma) {
            // Significant improvement, keep going
            best_result = result;
            prev_mad_sigma = result.mad_sigma;
            best_n_comp = nc;
        } else {
            // Diminishing returns, stop
            break;
        }
    }
    
    // Convert best result to Python
    auto predicted_arr = py::array_t<double>(n);
    auto residuals_arr = py::array_t<double>(n);
    
    double* pred_ptr = static_cast<double*>(predicted_arr.request().ptr);
    double* res_ptr = static_cast<double*>(residuals_arr.request().ptr);
    
    std::copy(best_result.predicted.begin(), best_result.predicted.end(), pred_ptr);
    std::copy(best_result.residuals.begin(), best_result.residuals.end(), res_ptr);
    
    py::list comp_list;
    for (const auto& comp : best_result.components) {
        py::dict comp_dict;
        comp_dict["amplitude"] = comp.amplitude;
        comp_dict["sigma"] = comp.sigma;
        comp_dict["omega"] = comp.omega;
        comp_dict["phi"] = comp.phi;
        comp_list.append(comp_dict);
    }
    
    py::dict out;
    out["predicted"] = predicted_arr;
    out["residuals"] = residuals_arr;
    out["components"] = comp_list;
    out["rms"] = best_result.rms_residual;
    out["mad_sigma"] = best_result.mad_sigma;
    out["converged"] = best_result.converged;
    out["n_iterations"] = best_result.n_iterations;
    out["n_components"] = best_n_comp;
    
    return out;
}
