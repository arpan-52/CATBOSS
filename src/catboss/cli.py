#!/usr/bin/env python3
"""
CATBOSS CLI - Command Line Interface

Unified CLI for POOH and NIMKI RFI flagging.

Author: Arpan Pal
Institution: NCRA-TIFR
"""

import argparse
import sys
from typing import Dict, Any, List, Optional


def parse_multi_float(value: str) -> List[float]:
    """Parse comma-separated floats for multi-pass parameters."""
    return [float(x.strip()) for x in value.split(',') if x.strip()]


def parse_multi_int(value: str) -> List[int]:
    """Parse comma-separated integers."""
    return [int(x.strip()) for x in value.split(',') if x.strip()]


def expand_to_passes(values: List, n_passes: int) -> List:
    """Expand parameter list to match number of passes.
    
    If fewer values than passes, repeat the last value.
    """
    if len(values) >= n_passes:
        return values[:n_passes]
    else:
        return values + [values[-1]] * (n_passes - len(values))


def build_passes_config(args, n_passes: int) -> List[Dict[str, Any]]:
    """Build per-pass configuration from CLI arguments.
    
    Supports comma-separated values for multi-pass:
      --sigma 8.0,5.0,4.0 means pass1=8.0, pass2=5.0, pass3=4.0
    """
    # Parse comma-separated parameters
    sigmas = parse_multi_float(args.sigma) if isinstance(args.sigma, str) else [args.sigma]
    rhos = parse_multi_float(args.rho) if isinstance(args.rho, str) else [args.rho]
    iqr_factors = parse_multi_float(args.iqr_factor) if isinstance(args.iqr_factor, str) else [args.iqr_factor]
    mad_sigmas = parse_multi_float(args.mad_sigma) if isinstance(args.mad_sigma, str) else [args.mad_sigma]
    
    # Infer number of passes from longest parameter list if not explicitly set
    if n_passes == 1:
        max_len = max(len(sigmas), len(rhos), len(iqr_factors), len(mad_sigmas))
        if max_len > 1:
            n_passes = max_len
    
    # Expand all to n_passes
    sigmas = expand_to_passes(sigmas, n_passes)
    rhos = expand_to_passes(rhos, n_passes)
    iqr_factors = expand_to_passes(iqr_factors, n_passes)
    mad_sigmas = expand_to_passes(mad_sigmas, n_passes)
    
    # Combinations is shared across all passes (usually same window sizes)
    combinations = args.combinations
    
    # Build config for each pass
    passes = []
    for i in range(n_passes):
        pass_config = {
            'method': args.method,
            'use_gpu': args.mode != 'cpu',
        }
        
        if args.method == 'sumthreshold':
            pass_config['sigma'] = sigmas[i]
            pass_config['rho'] = rhos[i]
            pass_config['combinations'] = [int(x) for x in combinations.split(',')]
        elif args.method == 'iqr':
            pass_config['iqr_factor'] = iqr_factors[i]
        elif args.method == 'mad':
            pass_config['mad_sigma'] = mad_sigmas[i]
        
        passes.append(pass_config)
    
    return passes


def create_pooh_parser(subparsers) -> argparse.ArgumentParser:
    """Create POOH subparser with all options."""
    
    pooh = subparsers.add_parser(
        'pooh',
        help='POOH: Parallelized Optimized Outlier Hunter (dynamic spectra flagging)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
POOH - Parallelized Optimized Outlier Hunter

Dynamic spectra-based RFI flagging using:
  - SumThreshold: Multi-scale combinatorial thresholding (Offringa et al. 2010)
  - IQR: Inter-quartile range outlier detection  
  - MAD: Median Absolute Deviation flagging

Multi-pass: Use comma-separated values for per-pass parameters.
  e.g., --sigma 8.0,5.0,4.0 runs 3 passes with decreasing threshold.
  Number of passes is inferred from longest parameter list,
  or set explicitly with --passes.
        """
    )
    
    # Required
    pooh.add_argument('ms_path', help='Path to Measurement Set')
    
    # Method selection
    method_group = pooh.add_argument_group('Method Selection')
    method_group.add_argument(
        '--method', type=str, default='sumthreshold',
        choices=['sumthreshold', 'iqr', 'mad'],
        help='Flagging method (default: sumthreshold)'
    )
    
    # SumThreshold parameters
    st_group = pooh.add_argument_group('SumThreshold Parameters')
    st_group.add_argument(
        '--sigma', type=str, default='6.0',
        help='Base threshold multiplier, comma-separated for multi-pass (default: 6.0)'
    )
    st_group.add_argument(
        '--rho', type=str, default='1.5',
        help='Window reduction factor, comma-separated for multi-pass (default: 1.5)'
    )
    st_group.add_argument(
        '--combinations', type=str, default='1,2,4,8,16,32,64',
        help='Window sizes M (default: 1,2,4,8,16,32,64)'
    )
    
    # IQR parameters
    iqr_group = pooh.add_argument_group('IQR Parameters')
    iqr_group.add_argument(
        '--iqr-factor', type=str, default='1.5',
        help='IQR multiplier, comma-separated for multi-pass (default: 1.5)'
    )
    
    # MAD parameters
    mad_group = pooh.add_argument_group('MAD Parameters')
    mad_group.add_argument(
        '--mad-sigma', type=str, default='5.0',
        help='MAD sigma threshold, comma-separated for multi-pass (default: 5.0)'
    )
    
    # Bandpass normalization
    bp_group = pooh.add_argument_group('Bandpass Normalization')
    bp_group.add_argument(
        '--poly-order', type=int, default=5,
        help='Polynomial degree for bandpass fit (default: 5)'
    )
    bp_group.add_argument(
        '--deviation-threshold', type=float, default=5.0,
        help='Bad channel detection sigma (default: 5.0)'
    )
    bp_group.add_argument(
        '--no-bandpass', action='store_true',
        help='Skip bandpass normalization'
    )
    
    # Multi-pass
    pass_group = pooh.add_argument_group('Multi-Pass Processing')
    pass_group.add_argument(
        '--passes', type=int, default=1,
        help='Number of flagging passes (default: 1, or inferred from params)'
    )
    
    # Data selection
    sel_group = pooh.add_argument_group('Data Selection')
    sel_group.add_argument(
        '--datacolumn', type=str, default='DATA',
        choices=['DATA', 'CORRECTED_DATA', 'RESIDUAL_DATA', 'MODEL_DATA'],
        help='Data column to process (default: DATA)'
    )
    sel_group.add_argument(
        '--field', type=str, default=None,
        help='Field IDs, comma-separated (default: all)'
    )
    sel_group.add_argument(
        '--spw', type=str, default=None,
        help='SPW IDs, comma-separated (default: all)'
    )
    sel_group.add_argument(
        '--corr', type=str, default=None,
        help='Correlation indices, comma-separated (default: all)'
    )
    sel_group.add_argument(
        '--baseline', type=str, default=None,
        help='Baselines as "0-1,0-2,1-2" (default: all)'
    )
    sel_group.add_argument(
        '--exclude-autocorr', action='store_true',
        help='Exclude auto-correlations'
    )
    
    # Chunking
    chunk_group = pooh.add_argument_group('Time-Frequency Chunking')
    chunk_group.add_argument(
        '--timebin', type=int, default=None,
        help='Time chunk size in time samples (default: full range)'
    )
    chunk_group.add_argument(
        '--freqbin', type=int, default=None,
        help='Frequency chunk size in channels (default: full range)'
    )
    
    # Flag behavior
    flag_group = pooh.add_argument_group('Flag Behavior')
    flag_group.add_argument(
        '--apply-flags', action='store_true',
        help='Write flags to MS file'
    )
    flag_group.add_argument(
        '--dry-run', action='store_true',
        help='Generate plots only, no flag writing'
    )
    flag_group.add_argument(
        '--propagate-flags', action='store_true',
        help='Propagate flags across all correlations'
    )
    
    # Output
    out_group = pooh.add_argument_group('Output')
    out_group.add_argument(
        '--plots', action='store_true',
        help='Generate diagnostic plots'
    )
    out_group.add_argument(
        '--plot-dir', type=str, default='pooh_plots',
        help='Directory for plots (default: pooh_plots)'
    )
    
    # Performance
    perf_group = pooh.add_argument_group('Performance')
    perf_group.add_argument(
        '--mode', type=str, default='auto',
        choices=['auto', 'cpu', 'gpu'],
        help='Processing mode (default: auto)'
    )
    perf_group.add_argument(
        '--chunk-size', type=int, default=200000,
        help='I/O chunk size in rows (default: 200000)'
    )
    perf_group.add_argument(
        '--max-memory', type=float, default=0.8,
        help='Max memory fraction to use (default: 0.8)'
    )
    
    # General
    gen_group = pooh.add_argument_group('General')
    gen_group.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose output'
    )
    
    return pooh


def create_nimki_parser(subparsers) -> argparse.ArgumentParser:
    """Create NIMKI subparser with all options."""
    
    nimki = subparsers.add_parser(
        'nimki',
        help='NIMKI: Non-linear Interference Modeling and Korrection Interface (UV-domain)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
NIMKI - Non-linear Interference Modeling and Korrection Interface

UV-domain RFI flagging using Gabor basis fitting:
  Model: V(r) = Sum[ A_i * exp(-(r/sigma_i)^2/2) * cos(omega_i*r + phi_i) ]

This captures both smooth Gaussian decay AND oscillations from
source structure (disks, rings, double sources, etc.)
        """
    )
    
    # Required
    nimki.add_argument('ms_path', help='Path to Measurement Set')
    
    # Gabor fitting parameters
    gabor_group = nimki.add_argument_group('Gabor Fitting Parameters')
    gabor_group.add_argument(
        '--n-components', type=int, default=5,
        help='Number of Gabor components (default: 5)'
    )
    gabor_group.add_argument(
        '--roam-around', action='store_true',
        help='Adaptively add components until diminishing returns'
    )
    gabor_group.add_argument(
        '--max-components', type=int, default=12,
        help='Max components when roam-around enabled (default: 12)'
    )
    gabor_group.add_argument(
        '--min-improvement', type=float, default=0.05,
        help='Stop roam-around if improvement < this fraction (default: 0.05)'
    )
    gabor_group.add_argument(
        '--max-iter', type=int, default=500,
        help='Max iterations per fit (default: 500)'
    )
    gabor_group.add_argument(
        '--tolerance', type=float, default=1e-8,
        help='Convergence tolerance (default: 1e-8)'
    )
    
    # Flagging parameters
    flag_param_group = nimki.add_argument_group('Flagging Parameters')
    flag_param_group.add_argument(
        '--sigma', type=float, default=5.0,
        help='Flagging threshold in sigma (default: 5.0)'
    )
    
    # Data selection
    sel_group = nimki.add_argument_group('Data Selection')
    sel_group.add_argument(
        '--datacolumn', type=str, default='DATA',
        choices=['DATA', 'CORRECTED_DATA', 'MODEL_DATA'],
        help='Data column to process (default: DATA)'
    )
    sel_group.add_argument(
        '--field', type=str, default=None,
        help='Field IDs, comma-separated (default: all)'
    )
    sel_group.add_argument(
        '--spw', type=str, default=None,
        help='SPW IDs, comma-separated (default: all)'
    )
    sel_group.add_argument(
        '--corr', type=str, default=None,
        help='Correlation indices, comma-separated (default: all)'
    )
    
    # Time chunking
    chunk_group = nimki.add_argument_group('Time Chunking')
    chunk_group.add_argument(
        '--timebin', type=float, default=30.0,
        help='Time bin size in minutes (default: 30.0)'
    )
    
    # Flag behavior
    flag_group = nimki.add_argument_group('Flag Behavior')
    flag_group.add_argument(
        '--apply-flags', action='store_true',
        help='Write flags to MS file'
    )
    flag_group.add_argument(
        '--dry-run', action='store_true',
        help='Generate plots only, no flag writing'
    )
    flag_group.add_argument(
        '--flag-all-corr', action='store_true', default=True,
        help='Flag all correlations if any is bad (default: True)'
    )
    flag_group.add_argument(
        '--flag-selected-only', action='store_true',
        help='Only flag the bad correlation'
    )
    
    # Output
    out_group = nimki.add_argument_group('Output')
    out_group.add_argument(
        '--plots', action='store_true',
        help='Generate diagnostic plots'
    )
    out_group.add_argument(
        '--plot-dir', type=str, default='nimki_plots',
        help='Directory for plots (default: nimki_plots)'
    )
    
    # Performance
    perf_group = nimki.add_argument_group('Performance')
    perf_group.add_argument(
        '--ncpu', type=int, default=0,
        help='Number of CPUs (0 = all available)'
    )
    
    # General
    gen_group = nimki.add_argument_group('General')
    gen_group.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose output'
    )
    
    return nimki


def run_pooh(args) -> int:
    """Run POOH with parsed arguments."""
    from catboss.logger import setup_logger, print_banner, print_cat_on_hunt
    from catboss.pooh import hunt_ms
    
    # Print banner
    print_banner()
    print_cat_on_hunt("POOH")
    
    # Setup logger
    logger = setup_logger("catboss", verbose=args.verbose)
    
    # Build passes configuration
    passes_config = build_passes_config(args, args.passes)
    n_passes = len(passes_config)
    
    # Build options dictionary
    options = {
        'method': args.method,
        'passes_config': passes_config,
        'passes': n_passes,
        'poly_order': args.poly_order,
        'deviation_threshold': args.deviation_threshold,
        'no_bandpass': args.no_bandpass,
        'datacolumn': args.datacolumn,
        'field': args.field,
        'spw': args.spw,
        'corr': args.corr,
        'baseline': args.baseline,
        'exclude_autocorr': args.exclude_autocorr,
        'timebin': args.timebin,
        'freqbin': args.freqbin,
        'apply_flags': args.apply_flags,
        'dry_run': args.dry_run,
        'propagate_flags': args.propagate_flags,
        'plots': args.plots,
        'plot_dir': args.plot_dir,
        'mode': args.mode,
        'chunk_size': args.chunk_size,
        'max_memory': args.max_memory,
        'logger': logger,
    }
    
    # Log configuration
    logger.info("  CONFIGURATION")
    logger.info(f"  MS file: {args.ms_path}")
    logger.info(f"  Method: {args.method}")
    logger.info(f"  Data column: {args.datacolumn}")
    logger.info(f"  Passes: {n_passes}")
    for i, pc in enumerate(passes_config):
        if args.method == 'sumthreshold':
            logger.info(f"    Pass {i+1}: sigma={pc['sigma']}, rho={pc['rho']}")
        elif args.method == 'iqr':
            logger.info(f"    Pass {i+1}: iqr_factor={pc['iqr_factor']}")
        elif args.method == 'mad':
            logger.info(f"    Pass {i+1}: mad_sigma={pc['mad_sigma']}")
    if args.timebin:
        logger.info(f"  Time bin: {args.timebin} samples")
    if args.freqbin:
        logger.info(f"  Freq bin: {args.freqbin} channels")
    logger.info(f"  Apply flags: {args.apply_flags}")
    logger.info(f"  Dry run: {args.dry_run}")
    
    # Run
    try:
        results = hunt_ms(args.ms_path, options)
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def run_nimki(args) -> int:
    """Run NIMKI with parsed arguments."""
    from catboss.logger import setup_logger, print_banner, print_cat_on_hunt
    
    # Print banner
    print_banner()
    print_cat_on_hunt("NIMKI")
    
    # Setup logger
    logger = setup_logger("catboss", verbose=args.verbose)
    
    # Build options
    options = {
        'n_components': args.n_components,
        'roam_around': args.roam_around,
        'max_components': args.max_components,
        'min_improvement': args.min_improvement,
        'max_iter': args.max_iter,
        'tolerance': args.tolerance,
        'sigma': args.sigma,
        'datacolumn': args.datacolumn,
        'field': args.field,
        'spw': args.spw,
        'corr': args.corr,
        'timebin': args.timebin,
        'apply_flags': args.apply_flags and not args.dry_run,
        'dry_run': args.dry_run,
        'flag_all_corr': args.flag_all_corr and not args.flag_selected_only,
        'plots': args.plots,
        'plot_dir': args.plot_dir,
        'ncpu': args.ncpu,
        'logger': logger,
    }
    
    logger.info("  CONFIGURATION")
    logger.info(f"  MS file: {args.ms_path}")
    logger.info(f"  Components: {args.n_components}" + 
                (f" -> {args.max_components} (adaptive)" if args.roam_around else ""))
    logger.info(f"  Sigma threshold: {args.sigma}")
    logger.info(f"  Time bin: {args.timebin} min")
    logger.info(f"  Apply flags: {options['apply_flags']}")
    logger.info(f"  Dry run: {args.dry_run}")
    
    # Try to import and run NIMKI
    try:
        from catboss.nimki import hunt_ms as nimki_hunt
        results = nimki_hunt(args.ms_path, options)
        return 0
    except ImportError as e:
        logger.warning(f"NIMKI C++ extension not available: {e}")
        logger.warning("Please build the C++ extension or use POOH instead.")
        logger.info("To build: cd catboss/nimki && pip install -e .")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main():
    """Main CLI entry point."""
    
    # Main parser
    parser = argparse.ArgumentParser(
        prog='catboss',
        description='CATBOSS - Comprehensive Automated Time-frequency Baseline Outlier and Signal Suppression',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available cats:
  pooh     POOH: Parallelized Optimized Outlier Hunter
           Dynamic spectra flagging (SumThreshold, IQR, MAD)

  nimki    NIMKI: Non-linear Interference Modeling and Korrection Interface
           UV-domain Gabor basis fitting

Use 'catboss <cat> --help' for cat-specific options.

Author: Arpan Pal, NCRA-TIFR
        """
    )
    
    parser.add_argument(
        '--version', action='version', version='catboss 1.0.0'
    )
    
    # Subparsers for each cat
    subparsers = parser.add_subparsers(
        dest='cat',
        title='Available cats',
        description='Choose a flagging algorithm',
        metavar='<cat>'
    )
    
    # Add POOH and NIMKI parsers
    create_pooh_parser(subparsers)
    create_nimki_parser(subparsers)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if a cat was selected
    if args.cat is None:
        parser.print_help()
        print("\nNo cat selected. Use 'catboss pooh --help' or 'catboss nimki --help'")
        return 1
    
    # Run the selected cat
    if args.cat == 'pooh':
        return run_pooh(args)
    elif args.cat == 'nimki':
        return run_nimki(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
