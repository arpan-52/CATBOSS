"""
Basic tests for CATBOSS.
"""

import numpy as np
import pytest


class TestThresholds:
    """Tests for threshold calculation."""
    
    def test_calculate_channel_medians(self):
        """Test median calculation."""
        from catboss.pooh.thresholds import calculate_robust_thresholds
        
        # Create test data
        np.random.seed(42)
        amp = np.random.randn(100, 50).astype(np.float32) + 10
        flags = np.zeros_like(amp, dtype=bool)
        
        thresholds = calculate_robust_thresholds(amp, flags, sigma_factor=3.0)
        
        assert thresholds.shape == (50,)
        assert np.all(thresholds > 0)
    
    def test_calculate_mad_thresholds(self):
        """Test MAD calculation."""
        from catboss.pooh.thresholds import calculate_robust_thresholds
        
        np.random.seed(42)
        amp = np.random.randn(100, 50).astype(np.float32) + 10
        flags = np.zeros_like(amp, dtype=bool)
        
        thresholds = calculate_robust_thresholds(amp, flags, sigma_factor=3.0, method='mad')
        
        assert thresholds.shape == (50,)
        assert np.all(thresholds > 0)


class TestBandpass:
    """Tests for bandpass normalization."""
    
    def test_polynomial_bandpass(self):
        """Test polynomial bandpass fitting."""
        from catboss.pooh.bandpass import compute_polynomial_bandpass
        
        # Create test data with bandpass shape
        np.random.seed(42)
        n_time, n_chan = 100, 256
        
        # Synthetic bandpass
        channels = np.arange(n_chan)
        bandpass_true = 10 + 2 * np.sin(2 * np.pi * channels / n_chan)
        
        # Add noise
        amp = np.outer(np.ones(n_time), bandpass_true) + np.random.randn(n_time, n_chan) * 0.5
        amp = amp.astype(np.float32)
        flags = np.zeros_like(amp, dtype=bool)
        
        poly_fit, bad_channels = compute_polynomial_bandpass(
            amp, flags, deviation_threshold=5.0, poly_order=5
        )
        
        assert poly_fit.shape == (n_chan,)
        assert bad_channels.shape == (n_chan,)
        
        # Fit should be reasonable
        correlation = np.corrcoef(poly_fit, bandpass_true)[0, 1]
        assert correlation > 0.9


class TestMethods:
    """Tests for flagging methods."""
    
    def test_sumthreshold_method(self):
        """Test SumThreshold flagging."""
        from catboss.pooh.methods import SumThresholdMethod
        
        np.random.seed(42)
        n_time, n_chan = 100, 50
        
        # Create data with RFI
        amp = np.random.randn(n_time, n_chan).astype(np.float32) + 5
        
        # Add RFI
        amp[20:25, 10:15] = 50  # Broadband burst
        amp[50, :] = 30  # Time sample
        amp[:, 40] = 25  # Frequency channel
        
        flags = np.zeros_like(amp, dtype=bool)
        
        method = SumThresholdMethod({
            'sigma': 5.0,
            'rho': 1.5,
            'combinations': [1, 2, 4, 8],
            'use_gpu': False,
        })
        
        new_flags = method.flag(amp, flags)
        
        # Should flag the RFI
        assert np.sum(new_flags) > 0
        assert new_flags[22, 12]  # Broadband
        assert new_flags[50, 25]  # Time
    
    def test_iqr_method(self):
        """Test IQR flagging."""
        from catboss.pooh.methods import IQRMethod
        
        np.random.seed(42)
        n_time, n_chan = 100, 50
        
        amp = np.random.randn(n_time, n_chan).astype(np.float32) + 5
        amp[30, 20] = 100  # Outlier
        
        flags = np.zeros_like(amp, dtype=bool)
        
        method = IQRMethod({'iqr_factor': 1.5, 'use_gpu': False})
        new_flags = method.flag(amp, flags)
        
        assert new_flags[30, 20]  # Outlier flagged
    
    def test_mad_method(self):
        """Test MAD flagging."""
        from catboss.pooh.methods import MADMethod
        
        np.random.seed(42)
        n_time, n_chan = 100, 50
        
        amp = np.random.randn(n_time, n_chan).astype(np.float32) + 5
        amp[40, 30] = 100  # Outlier
        
        flags = np.zeros_like(amp, dtype=bool)
        
        method = MADMethod({'mad_sigma': 5.0, 'use_gpu': False})
        new_flags = method.flag(amp, flags)
        
        assert new_flags[40, 30]  # Outlier flagged


class TestNIMKI:
    """Tests for NIMKI functions."""
    
    def test_uv_distances(self):
        """Test UV distance calculation."""
        from catboss.nimki.core_functions import calculate_uv_distances
        
        # Test UVW
        uvw = np.array([
            [100.0, 200.0, 50.0],
            [150.0, 100.0, 30.0],
        ], dtype=np.float64)
        
        wavelengths = np.array([0.21, 0.20, 0.19], dtype=np.float64)  # ~1.4 GHz
        
        uv_dist = calculate_uv_distances(uvw, wavelengths)
        
        assert uv_dist.shape == (2, 3)
        assert np.all(uv_dist > 0)
    
    def test_gabor_fit_python(self):
        """Test Python Gabor fitting fallback."""
        from catboss.nimki.core_functions import fit_gabor
        
        np.random.seed(42)
        
        # Create test data
        uv = np.linspace(10, 1000, 500)
        amp = 10 * np.exp(-uv / 300) + np.random.randn(500) * 0.5
        
        result = fit_gabor(uv, amp, n_components=3)
        
        assert 'predicted' in result
        assert 'residuals' in result
        assert 'rms' in result
        assert len(result['predicted']) == 500
    
    def test_flag_outliers(self):
        """Test outlier flagging."""
        from catboss.nimki.core_functions import flag_outliers
        
        np.random.seed(42)
        
        amplitudes = np.random.randn(100) + 10
        predicted = np.full(100, 10.0)
        
        # Add outlier
        amplitudes[50] = 100
        
        outliers, residuals, mad_sigma = flag_outliers(amplitudes, predicted, sigma_threshold=5.0)
        
        assert outliers[50]  # Outlier detected
        assert np.sum(outliers) >= 1


class TestIO:
    """Tests for I/O functions."""
    
    def test_parse_selection(self):
        """Test selection parsing."""
        from catboss.io import parse_selection
        
        assert parse_selection(None) is None
        assert parse_selection('all') is None
        assert parse_selection('0,1,2') == [0, 1, 2]
        assert parse_selection('0, 1, 2') == [0, 1, 2]
    
    def test_parse_baseline_selection(self):
        """Test baseline selection parsing."""
        from catboss.io import parse_baseline_selection
        
        assert parse_baseline_selection(None) is None
        assert parse_baseline_selection('all') is None
        assert parse_baseline_selection('0-1,0-2') == [(0, 1), (0, 2)]


class TestGPU:
    """Tests for GPU utilities."""
    
    def test_gpu_detection(self):
        """Test GPU availability check."""
        from catboss.utils import is_gpu_available
        
        # Should return bool without error
        result = is_gpu_available()
        assert isinstance(result, bool)
    
    def test_batch_size_calculation(self):
        """Test batch size calculation."""
        from catboss.utils import calculate_batch_size

        batch_size, _ = calculate_batch_size(
            n_time=1000,
            n_freq=256,
            n_corr=4,
            gpu_free_mem=4e9,  # 4 GB
            system_free_mem=16e9,  # 16 GB
        )

        assert batch_size >= 1
        assert isinstance(batch_size, int)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
