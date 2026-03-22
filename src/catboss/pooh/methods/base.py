"""
Base class for POOH flagging methods.

Author: Arpan Pal
Institution: NCRA-TIFR
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple


class BaseFlaggingMethod(ABC):
    """
    Abstract base class for flagging methods.
    
    All methods must implement:
    - flag(): Single baseline flagging (legacy interface)
    - flag_batch(): Batch processing (primary interface for GPU)
    """
    
    name: str = "base"
    
    def __init__(self, params: Dict[str, Any], logger=None):
        """
        Initialize method with parameters.
        
        Args:
            params: Dictionary of method parameters
            logger: Optional logger
        """
        self.params = params
        self.logger = logger
    
    @abstractmethod
    def flag(
        self,
        amp: np.ndarray,
        flags: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply flagging to single baseline data (legacy interface).
        
        Args:
            amp: Amplitude array (n_time × n_freq)
            flags: Existing flag array (same shape)
            thresholds: Optional pre-calculated thresholds
            
        Returns:
            New flag array (combined with existing)
        """
        pass
    
    @abstractmethod
    def flag_batch(
        self,
        amp_batch: np.ndarray,
        flags_batch: np.ndarray,
        thresholds_batch: Optional[np.ndarray] = None,
        gpu_arrays=None
    ) -> np.ndarray:
        """
        Apply flagging to batch of baselines (primary GPU interface).
        
        This is the preferred method for GPU processing as it allows
        all baselines to be processed together with minimal transfers.
        
        Args:
            amp_batch: Amplitude array (n_baselines × n_time × n_freq), float32
            flags_batch: Flag array (n_baselines × n_time × n_freq), uint8
            thresholds_batch: Optional thresholds (n_baselines × n_freq)
            
        Returns:
            Updated flags array (modified in-place and returned)
        """
        pass
    
    def log(self, msg: str, level: str = 'debug'):
        """Log message if logger available."""
        if self.logger:
            getattr(self.logger, level)(msg)
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Get default parameters for this method."""
        return {}
    
    @classmethod
    def get_param_help(cls) -> Dict[str, str]:
        """Get parameter descriptions."""
        return {}
