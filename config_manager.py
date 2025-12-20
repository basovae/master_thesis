"""
Configuration Manager for Portfolio Optimization Experiments

Handles loading default config from YAML and merging with runtime overrides.
Designed for use in Jupyter notebooks with clean syntax.

Usage in notebook:
    from config_manager import Config
    
    # Load defaults
    cfg = Config()
    
    # Override specific values
    cfg = Config(
        ddpg={'actor_lr': 0.0005, 'total_timesteps': 100000},
        validation={'strategy': 'holdout'}
    )
    
    # Access values
    print(cfg.ddpg.actor_lr)  # 0.0005
    print(cfg.ddpg.batch_size)  # 64 (from defaults)
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from copy import deepcopy
import json


class DotDict(dict):
    """Dictionary with dot notation access for cleaner notebook usage."""
    
    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, dict) and not isinstance(value, DotDict):
                value = DotDict(value)
                self[key] = value
            return value
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
    
    def to_dict(self) -> dict:
        """Convert back to regular dict (recursive)."""
        result = {}
        for k, v in self.items():
            if isinstance(v, DotDict):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result


def deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge override into base dict.
    Override values take precedence.
    """
    result = deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    
    return result


class Config:
    """
    Configuration manager with YAML defaults and runtime overrides.
    
    Parameters
    ----------
    config_path : str or Path, optional
        Path to YAML config file. Defaults to 'config/default_config.yaml'
    **overrides : dict
        Keyword arguments to override specific config sections.
        Each kwarg should be a dict matching the config structure.
    
    Examples
    --------
    >>> cfg = Config()
    >>> cfg.ddpg.actor_lr
    0.0001
    
    >>> cfg = Config(ddpg={'actor_lr': 0.0005})
    >>> cfg.ddpg.actor_lr
    0.0005
    
    >>> cfg = Config(
    ...     validation={'strategy': 'holdout'},
    ...     ddpg={'total_timesteps': 100000}
    ... )
    """
    
    DEFAULT_CONFIG_PATHS = [
        Path("default_config.yaml"),
        Path("config/default_config.yaml"),
        Path("../config/default_config.yaml"),
    ]
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        **overrides
    ):
        # Load base config
        self._config_path = self._find_config(config_path)
        self._base_config = self._load_yaml(self._config_path)
        
        # Apply overrides
        self._overrides = overrides
        self._config = deep_merge(self._base_config, overrides)
        
        # Convert to DotDict for convenient access
        self._dotdict = DotDict(self._config)
    
    def _find_config(self, config_path: Optional[Union[str, Path]]) -> Path:
        """Find config file, checking multiple default locations."""
        if config_path is not None:
            path = Path(config_path)
            if path.exists():
                return path
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        for path in self.DEFAULT_CONFIG_PATHS:
            if path.exists():
                return path
        
        raise FileNotFoundError(
            f"No default config found. Searched: {self.DEFAULT_CONFIG_PATHS}"
        )
    
    def _load_yaml(self, path: Path) -> dict:
        """Load YAML config file."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def __getattr__(self, key):
        """Allow attribute access to config sections."""
        if key.startswith('_'):
            return super().__getattribute__(key)
        return getattr(self._dotdict, key)
    
    def __getitem__(self, key):
        """Allow dictionary-style access."""
        return self._dotdict[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get nested value using dot notation string."""
        keys = key.split('.')
        value = self._dotdict
        for k in keys:
            try:
                value = value[k]
            except (KeyError, TypeError):
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set nested value using dot notation string."""
        keys = key.split('.')
        d = self._dotdict
        for k in keys[:-1]:
            if k not in d:
                d[k] = DotDict()
            d = d[k]
        d[keys[-1]] = value
    
    def to_dict(self) -> dict:
        """Export full config as regular dict."""
        return self._dotdict.to_dict()
    
    def save(self, path: Union[str, Path]) -> None:
        """Save current config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def __repr__(self) -> str:
        return f"Config(path='{self._config_path}', overrides={list(self._overrides.keys())})"
    
    def summary(self) -> str:
        """Print human-readable config summary."""
        lines = ["=" * 60, "Configuration Summary", "=" * 60]
        
        for section, values in self._dotdict.items():
            lines.append(f"\n[{section}]")
            if isinstance(values, dict):
                for k, v in values.items():
                    if isinstance(v, dict):
                        lines.append(f"  {k}:")
                        for kk, vv in v.items():
                            lines.append(f"    {kk}: {vv}")
                    else:
                        lines.append(f"  {k}: {v}")
            else:
                lines.append(f"  {values}")
        
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Convenience functions for common experiment setups
# -----------------------------------------------------------------------------

def quick_test_config() -> Config:
    """
    Config for quick testing - reduced training steps and simpler setup.
    Good for debugging and initial development.
    """
    return Config(
        data={'years_of_data': 5},
        validation={
            'strategy': 'holdout',
            'holdout': {'train_ratio': 0.8, 'val_ratio': 0.1, 'test_ratio': 0.1},
            'n_seeds': 1
        },
        ddpg={'total_timesteps': 10000, 'warmup_steps': 1000},
        div_ddpg={'total_timesteps': 10000, 'warmup_steps': 1000},
        pga_map_elites={
            'n_iterations': 50,
            'archive': {'n_niches': 100},
            'initial_population': 20
        },
        logging={'log_freq': 100, 'eval_freq': 1000}
    )


def full_experiment_config() -> Config:
    """
    Config for full thesis experiments with multiple seeds and thorough evaluation.
    """
    return Config(
        validation={
            'strategy': 'expanding_window',
            'n_seeds': 5,
            'temporal': {
                'initial_train_years': 10,
                'val_window_years': 1,
                'final_test_years': 3
            }
        },
        ddpg={'total_timesteps': 500000},
        div_ddpg={'total_timesteps': 500000},
        pga_map_elites={
            'n_iterations': 1000,
            'archive': {'n_niches': 1024}
        },
        logging={'use_wandb': True}
    )


def hyperparameter_search_config(method: str) -> Config:
    """
    Base config for hyperparameter search.
    Uses reduced timesteps since we're comparing many configs.
    
    Parameters
    ----------
    method : str
        One of 'ddpg', 'div_ddpg', 'pga_map_elites'
    """
    base = Config(
        validation={
            'strategy': 'holdout',
            'n_seeds': 3  # Fewer seeds during search
        }
    )
    
    if method == 'ddpg':
        base.set('ddpg.total_timesteps', 100000)
    elif method == 'div_ddpg':
        base.set('div_ddpg.total_timesteps', 100000)
    elif method == 'pga_map_elites':
        base.set('pga_map_elites.n_iterations', 200)
    
    return base


# -----------------------------------------------------------------------------
# Hyperparameter search space definitions (for use with Optuna)
# -----------------------------------------------------------------------------

DDPG_SEARCH_SPACE = {
    'actor_lr': {'type': 'loguniform', 'low': 1e-5, 'high': 1e-3},
    'critic_lr': {'type': 'loguniform', 'low': 1e-5, 'high': 1e-2},
    'batch_size': {'type': 'categorical', 'choices': [32, 64, 128, 256]},
    'tau': {'type': 'loguniform', 'low': 0.001, 'high': 0.05},
    'gamma': {'type': 'uniform', 'low': 0.95, 'high': 0.999},
    'ou_sigma': {'type': 'uniform', 'low': 0.1, 'high': 0.5},
}

DIV_DDPG_SEARCH_SPACE = {
    **DDPG_SEARCH_SPACE,
    'diversity.alpha_initial': {'type': 'uniform', 'low': 0.1, 'high': 2.0},
    'diversity.scaling_method': {'type': 'categorical', 'choices': ['linear_decay', 'distance_based', 'fixed']},
}

PGA_SEARCH_SPACE = {
    'actor_lr': {'type': 'loguniform', 'low': 1e-5, 'high': 1e-3},
    'critic_lr': {'type': 'loguniform', 'low': 1e-5, 'high': 1e-3},
    'batch_size': {'type': 'categorical', 'choices': [64, 128, 256, 512]},
    'variation.pg_probability': {'type': 'uniform', 'low': 0.3, 'high': 0.7},
    'variation.iso_sigma': {'type': 'loguniform', 'low': 0.001, 'high': 0.1},
    'variation.line_sigma': {'type': 'uniform', 'low': 0.05, 'high': 0.3},
}


if __name__ == "__main__":
    # Test the config system
    cfg = Config()
    print(cfg.summary())