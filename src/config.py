"""Configuration management for VARBX due diligence."""

import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional


class Config:
    """Configuration manager that loads and validates config.yml."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize configuration from YAML file.

        Args:
            config_path: Path to config.yml. If None, uses project root.
        """
        if config_path is None:
            # Assume we're in src/, go up to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config.yml"

        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation).

        Args:
            key: Configuration key, e.g., 'paths.data_raw' or 'analysis.risk_free_rate'
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    @property
    def paths(self) -> Dict[str, str]:
        """Get all path configurations."""
        return self._config.get("paths", {})

    @property
    def data(self) -> Dict[str, Any]:
        """Get all data configurations."""
        return self._config.get("data", {})

    @property
    def analysis(self) -> Dict[str, Any]:
        """Get all analysis configurations."""
        return self._config.get("analysis", {})

    @property
    def viz(self) -> Dict[str, Any]:
        """Get all visualization configurations."""
        return self._config.get("viz", {})

    @property
    def export(self) -> Dict[str, Any]:
        """Get all export configurations."""
        return self._config.get("export", {})


# Global config instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance.

    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

