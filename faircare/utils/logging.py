# ============================================================================
# faircare/utils/logging.py
# ============================================================================

import logging
import csv
from pathlib import Path


def setup_logger(name: str, log_file: Optional[Path] = None, 
                level: int = logging.INFO) -> logging.Logger:
    """Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


class MetricsLogger:
    """Logger for experiment metrics."""
    
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.initialized = False
    
    def log(self, metrics: Dict[str, Any]):
        """Log metrics to CSV file.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        # Initialize CSV with headers
        if not self.initialized:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
                writer.writeheader()
            self.initialized = True
        
        # Append metrics
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            
            # Convert non-scalar values to strings
            processed_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (list, tuple)):
                    processed_metrics[key] = str(value)
                else:
                    processed_metrics[key] = value
            
            writer.writerow(processed_metrics)
