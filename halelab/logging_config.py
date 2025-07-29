import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
import json

class CustomFormatter(logging.Formatter):
    
    def format(self, record):
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        level = record.levelname
        
        if hasattr(record, 'client_id'):
            prefix = f"[Client {record.client_id}]"
        elif "server" in record.name.lower():
            prefix = "[Server]"
        else:
            prefix = f"[{record.name.split('.')[-1]}]"
        
        message = record.getMessage()
        
        if level == "INFO":
            return f"{timestamp} {prefix} {message}"
        elif level == "ERROR":
            return f"{timestamp} {prefix} ❌ ERROR: {message}"
        elif level == "WARNING":
            return f"{timestamp} {prefix} ⚠️  WARNING: {message}"
        elif level == "DEBUG":
            return f"{timestamp} {prefix} 🔍 DEBUG: {message}"
        else:
            return f"{timestamp} {prefix} [{level}] {message}"

def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    service_name: str = "halelab-fl",
    enable_console: bool = True,
    enable_structured: bool = True
):
    
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    logger.handlers.clear()
    
    if enable_structured:
        formatter = CustomFormatter()
    else:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        logger.addHandler(console_handler)
    
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        logger.addHandler(file_handler)
    
    logging.getLogger('flwr').setLevel(getattr(logging, log_level.upper()))
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    logger.info(f"Logging setup complete for {service_name}", extra={
        'service': service_name,
        'log_level': log_level,
        'structured_logging': enable_structured,
        'log_file': log_file
    })
    
    return logger

def get_logger(name: str):
    return logging.getLogger(name)

def log_federated_round(logger, round_num: int, metrics: dict = None, client_id: str = None):
    logger.info(f"Federated round {round_num} completed", extra={
        'round': round_num,
        'client_id': client_id,
        'metrics': metrics,
        'event_type': 'federated_round'
    })

def log_ssl_metrics(logger, ssl_task: str, metrics: dict, client_id: str = None):
    logger.info(f"SSL metrics for {ssl_task}", extra={
        'ssl_task': ssl_task,
        'client_id': client_id,
        'metrics': metrics,
        'event_type': 'ssl_metrics'
    })

def log_client_connection(logger, client_id: str, status: str, server_address: str = None):
    logger.info(f"Client {client_id} {status}", extra={
        'client_id': client_id,
        'status': status,
        'server_address': server_address,
        'event_type': 'client_connection'
    })

def log_server_status(logger, status: str, num_clients: int = None, round_num: int = None):
    logger.info(f"Server {status}", extra={
        'status': status,
        'num_clients': num_clients,
        'round': round_num,
        'event_type': 'server_status'
    })

def log_error(logger, error: Exception, context: str = None, **kwargs):
    logger.error(f"Error in {context}: {str(error)}", extra={
        'error_type': type(error).__name__,
        'context': context,
        'event_type': 'error',
        **kwargs
    }, exc_info=True)