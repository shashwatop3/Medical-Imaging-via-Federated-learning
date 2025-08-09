#!/usr/bin/env python3
"""
Distributed Server Entry Point for HaleLab Federated Learning
This is the main entry point for running the FL server in Kubernetes
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the halelab package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from halelab.server_app import multi_task_ssl_server_fn
from halelab.logging_config import setup_logging
import flwr as fl


def main():
    """Main entry point for distributed FL server"""
    parser = argparse.ArgumentParser(description='HaleLab FL Distributed Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--num-rounds', type=int, default=10, help='Number of FL rounds')
    parser.add_argument('--min-clients', type=int, default=2, help='Minimum clients')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting HaleLab FL Server on {args.host}:{args.port}")
    logger.info(f"Configuration: {args.num_rounds} rounds, {args.min_clients} min clients")
    
    # Get server strategy
    strategy = multi_task_ssl_server_fn()
    
    # Start Flower server
    fl.server.start_server(
        server_address=f"{args.host}:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()