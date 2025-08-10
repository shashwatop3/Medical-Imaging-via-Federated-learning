#!/usr/bin/env python3
"""
Distributed Client Entry Point for HaleLab Federated Learning
This is the main entry point for running FL clients in Kubernetes
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the halelab package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from halelab.client_app import create_distributed_client
from halelab.logging_config import setup_logging
import flwr as fl


def main():
    """Main entry point for distributed FL client"""
    parser = argparse.ArgumentParser(description='HaleLab FL Distributed Client')
    parser.add_argument('--server-address', type=str, required=True, help='FL server address')
    parser.add_argument('--client-id', type=int, required=True, help='Client ID')
    parser.add_argument('--ssl-task', type=str, required=True, 
                       choices=['rotation', 'contrastive'],
                       help='SSL task type')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting HaleLab FL Client {args.client_id}")
    logger.info(f"Server: {args.server_address}, SSL Task: {args.ssl_task}")
    
    # Create client instance
    client = create_distributed_client(
        client_id=args.client_id,
        ssl_task=args.ssl_task,
        dataset_name="HAM10000"
    )
    
    # Start Flower client
    fl.client.start_client(
        server_address=args.server_address,
        client=client
    )


if __name__ == "__main__":
    main()