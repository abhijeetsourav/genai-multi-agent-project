#!/usr/bin/env python3
"""
Main script to run multi-agent system
"""
import argparse
from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger()


def main(task: str):
    """Run multi-agent system."""
    logger.info(f"Starting multi-agent system with task: {task}")
    
    # Initialize agents
    # Run orchestration
    # Return results
    
    logger.info("Task completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-agent system")
    parser.add_argument("--task", type=str, required=True, help="Task description")
    args = parser.parse_args()
    
    main(args.task)
