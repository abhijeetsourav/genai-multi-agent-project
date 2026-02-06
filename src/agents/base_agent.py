"""
Base Agent Class for Multi-Agent System
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(
        self,
        name: str,
        llm: Any,
        tools: Optional[list] = None,
        verbose: bool = False
    ):
        self.name = name
        self.llm = llm
        self.tools = tools or []
        self.verbose = verbose
        logger.info(f"Initialized {self.name} agent")
    
    @abstractmethod
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Execute the agent's main task."""
        pass
    
    def log(self, message: str) -> None:
        """Log agent activity."""
        if self.verbose:
            logger.info(f"[{self.name}] {message}")
