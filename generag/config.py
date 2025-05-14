"""Configuration module for Gene RAG system with backward compatibility."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Configuration for Gene RAG system."""
    
    db_dir: Optional[str] = None
    citation_dir: Optional[str] = None
    api_key: Optional[str] = None
    model: str = "deepseek-chat"
    api_url: str = "https://api.deepseek.com/v1/chat/completions"
    
    # Performance options with defaults
    cache_enabled: bool = field(default=True)
    max_retries: int = field(default=3)
    timeout: int = field(default=120)
    batch_size: int = field(default=10)
    
    def __post_init__(self):
        """Override with environment variables if available and validate."""
        # Use environment variables if not set
        if self.db_dir is None:
            self.db_dir = os.getenv("GENERAG_DB_DIR")
        if self.citation_dir is None:
            self.citation_dir = os.getenv("GENERAG_CITATION_DIR")
        if self.api_key is None:
            self.api_key = os.getenv("GENERAG_API_KEY")
            
        # Override model and API URL from environment
        self.model = os.getenv("GENERAG_MODEL", self.model)
        self.api_url = os.getenv("GENERAG_API_URL", self.api_url)
        
        # Override performance options from environment
        if os.getenv("GENERAG_MAX_RETRIES"):
            self.max_retries = int(os.getenv("GENERAG_MAX_RETRIES"))
        if os.getenv("GENERAG_TIMEOUT"):
            self.timeout = int(os.getenv("GENERAG_TIMEOUT"))
        
        # Validate required fields
        if not self.db_dir:
            raise ValueError("db_dir must be specified either directly or via GENERAG_DB_DIR environment variable")
        if not self.api_key:
            raise ValueError("api_key must be specified either directly or via GENERAG_API_KEY environment variable")
            
        # Expand user paths
        self.db_dir = os.path.expanduser(self.db_dir)
        if self.citation_dir:
            self.citation_dir = os.path.expanduser(self.citation_dir)