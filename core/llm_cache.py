"""
LLM Response Caching System for Process Tracing Analysis

Provides intelligent caching to reduce redundant LLM calls with hash-based
cache key generation, persistent storage, and performance metrics.

Author: Claude Code Implementation  
Date: August 2025
"""

import hashlib
import json
import time
import sqlite3
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from threading import Lock
import logging


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    content: Any
    created_at: float
    last_accessed: float
    access_count: int
    content_size: int
    model_name: str
    prompt_hash: str
    ttl_seconds: Optional[int] = None


@dataclass
class CacheStats:
    """Cache performance statistics"""
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    total_time_saved: float
    storage_size_mb: float
    entries_count: int


class LLMCache:
    """
    Intelligent LLM response caching system.
    
    Features:
    - Hash-based cache key generation from input text and prompts
    - Persistent SQLite storage with configurable TTL
    - Cache invalidation strategies for updated prompts
    - Performance metrics and analytics
    - Thread-safe operation
    - Automatic cleanup of expired entries
    """
    
    def __init__(self, 
                 cache_dir: Path = None, 
                 default_ttl: int = 3600,  # 1 hour
                 max_cache_size_mb: int = 500):
        """
        Initialize LLM cache.
        
        Args:
            cache_dir: Directory for cache storage (default: ./cache)
            default_ttl: Default TTL in seconds
            max_cache_size_mb: Maximum cache size in MB
        """
        self.cache_dir = cache_dir or Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = self.cache_dir / "llm_cache.db"
        self.default_ttl = default_ttl
        self.max_cache_size_mb = max_cache_size_mb
        self.lock = Lock()
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.stats = CacheStats(
            total_requests=0,
            cache_hits=0, 
            cache_misses=0,
            hit_rate=0.0,
            total_time_saved=0.0,
            storage_size_mb=0.0,
            entries_count=0
        )
        
        self._init_database()
        self._load_stats()
    
    def _init_database(self):
        """Initialize SQLite database for cache storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    content BLOB,
                    created_at REAL,
                    last_accessed REAL,
                    access_count INTEGER,
                    content_size INTEGER,
                    model_name TEXT,
                    prompt_hash TEXT,
                    ttl_seconds INTEGER
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at);
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed);
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_prompt_hash ON cache_entries(prompt_hash);
            """)
    
    def _load_stats(self):
        """Load cache statistics from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count entries
            cursor.execute("SELECT COUNT(*) FROM cache_entries")
            self.stats.entries_count = cursor.fetchone()[0]
            
            # Calculate total size
            cursor.execute("SELECT SUM(content_size) FROM cache_entries")
            total_size = cursor.fetchone()[0] or 0
            self.stats.storage_size_mb = total_size / (1024 * 1024)
    
    def generate_cache_key(self, 
                          text: str, 
                          prompt_template: str,
                          model_name: str = "gemini",
                          additional_params: Optional[Dict] = None) -> str:
        """
        Generate hash-based cache key from input parameters.
        
        Args:
            text: Input text to analyze
            prompt_template: LLM prompt template
            model_name: Model identifier
            additional_params: Additional parameters affecting output
            
        Returns:
            Hexadecimal cache key
        """
        # Create content for hashing
        content_parts = [
            text.strip(),
            prompt_template.strip(), 
            model_name,
            json.dumps(additional_params or {}, sort_keys=True)
        ]
        
        content = "|".join(content_parts)
        
        # Generate SHA-256 hash
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def generate_prompt_hash(self, prompt_template: str) -> str:
        """Generate hash for prompt template for invalidation tracking"""
        return hashlib.md5(prompt_template.strip().encode('utf-8')).hexdigest()
    
    def get(self, cache_key: str) -> Optional[Any]:
        """
        Retrieve entry from cache.
        
        Args:
            cache_key: Cache key to look up
            
        Returns:
            Cached content or None if not found/expired
        """
        with self.lock:
            self.stats.total_requests += 1
            
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT content, created_at, ttl_seconds, access_count
                        FROM cache_entries 
                        WHERE key = ?
                    """, (cache_key,))
                    
                    result = cursor.fetchone()
                    if not result:
                        self.stats.cache_misses += 1
                        self._update_hit_rate()
                        return None
                    
                    content_blob, created_at, ttl_seconds, access_count = result
                    
                    # Check TTL expiration
                    if ttl_seconds and (time.time() - created_at) > ttl_seconds:
                        # Entry expired, remove it
                        cursor.execute("DELETE FROM cache_entries WHERE key = ?", (cache_key,))
                        conn.commit()
                        self.stats.cache_misses += 1
                        self._update_hit_rate()
                        return None
                    
                    # Update access statistics
                    cursor.execute("""
                        UPDATE cache_entries 
                        SET last_accessed = ?, access_count = ?
                        WHERE key = ?
                    """, (time.time(), access_count + 1, cache_key))
                    
                    conn.commit()
                    
                    # Deserialize content
                    content = pickle.loads(content_blob)
                    
                    self.stats.cache_hits += 1
                    self._update_hit_rate()
                    
                    self.logger.debug(f"Cache HIT for key {cache_key[:8]}...")
                    return content
                    
            except Exception as e:
                self.logger.error(f"Cache get error: {e}")
                return None
    
    def put(self, 
            cache_key: str, 
            content: Any,
            model_name: str = "gemini",
            prompt_template: str = "",
            ttl_seconds: Optional[int] = None) -> bool:
        """
        Store entry in cache.
        
        Args:
            cache_key: Cache key
            content: Content to cache
            model_name: Model name for metadata
            prompt_template: Prompt template for invalidation
            ttl_seconds: TTL override (uses default if None)
            
        Returns:
            True if stored successfully
        """
        with self.lock:
            try:
                # Serialize content
                content_blob = pickle.dumps(content)
                content_size = len(content_blob)
                
                # Check cache size limits
                if content_size > 50 * 1024 * 1024:  # 50MB per entry limit
                    self.logger.warning(f"Entry too large ({content_size/1024/1024:.1f}MB), skipping cache")
                    return False
                
                current_time = time.time()
                prompt_hash = self.generate_prompt_hash(prompt_template)
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Insert or replace entry
                    cursor.execute("""
                        INSERT OR REPLACE INTO cache_entries
                        (key, content, created_at, last_accessed, access_count, 
                         content_size, model_name, prompt_hash, ttl_seconds)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        cache_key, content_blob, current_time, current_time, 0,
                        content_size, model_name, prompt_hash, ttl_seconds or self.default_ttl
                    ))
                    
                    conn.commit()
                
                # Update stats
                self.stats.entries_count += 1
                self.stats.storage_size_mb += content_size / (1024 * 1024)
                
                # Cleanup if cache too large
                self._cleanup_if_needed()
                
                self.logger.debug(f"Cache STORE for key {cache_key[:8]}... ({content_size} bytes)")
                return True
                
            except Exception as e:
                self.logger.error(f"Cache put error: {e}")
                return False  
    
    def invalidate_by_prompt(self, prompt_template: str) -> int:
        """
        Invalidate all cache entries with matching prompt template.
        
        Args:
            prompt_template: Prompt template to invalidate
            
        Returns:
            Number of entries invalidated
        """
        with self.lock:
            try:
                prompt_hash = self.generate_prompt_hash(prompt_template)
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Count entries to be deleted
                    cursor.execute("SELECT COUNT(*) FROM cache_entries WHERE prompt_hash = ?", (prompt_hash,))
                    count = cursor.fetchone()[0]
                    
                    # Delete entries
                    cursor.execute("DELETE FROM cache_entries WHERE prompt_hash = ?", (prompt_hash,))
                    conn.commit()
                
                self.stats.entries_count -= count
                self._load_stats()  # Reload storage size
                
                self.logger.info(f"Invalidated {count} cache entries for prompt change")
                return count
                
            except Exception as e:
                self.logger.error(f"Cache invalidation error: {e}")
                return 0
    
    def clear_expired(self) -> int:
        """
        Remove expired cache entries.
        
        Returns:
            Number of entries removed
        """
        with self.lock:
            try:
                current_time = time.time()
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Count expired entries
                    cursor.execute("""
                        SELECT COUNT(*) FROM cache_entries 
                        WHERE ttl_seconds IS NOT NULL 
                        AND (? - created_at) > ttl_seconds
                    """, (current_time,))
                    count = cursor.fetchone()[0]
                    
                    # Delete expired entries
                    cursor.execute("""
                        DELETE FROM cache_entries 
                        WHERE ttl_seconds IS NOT NULL 
                        AND (? - created_at) > ttl_seconds
                    """, (current_time,))
                    conn.commit()
                
                self.stats.entries_count -= count
                self._load_stats()  # Reload storage size
                
                if count > 0:
                    self.logger.info(f"Cleaned up {count} expired cache entries")
                return count
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                return 0
    
    def _cleanup_if_needed(self):
        """Cleanup cache if it exceeds size limits"""
        if self.stats.storage_size_mb > self.max_cache_size_mb:
            # Remove oldest 25% of entries
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total count
                cursor.execute("SELECT COUNT(*) FROM cache_entries")
                total_count = cursor.fetchone()[0]
                
                # Remove oldest entries
                remove_count = max(1, total_count // 4)
                cursor.execute("""
                    DELETE FROM cache_entries 
                    WHERE key IN (
                        SELECT key FROM cache_entries 
                        ORDER BY last_accessed ASC 
                        LIMIT ?
                    )
                """, (remove_count,))
                
                conn.commit()
                
            self._load_stats()
            self.logger.info(f"Cleaned up {remove_count} old cache entries (size limit)")
    
    def _update_hit_rate(self):
        """Update cache hit rate statistic"""
        if self.stats.total_requests > 0:
            self.stats.hit_rate = self.stats.cache_hits / self.stats.total_requests
    
    def get_stats(self) -> CacheStats:
        """Get current cache statistics"""
        with self.lock:
            self._load_stats()
            return self.stats
    
    def estimate_time_saved(self, cache_hit_duration: float = 5.0) -> float:
        """
        Estimate time saved by cache hits.
        
        Args:
            cache_hit_duration: Estimated time per LLM call in seconds
            
        Returns:
            Total estimated time saved in seconds
        """
        return self.stats.cache_hits * cache_hit_duration
    
    def print_stats(self):
        """Print cache performance statistics"""
        stats = self.get_stats()
        
        print("\n" + "="*50)
        print("LLM CACHE PERFORMANCE STATISTICS")
        print("="*50)
        print(f"Total Requests: {stats.total_requests}")
        print(f"Cache Hits: {stats.cache_hits}")
        print(f"Cache Misses: {stats.cache_misses}")
        print(f"Hit Rate: {stats.hit_rate:.1%}")
        print(f"Entries Count: {stats.entries_count}")
        print(f"Storage Size: {stats.storage_size_mb:.1f}MB")
        
        time_saved = self.estimate_time_saved()
        if time_saved > 0:
            print(f"Estimated Time Saved: {time_saved:.1f}s ({time_saved/60:.1f}m)")
        
        print("="*50)


# Global cache instance
_global_cache: Optional[LLMCache] = None


def get_cache() -> LLMCache:
    """Get or create global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = LLMCache()
    return _global_cache


def cached_llm_call(text: str,
                   prompt_template: str, 
                   model_name: str = "gemini",
                   llm_function: callable = None,
                   additional_params: Optional[Dict] = None,
                   ttl_seconds: Optional[int] = None) -> Any:
    """
    Wrapper for cached LLM calls.
    
    Args:
        text: Input text
        prompt_template: Prompt template
        model_name: Model identifier
        llm_function: Function to call if cache miss
        additional_params: Additional parameters
        ttl_seconds: Cache TTL override
        
    Returns:
        LLM response (from cache or fresh call)
    """
    cache = get_cache()
    
    # Generate cache key
    cache_key = cache.generate_cache_key(text, prompt_template, model_name, additional_params)
    
    # Try cache first
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    # Cache miss - make fresh LLM call
    if llm_function is None:
        raise ValueError("llm_function required for cache miss")
    
    fresh_result = llm_function(text, prompt_template, **additional_params or {})
    
    # Store in cache
    cache.put(cache_key, fresh_result, model_name, prompt_template, ttl_seconds)
    
    return fresh_result


if __name__ == "__main__":
    # Demo usage
    cache = LLMCache()
    
    # Simulate cache operations
    cache_key = cache.generate_cache_key("test text", "analyze: {text}", "gemini")
    cache.put(cache_key, {"result": "test analysis"}, "gemini", "analyze: {text}")
    
    result = cache.get(cache_key)
    print(f"Cache result: {result}")
    
    cache.print_stats()