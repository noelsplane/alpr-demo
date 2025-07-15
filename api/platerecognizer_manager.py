"""
Intelligent management of PlateRecognizer API calls with caching and rate limiting.
"""

import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class PlateRecognizerManager:
    """Manage PlateRecognizer API calls with caching and rate limiting."""
    
    def __init__(self, cache_dir: str = "api_cache", 
                 cache_ttl_hours: int = 24,
                 monthly_limit: int = 2500):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.monthly_limit = monthly_limit
        
        # Load or create usage stats
        self.stats_file = self.cache_dir / "usage_stats.json"
        self.usage_stats = self._load_usage_stats()
    
    def _load_usage_stats(self) -> Dict:
        """Load usage statistics from file."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'current_month': datetime.now().strftime('%Y-%m'),
            'calls_this_month': 0,
            'last_reset': datetime.now().isoformat(),
            'total_calls': 0,
            'cache_hits': 0
        }
    
    def _save_usage_stats(self):
        """Save usage statistics to file."""
        with open(self.stats_file, 'w') as f:
            json.dump(self.usage_stats, f, indent=2)
    
    def _reset_monthly_counter_if_needed(self):
        """Reset monthly counter if we're in a new month."""
        current_month = datetime.now().strftime('%Y-%m')
        if self.usage_stats['current_month'] != current_month:
            self.usage_stats['current_month'] = current_month
            self.usage_stats['calls_this_month'] = 0
            self.usage_stats['last_reset'] = datetime.now().isoformat()
            self._save_usage_stats()
    
    def _hash_image(self, image_path: str) -> str:
        """Create hash of image file for caching."""
        hasher = hashlib.md5()
        with open(image_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    
    def _get_cache_path(self, image_hash: str) -> Path:
        """Get cache file path for image hash."""
        return self.cache_dir / f"{image_hash}.json"
    
    def _load_from_cache(self, image_hash: str) -> Optional[Dict]:
        """Load cached result if available and not expired."""
        cache_path = self._get_cache_path(image_hash)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            # Check if cache is still valid
            cached_time = datetime.fromisoformat(cached_data['timestamp'])
            if datetime.now() - cached_time < self.cache_ttl:
                self.usage_stats['cache_hits'] += 1
                self._save_usage_stats()
                logger.info(f"Cache hit for image hash: {image_hash}")
                return cached_data['result']
            else:
                # Cache expired, delete it
                cache_path.unlink()
                logger.info(f"Cache expired for image hash: {image_hash}")
                
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
        
        return None
    
    def _save_to_cache(self, image_hash: str, result: Dict):
        """Save API result to cache."""
        cache_path = self._get_cache_path(image_hash)
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'result': result
        }
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Cached result for image hash: {image_hash}")
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    def can_make_api_call(self) -> bool:
        """Check if we can make another API call within limits."""
        self._reset_monthly_counter_if_needed()
        return self.usage_stats['calls_this_month'] < self.monthly_limit
    
    def get_remaining_calls(self) -> int:
        """Get number of API calls remaining this month."""
        self._reset_monthly_counter_if_needed()
        return max(0, self.monthly_limit - self.usage_stats['calls_this_month'])
    
    def process_image(self, image_path: str, api_function) -> Optional[Dict]:
        """
        Process image with caching and rate limiting.
        
        Args:
            image_path: Path to image file
            api_function: Function to call PlateRecognizer API
            
        Returns:
            API result or None if rate limited
        """
        # Check cache first
        image_hash = self._hash_image(image_path)
        cached_result = self._load_from_cache(image_hash)
        
        if cached_result is not None:
            logger.info("Returning cached result")
            return cached_result
        
        # Check rate limit
        if not self.can_make_api_call():
            logger.warning(f"Rate limit reached: {self.usage_stats['calls_this_month']}/{self.monthly_limit}")
            return None
        
        # Make API call
        logger.info(f"Making API call ({self.get_remaining_calls()} remaining)")
        result = api_function(image_path)
        
        if result:
            # Update stats
            self.usage_stats['calls_this_month'] += 1
            self.usage_stats['total_calls'] += 1
            self._save_usage_stats()
            
            # Cache result
            self._save_to_cache(image_hash, result)
        
        return result
    
    def get_usage_report(self) -> Dict:
        """Get current usage statistics."""
        self._reset_monthly_counter_if_needed()
        return {
            'current_month': self.usage_stats['current_month'],
            'calls_this_month': self.usage_stats['calls_this_month'],
            'remaining_calls': self.get_remaining_calls(),
            'monthly_limit': self.monthly_limit,
            'total_calls': self.usage_stats['total_calls'],
            'cache_hits': self.usage_stats['cache_hits'],
            'cache_hit_rate': (self.usage_stats['cache_hits'] / 
                             max(1, self.usage_stats['total_calls'] + self.usage_stats['cache_hits']))
        }
    
    def clear_cache(self, older_than_hours: Optional[int] = None):
        """Clear cache files older than specified hours."""
        cleared_count = 0
        
        for cache_file in self.cache_dir.glob("*.json"):
            if cache_file.name == "usage_stats.json":
                continue
                
            try:
                if older_than_hours:
                    # Check age
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    cached_time = datetime.fromisoformat(data['timestamp'])
                    
                    if datetime.now() - cached_time > timedelta(hours=older_than_hours):
                        cache_file.unlink()
                        cleared_count += 1
                else:
                    # Clear all cache
                    cache_file.unlink()
                    cleared_count += 1
                    
            except Exception as e:
                logger.error(f"Error clearing cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {cleared_count} cache files")
        return cleared_count


# Create global instance
pr_manager = PlateRecognizerManager()