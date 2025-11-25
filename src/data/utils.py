"""Helper Utility functions"""
from typing import Callable

def lazy(func: Callable) -> property:
    """Decorator for caching properties after first load."""
    attr_name = f"_{func.__name__}"

    @property
    def _lazy(self):
        """Gets attribute if it exists otherwise it sets the attribute"""
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return _lazy