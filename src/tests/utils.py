import contextlib
import os
from typing import Dict, Optional


@contextlib.contextmanager
def environment_variables(to_set: Dict[str, Optional[str]]):
    """
    Context manager to configure the os environ.

    After exiting the context, the original env vars will be back in place.

    Parameters
    ----------
    to_set:
        A dict from env var name to env var value. If the env var value is None, that will
        be treated as indicating that the env var should be unset within the managed
        context.
    """
    backup_of_changed_keys = {k: os.environ.get(k, None) for k in to_set.keys()}

    def update_environ_with(env_dict):
        for key, value in env_dict.items():
            if value is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = value

    update_environ_with(to_set)

    try:
        yield
    finally:
        update_environ_with(backup_of_changed_keys)
