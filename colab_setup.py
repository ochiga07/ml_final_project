# thin wrapper so notebooks can do: from colab_setup import setup_project
import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_spec = importlib.util.spec_from_file_location(
    '_src_colab_setup',
    os.path.join(_SRC, 'colab_setup.py'),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

setup_project = _mod.setup_project
mount_drive = _mod.mount_drive
clone_or_pull_repo = _mod.clone_or_pull_repo
sync_repo_to_drive = _mod.sync_repo_to_drive
