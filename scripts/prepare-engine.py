"""Bundle the active Python runtime and installed CPU worker for Windows packaging.

Use backend/.venv/Scripts/python.exe to run this script. Source files stay present
because PyTorch's exporter inspects Python source at runtime.
"""
from pathlib import Path
import shutil
import sys

root = Path(__file__).resolve().parents[1]
destination = root / "desktop" / "engine"
base = Path(sys.base_prefix)
site_packages = Path(sys.prefix) / "Lib" / "site-packages"
if sys.platform != "win32" or sys.prefix == sys.base_prefix:
    raise SystemExit("Run this with the Windows backend virtual environment Python.")
destination.mkdir(parents=True, exist_ok=True)
for name in ("python.exe", "python3.dll", f"python{sys.version_info.major}{sys.version_info.minor}.dll", "vcruntime140.dll", "vcruntime140_1.dll", "LICENSE.txt"):
    source = base / name
    if source.exists():
        shutil.copy2(source, destination / name)
shutil.copytree(base / "DLLs", destination / "DLLs", dirs_exist_ok=True)
def ignore_lib(directory, names):
    return [name for name in names if name == "__pycache__" or (Path(directory) == base / "Lib" and name == "site-packages")]
shutil.copytree(base / "Lib", destination / "Lib", dirs_exist_ok=True, ignore=ignore_lib)
shutil.copytree(site_packages, destination / "Lib" / "site-packages", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
shutil.copytree(root / "backend" / "app", destination / "app", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
shutil.copy2(root / "backend" / "desktop_entry.py", destination / "desktop_entry.py")
shutil.copy2(root / "backend" / "cli.py", destination / "cli.py")
shutil.copy2(root / "backend" / "check_install.py", destination / "check_install.py")
shutil.copy2(root / "backend" / "requirements-windows-tested.txt", destination / "requirements-windows-tested.txt")
print(f"Portable Python engine prepared in {destination}")
