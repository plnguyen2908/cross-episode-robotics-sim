"""Filesystem locations used by the simulator.

Source-controlled assets and configs live inside the package. Downloaded data
(qualified grasp registry, starting scenes, converted objects) and run outputs
live outside it, under directories that can be overridden with environment
variables:

- ``CES_DATA_DIR``: installed data bundle (default: ``<repo>/data``)
- ``CES_RUNS_DIR``: episode outputs (default: ``<repo>/runs``)
"""

import hashlib
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent
ASSETS_DIR = PACKAGE_DIR / "assets"
FIXTURE_ASSETS_DIR = ASSETS_DIR / "fixtures"
CONFIG_DIR = PACKAGE_DIR / "configs"

DATA_DIR = Path(os.environ.get("CES_DATA_DIR", REPO_DIR / "data")).resolve()
RUNS_DIR = Path(os.environ.get("CES_RUNS_DIR", REPO_DIR / "runs")).resolve()

# Grasp-qualification sweep: per-object starting scenes and qualified grasps.
GRASP_SWEEP_DIR = DATA_DIR / "grasp_sweep"
# MolmoSpaces objects converted to standalone MJCF with TidyBot grasp registries.
MOLMO_OBJECTS_DIR = DATA_DIR / "molmo_objects"
# Saved starting scenes for the atomic-skill demonstrations.
SKILL_SCENES_DIR = DATA_DIR / "skill_scenes"
# Espresso machine converted from MoonlakeAI/sim-env-builder.
COFFEE_DIR = DATA_DIR / "coffee"
# Robot model files generated for RoboSuite at runtime.
GENERATED_DIR = DATA_DIR / "generated"


def source_digests(*modules):
    """SHA-256 of the given modules' source files, keyed by package-relative path."""
    digests = {}
    for module in modules:
        path = Path(module.__file__).resolve()
        digests[str(path.relative_to(PACKAGE_DIR))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


# Asset files store machine-specific roots as ${TOKEN}s. `localize` resolves
# them for this machine; `portable` turns local absolute paths back into tokens.
def robocasa_dir():
    configured = os.environ.get("ROBOCASA_DIR")
    if configured:
        return Path(configured)
    from importlib.util import find_spec

    spec = find_spec("robocasa")
    if spec is None or spec.origin is None:
        raise RuntimeError("Set ROBOCASA_DIR to the installed robocasa package directory")
    return Path(spec.origin).parent


def mlspaces_assets_dir():
    configured = os.environ.get("MLSPACES_ASSETS_DIR")
    if configured:
        return Path(configured)
    from molmo_spaces.molmo_spaces_constants import ASSETS_DIR as MLSPACES_ASSETS

    return Path(MLSPACES_ASSETS)


def _mlspaces_cache_dir():
    return Path(os.environ.get("MLSPACES_CACHE_DIR", Path.home() / ".cache/molmo-spaces-resources"))


def token_roots():
    return {
        "${CES_ASSETS}": ASSETS_DIR,
        "${CES_DATA}": DATA_DIR,
        # Development evidence cited by path in registries and reports; not shipped.
        "${CES_EVIDENCE}": DATA_DIR / "evidence",
        "${MLSPACES_CACHE_DIR}": _mlspaces_cache_dir(),
        "${MLSPACES_ASSETS_DIR}": mlspaces_assets_dir(),
        "${ROBOCASA_DIR}": robocasa_dir(),
    }


def localize(text):
    for token, root in token_roots().items():
        if token in text:
            text = text.replace(token, str(root))
    return text


def portable(text, local_roots):
    """Replace each local absolute root with its token. `local_roots` maps root -> token."""
    for root, token in sorted(local_roots.items(), key=lambda item: -len(str(item[0]))):
        text = text.replace(str(root), token)
    return text


def read_localized(path):
    return localize(Path(path).read_text())


def local_roots():
    """Inverse of `token_roots` for this machine, used to make text portable again."""
    return {root: token for token, root in token_roots().items()}


def model_digest(path):
    """SHA-256 of an asset file with machine-specific roots replaced by tokens.

    Grasp registries key their evidence by this digest, so a registry stays
    valid when the data bundle is installed under a different directory.
    """
    text = portable(Path(path).read_text(), local_roots())
    return hashlib.sha256(text.encode()).hexdigest()


GRASP_REGISTRY_NAME = "grasp_annotations_franka_tidybot.json"
# Registries for RoboCasa's downloaded objects are kept here, mirroring the
# object's path inside the robocasa package, so the RoboCasa install stays untouched.
ROBOCASA_GRASPS_DIR = DATA_DIR / "robocasa_grasps"


def grasp_registry_path(model_path):
    """Where the TidyBot grasp registry for an object model lives."""
    model_path = Path(model_path).resolve()
    try:
        relative = model_path.parent.relative_to(robocasa_dir().resolve())
    except (RuntimeError, ValueError):
        return model_path.parent / GRASP_REGISTRY_NAME
    return ROBOCASA_GRASPS_DIR / relative / GRASP_REGISTRY_NAME
