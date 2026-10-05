"""Build the portable data bundle from a development workspace (maintainers only).

The bundle holds everything the tasks read but the source tree does not:
qualified-grasp starting scenes, prepared base episodes, converted objects, the
converted espresso machine, atomic-skill starting scenes and TidyBot grasp
registries. Machine-specific absolute paths are replaced by ${TOKEN}s, which
`scripts/install_data.py` resolves on the user's machine. Grasp registries are
re-keyed to the path-independent model digest (`paths.model_digest`).

Example:
    python scripts/build_data_bundle.py \
        --workspace /path/to/research/cross_episode_memory \
        --robocasa /path/to/robocasa/robocasa --output dist/data
"""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

TEXT_SUFFIXES = {".xml", ".json", ".txt", ".md", ".jsonl"}
RUN_OUTPUTS = {"trace.json", "trace_trimmed.json", "run.log", "grip_force_trace.jsonl"}
RUN_SUFFIXES = {".mp4", ".png", ".log"}
REGISTRY = "grasp_annotations_franka_tidybot.json"
# Tokens that `install_data.py` resolves, in addition to those in cross_episode_sim.paths.
ABSOLUTE = re.compile(r"(?<![\w$}])/(?:nobackup2?|afs|home|Users|tmp)/[^\s\"'<>]*")


def bundle_layout(workspace, robocasa):
    art = workspace / "artifacts"
    sweep = art / "shared_curobo_all_objects"
    copies = [
        (art / "breakfast_20261001_001443", "base_episodes/breakfast_three_room", "run"),
        (art / "coffee_box_tuck_20261004", "base_episodes/coffee_counter", "run"),
        (art / "moonlake_coffee_trial_20261004/asset_v3", "coffee/espresso_machine", "asset"),
        (workspace / "robocasa_robot/molmo_assets", "molmo_objects", "all"),
        (art / "toaster_bread_initial", "skill_scenes/toaster_bread_initial", "all"),
        (art / "thin_book_initial", "skill_scenes/thin_book_initial", "all"),
        (art / "molmo_robocasa_10/Egg_1_retry/trial_000", "skill_scenes/egg_retry", "run"),
        (art / "coffee_box_prepare_20261004", "base_episodes/coffee_prepare", "run"),
    ]
    for name in ("coffee_mug_grasps.npz", "grounds_cup_grasps.npz", "lid_handle_hypotheses.npz"):
        copies.append((art / "coffee_tune_20261003_v1" / name, f"base_episodes/coffee_tune/{name}", "all"))
    for scene in sorted((art / "saved_grasp_reuse").glob("*/initial_scene")):
        copies.append((scene, f"skill_scenes/saved_grasp_reuse/{scene.parent.name}/initial_scene", "all"))
    copies.append((sweep / "current_results.json", "grasp_sweep/current_results.json", "all"))
    for status in sorted((sweep / "objects").glob("*/status.json")):
        key = status.parent.name
        copies.append((status, f"grasp_sweep/objects/{key}/status.json", "all"))
        if (status.parent / "initial_scene/setup.json").is_file():
            copies.append((status.parent / "initial_scene", f"grasp_sweep/objects/{key}/initial_scene", "all"))
    for registry in sorted(robocasa.rglob(REGISTRY)):
        relative = registry.relative_to(robocasa)
        copies.append((registry, f"robocasa_grasps/{relative}", "all"))
    roots = {
        art / "breakfast_20261001_001443": "${CES_DATA}/base_episodes/breakfast_three_room",
        art / "coffee_box_tuck_20261004": "${CES_DATA}/base_episodes/coffee_counter",
        art / "coffee_box_prepare_20261004": "${CES_DATA}/base_episodes/coffee_prepare",
        art / "coffee_tune_20261003_v1": "${CES_DATA}/base_episodes/coffee_tune",
        art / "moonlake_coffee_trial_20261004/asset_v3": "${CES_DATA}/coffee/espresso_machine",
        art / "toaster_bread_initial": "${CES_DATA}/skill_scenes/toaster_bread_initial",
        art / "thin_book_initial": "${CES_DATA}/skill_scenes/thin_book_initial",
        art / "molmo_robocasa_10/Egg_1_retry/trial_000": "${CES_DATA}/skill_scenes/egg_retry",
        art / "saved_grasp_reuse": "${CES_DATA}/skill_scenes/saved_grasp_reuse",
        sweep: "${CES_DATA}/grasp_sweep",
        workspace / "robocasa_robot/molmo_assets": "${CES_DATA}/molmo_objects",
        workspace / "robocasa_robot/generated": "${CES_ASSETS}/fixtures",
        workspace / "assets": "${CES_ASSETS}",
        robocasa: "${ROBOCASA_DIR}",
        # Development-run evidence (trial reports, source snapshots) is cited by
        # path in registries and reports but not shipped.
        art: "${CES_EVIDENCE}",
        workspace: "${CES_EVIDENCE}/source",
    }
    return copies, roots


def skip(path, mode):
    if mode == "run":
        return path.name in RUN_OUTPUTS or path.suffix in RUN_SUFFIXES
    if mode == "asset":
        return path.suffix == ".usdz"
    return False


def tokenize(text, roots):
    for root, token in sorted(roots.items(), key=lambda item: -len(str(item[0]))):
        text = text.replace(str(root), token)
    return text


def copy_tree(source, destination, mode, roots):
    files = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
    for path in files:
        if skip(path, mode):
            continue
        target = destination if source.is_file() else destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in TEXT_SUFFIXES:
            target.write_text(tokenize(path.read_text(errors="strict"), roots))
        else:
            shutil.copyfile(path, target)


def rekey(registry, model, roots):
    """Replace raw-byte model digests with the digest of the tokenized model text."""
    payload = json.loads(registry.read_text())
    changed = 0
    if model.is_file():
        raw = hashlib.sha256(model.read_bytes()).hexdigest()
        portable = hashlib.sha256(tokenize(model.read_text(), roots).encode()).hexdigest()
        for grasp in payload["grasps"]:
            if grasp.get("model_sha256") == raw:
                grasp["model_sha256"] = portable
                changed += 1
    registry.write_text(json.dumps(payload, indent=2))
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, required=True, help="research/cross_episode_memory directory")
    parser.add_argument("--robocasa", type=Path, required=True, help="robocasa package directory")
    parser.add_argument("--extra-root", action="append", default=[], metavar="PATH=TOKEN",
                        help="Additional absolute root to tokenize, e.g. ~/.cache/molmo-spaces-resources=${MLSPACES_CACHE_DIR}")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace, robocasa = args.workspace.resolve(), args.robocasa.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"{output} already exists")
    copies, roots = bundle_layout(workspace, robocasa)
    for item in args.extra_root:
        path, token = item.split("=", 1)
        roots[Path(path).expanduser()] = token
    for source, relative, mode in copies:
        copy_tree(source, output / relative, mode, roots)
    changed = 0
    for registry in (output / "molmo_objects").rglob(REGISTRY):
        model = workspace / "robocasa_robot/molmo_assets" / registry.parent.name / "model.xml"
        changed += rekey(registry, model, roots)
    for registry in (output / "robocasa_grasps").rglob(REGISTRY):
        model = robocasa / registry.parent.relative_to(output / "robocasa_grasps") / "model.xml"
        changed += rekey(registry, model, roots)
    leftovers = {}
    for path in output.rglob("*"):
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            for match in ABSOLUTE.findall(path.read_text()):
                prefix = "/".join(match.split("/")[:6])
                leftovers.setdefault(prefix, set()).add(str(path.relative_to(output)))
    files = sum(1 for p in output.rglob("*") if p.is_file())
    print(json.dumps(dict(files=files, registry_entries_rekeyed=changed,
                          leftover_absolute_roots={k: sorted(v)[:3] for k, v in leftovers.items()}), indent=2))
    if leftovers:
        raise SystemExit("Absolute paths remain; add --extra-root mappings and rebuild")


if __name__ == "__main__":
    main()
