###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Compare the bare-metal install scripts with the release Dockerfile they mirror.

`tools/installation/setup.sh` and `tools/installation-jax/setup.sh` exist to
reproduce the published images without a container, and each declares which
Dockerfile it was derived from. Keeping them aligned is the most tedious part of
a release, and drift is invisible until someone's bare-metal install behaves
differently from the image.

The comparison is self-anchoring: the release is read from the script's own
`(from Dockerfile.primus-vX.Y)` header, so `--check` is meaningful without being
told which release is current.

With --baremetal-manifest it additionally diffs an actual install against the
image snapshot. `stage_manifest` writes a `pip list` in the same format the image
ships, which turns "is bare metal aligned with the Dockerfile" into a diff of two
package sets rather than a judgement call.

Nothing runs this automatically -- it is a Phase 3 gate in the release-docs skill.

Stdlib-only, so it needs no virtualenv wherever it is run.

Usage:
    python tools/release_docs/install_parity.py
    python tools/release_docs/install_parity.py --check
    python tools/release_docs/install_parity.py --family primus \\
        --baremetal-manifest /scratch/primus/.manifest/requirements.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402

RULES_FILE = Path(__file__).resolve().parent / "install_parity_rules.json"

SETUP_SCRIPTS = {
    "primus": C.ROOT / "tools/installation/setup.sh",
    "jax": C.ROOT / "tools/installation-jax/setup.sh",
}

DERIVED_FROM = re.compile(r"Dockerfile\.(?:primus|jax)-(v[\d.]+?)(?:\)|\s|$|\")")
ARG_ENV = re.compile(r"^(?:ARG|ENV)\s+([A-Za-z_][A-Za-z0-9_]*)=(\S+)", re.MULTILINE)
SH_ASSIGN = re.compile(r"^([A-Z][A-Z0-9_]*)=(.+)$", re.MULTILINE)
SH_DEFAULT = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*:-(.*)\}$")

# A package pinned with '==', whether the version is a literal or a variable.
# Both sides are scanned for names only: what matters is whether the script
# installs the same set of things, not how it spells the version.
PINNED_PKG = re.compile(
    r"""["']?([A-Za-z][A-Za-z0-9._-]{2,})==(?:\$\{?[A-Za-z_][A-Za-z0-9_]*\}?|[0-9][^\s\\"']*)"""
)
# Per-arch wheels are built from PYTORCH_ROCM_ARCH at run time, so the script
# never spells them literally.
ARCH_TEMPLATED = re.compile(r"-gfx\d+[a-z]*$")
INDEX_URL = re.compile(r"--(?:extra-)?index-url[= ]\s*(\S+)")


def load_rules():
    return json.loads(RULES_FILE.read_text())


def dockerfile_pins(path):
    """ARG/ENV defaults, first declaration winning (later stages redeclare)."""
    pins = {}
    for match in ARG_ENV.finditer(path.read_text()):
        pins.setdefault(match.group(1), match.group(2).strip("\"'"))
    return pins


def shell_pins(path):
    """Top-level variable assignments, resolving `${VAR:-default}` to its default."""
    pins = {}
    for match in SH_ASSIGN.finditer(path.read_text()):
        name, raw = match.group(1), match.group(2)
        value = raw.split("#", 1)[0].strip().strip("\"'")
        default = SH_DEFAULT.match(value)
        if default:
            value = default.group(1).strip("\"'")
        pins.setdefault(name, value)
    return pins


def inline_pins(path, patterns):
    """Pins the Dockerfile states inside RUN lines rather than as ARG."""
    text = path.read_text()
    found = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            found[name] = match.group(1).strip("\"'")
    return found


def mirrored_release(family):
    """Which Dockerfile the script says it mirrors."""
    match = DERIVED_FROM.search(SETUP_SCRIPTS[family].read_text())
    return match.group(1) if match else None


def pinned_package_names(text):
    """Canonical names of everything pinned with '==', arch wheels excluded."""
    names = set()
    for raw in PINNED_PKG.findall(text):
        name = C.canon(raw)
        if not ARCH_TEMPLATED.search(name):
            names.add(name)
    return names


def index_urls(text):
    """Every --index-url / --extra-index-url, normalised to compare cleanly.

    Dockerfiles wrap RUN lines, so a URL can arrive with the line-continuation
    backslash attached.
    """
    return {url.strip().strip("\"'").rstrip("\\").rstrip("/") for url in INDEX_URL.findall(text)}


def missing_packages(dockerfile, script, allowed):
    """Packages the Dockerfile pins that the script never mentions.

    This is the check that catches a changed *package set* rather than a changed
    version -- the thing variable-name comparison structurally cannot see. In
    v26.7 Transformer Engine went from two distributions to three
    (transformer_engine, transformer_engine_rocm10) and the JAX plugin pair was
    renamed jax_rocm10_*; none of that moved a shared variable, so the release
    shipped scripts that silently installed the wrong set.
    """
    gap = pinned_package_names(dockerfile) - pinned_package_names(script)
    return sorted(name for name in gap if name not in allowed)


def stale_indexes(dockerfile, script):
    """Index URLs the script uses that the Dockerfile no longer mentions.

    Index URLs live inside RUN lines rather than ARG declarations, so they used to
    land in the informational 'unmapped' bucket. v26.7 moved every wheel index to
    stable.repo.amd.com and parity still reported clean.
    """
    docker_urls = index_urls(dockerfile)
    if not docker_urls:
        return []
    stale = []
    for name, value in sorted(script.items()):
        if not re.search(r"INDEX", name) or not value.startswith("http"):
            continue
        if value.rstrip("/") not in docker_urls:
            stale.append((name, value, sorted(docker_urls)))
    return stale


def compare(family, release, rules):
    dockerfile = C.dockerfile_for(family, release)
    script = SETUP_SCRIPTS[family]
    if not dockerfile.exists():
        return None, f"{dockerfile.relative_to(C.ROOT)} does not exist"

    dockerfile_text = dockerfile.read_text()
    script_text = script.read_text()
    docker = dockerfile_pins(dockerfile)
    docker.update(inline_pins(dockerfile, rules["inline_pins"].get(family, {})))
    shell = shell_pins(script)

    aliases = rules["aliases"].get(family, {})
    not_pins = set(rules["not_pins"])
    documented = rules["documented_divergences"].get(family, {})

    matched, drift, divergent, unmapped = [], [], [], []
    for sh_name, sh_value in sorted(shell.items()):
        if sh_name in not_pins:
            continue
        docker_name = sh_name
        if sh_name not in docker:
            docker_name = next((dk for dk, sk in aliases.items() if sk == sh_name), sh_name)
        if docker_name not in docker:
            if sh_name in documented:
                divergent.append((sh_name, sh_value, documented[sh_name]))
            elif re.search(r"VERSION|COMMIT|BRANCH|INDEX|REPO", sh_name):
                unmapped.append((sh_name, sh_value))
            continue
        docker_value = docker[docker_name]
        entry = (sh_name, docker_name, docker_value, sh_value)
        if docker_value == sh_value:
            matched.append(entry)
        elif sh_name in documented:
            divergent.append((sh_name, f"{docker_value} -> {sh_value}", documented[sh_name]))
        else:
            drift.append(entry)

    allowed = set(rules.get("dockerfile_only_packages", {}).get(family, {}))
    return {
        "family": family,
        "release": release,
        "dockerfile": dockerfile.relative_to(C.ROOT).as_posix(),
        "script": script.relative_to(C.ROOT).as_posix(),
        "matched": matched,
        "drift": drift,
        "documented_divergences": divergent,
        "unmapped": unmapped,
        "missing_packages": missing_packages(dockerfile_text, script_text, allowed),
        "stale_indexes": stale_indexes(dockerfile_text, shell),
    }, None


def compare_manifest(family, release, manifest_path):
    """Diff an actual bare-metal install against the image snapshot."""
    snapshot = C.load_snapshot(release, family)
    if snapshot is None:
        return None, f"no snapshot for {release}-{family}; probe the image first"
    installed = C.parse_pip_list(Path(manifest_path).read_text())
    image = snapshot["pip"]

    differing = sorted(
        (name, image[name], installed[name])
        for name in set(image) & set(installed)
        if image[name] != installed[name]
    )
    return {
        "only_in_image": sorted(set(image) - set(installed)),
        "only_on_metal": sorted(set(installed) - set(image)),
        "differing": differing,
        "common": len(set(image) & set(installed)),
    }, None


def report(result):
    print(f"install parity: {result['family']} vs {result['release']}")
    print(f"  {result['script']}")
    print(f"  {result['dockerfile']}")
    print(f"  matched                {len(result['matched'])}")
    print(f"  drift                  {len(result['drift'])}")
    print(f"  missing packages       {len(result['missing_packages'])}")
    print(f"  stale indexes          {len(result['stale_indexes'])}")
    print(f"  documented divergence  {len(result['documented_divergences'])}")
    print(f"  unmapped               {len(result['unmapped'])}")

    if result["missing_packages"]:
        print("\n  MISSING PACKAGES (the Dockerfile pins these, the script never mentions them):")
        for name in result["missing_packages"]:
            print(f"    {name}")
        print("    A changed package set does not move any shared variable, so nothing else")
        print("    catches it. Install them, or add them to dockerfile_only_packages with a reason.")
    if result["stale_indexes"]:
        print("\n  STALE INDEXES (the script points at an index this release no longer uses):")
        for name, value, docker_urls in result["stale_indexes"]:
            print(f"    {name} = {value}")
            for url in docker_urls:
                print(f"      dockerfile uses: {url}")

    if result["drift"]:
        print("\n  DRIFT (the script does not match the Dockerfile it mirrors):")
        for sh_name, docker_name, docker_value, sh_value in result["drift"]:
            via = "" if docker_name == sh_name else f" (Dockerfile: {docker_name})"
            print(f"    {sh_name}{via}")
            print(f"      dockerfile  {docker_value}")
            print(f"      setup.sh    {sh_value}")
    if result["documented_divergences"]:
        print("\n  DOCUMENTED DIVERGENCE (deliberate):")
        for name, value, why in result["documented_divergences"]:
            print(f"    {name} = {value}")
            print(f"      {why}")
    if result["unmapped"]:
        print("\n  UNMAPPED (pin-shaped script variables with no Dockerfile counterpart):")
        for name, value in result["unmapped"]:
            print(f"    {name} = {value}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--family", choices=sorted(SETUP_SCRIPTS), help="default: both")
    parser.add_argument(
        "--release", help="Dockerfile release to compare against (default: what the script says)"
    )
    parser.add_argument("--check", action="store_true", help="exit non-zero on undocumented drift")
    parser.add_argument("--baremetal-manifest", help="a bare-metal .manifest/requirements.txt to diff")
    args = parser.parse_args()

    # A manifest belongs to one install, so without --family there is nothing to
    # compare it against. Silently ignoring it would report a clean run that never
    # looked at the manifest at all.
    if args.baremetal_manifest and not args.family:
        parser.error("--baremetal-manifest requires --family (a manifest describes one install)")

    rules = load_rules()
    families = [args.family] if args.family else sorted(SETUP_SCRIPTS)
    total_drift = 0

    for family in families:
        release = args.release or mirrored_release(family)
        if not release:
            print(f"ERROR: {SETUP_SCRIPTS[family]} does not name a Dockerfile; pass --release.")
            return 1
        result, error = compare(family, release, rules)
        if error:
            print(f"ERROR: {error}")
            return 1
        report(result)
        total_drift += len(result["drift"]) + len(result["missing_packages"]) + len(result["stale_indexes"])

        if args.baremetal_manifest and args.family == family:
            diff, error = compare_manifest(family, release, args.baremetal_manifest)
            if error:
                print(f"  ERROR: {error}")
                return 1
            print(f"\n  bare-metal manifest vs image snapshot ({diff['common']} packages in common)")
            print(f"    differing versions   {len(diff['differing'])}")
            print(f"    only in the image    {len(diff['only_in_image'])}")
            print(f"    only on bare metal   {len(diff['only_on_metal'])}")
            for name, image_version, metal_version in diff["differing"][:40]:
                print(f"      {name:32} image={image_version:28} metal={metal_version}")
        print()

    if args.check and total_drift:
        print(f"FAILED: {total_drift} undocumented pin drift(s) between setup.sh and its release Dockerfile.")
        return 1
    if args.check:
        print("OK: install scripts match the Dockerfiles they mirror.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
