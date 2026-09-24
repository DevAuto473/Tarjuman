"""
scripts/measure_rig.py - read the avatar's real proportions out of its GLB
==========================================================================
    npm run measurerig
    npm run measurerig -- tarjuman/public/some_other.glb

Prints the body-geometry block for `src/tarjuman_core/pose_to_bones.py`, in that
file's own units: shoulder width = 1.0, origin on the shoulder line, +y pointing
DOWN, and compares each value against what the file currently holds.

Why this exists
---------------
Those constants were measured by hand from `TarjumanRobot2.glb`. The rig was
later replaced by `last.glb` and the numbers stayed - describing a head twice
as wide, relative to its shoulders, as the one now on screen. The solver kept a
keep-out volume the size of a head that was not there, so every hand reaching
for the face was pushed away and down, and signs that touch the chin or the
forehead never arrived. Nothing failed; it just looked slightly wrong forever.

Hand-measured constants about a file that can be replaced will rot again. This
reads them back from whatever GLB is actually being served.
"""

import json
import os
import struct
import sys

import numpy as np

COMPONENT = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2),
             5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
COUNTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

DEFAULT_GLB = os.path.join("tarjuman", "public", "last.glb")


def load_glb(path):
    raw = open(path, "rb").read()
    if raw[:4] != b"glTF":
        raise SystemExit(f"[FAIL] Not a binary glTF: {path}")
    off, gltf, binary = 12, None, None
    while off < len(raw):
        length, kind = struct.unpack_from("<II", raw, off)
        off += 8
        chunk = raw[off:off + length]
        off += length
        if kind == 0x4E4F534A:
            gltf = json.loads(chunk.decode("utf-8"))
        elif kind == 0x004E4942:
            binary = chunk
    return gltf, binary


def node_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype=float).reshape(4, 4).T
    m = np.eye(4)
    scale = np.array(node.get("scale", [1, 1, 1]), dtype=float)
    x, y, z, w = np.array(node.get("rotation", [0, 0, 0, 1]), dtype=float)
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]])
    m[:3, :3] = rot * scale
    m[:3, 3] = np.array(node.get("translation", [0, 0, 0]), dtype=float)
    return m


class Rig:
    """World-space rest positions, and the mesh grouped by the bone that owns it."""

    def __init__(self, gltf, binary):
        self.g, self.bin = gltf, binary
        self.nodes = gltf["nodes"]
        self.by_name = {n["name"]: i for i, n in enumerate(self.nodes) if n.get("name")}
        self.parent = {}
        for i, n in enumerate(self.nodes):
            for c in n.get("children", []):
                self.parent[c] = i
        self._cache = {}

    def world(self, i):
        if i not in self._cache:
            m = node_matrix(self.nodes[i])
            if i in self.parent:
                m = self.world(self.parent[i]) @ m
            self._cache[i] = m
        return self._cache[i]

    def pos(self, name):
        i = self.by_name.get(name)
        return None if i is None else self.world(i)[:3, 3]

    def accessor(self, index):
        a = self.g["accessors"][index]
        view = self.g["bufferViews"][a["bufferView"]]
        fmt, size = COMPONENT[a["componentType"]]
        n = COUNTS[a["type"]]
        stride = view.get("byteStride") or size * n
        base = view.get("byteOffset", 0) + a.get("byteOffset", 0)
        dtype = np.float64 if fmt == "f" else np.int64
        out = np.empty((a["count"], n), dtype=dtype)
        for k in range(a["count"]):
            out[k] = struct.unpack_from("<" + fmt * n, self.bin, base + k * stride)
        return out

    def skinned_vertices(self):
        """(positions, owning bone name) for every skinned vertex, in world space."""
        chunks = []
        for i, node in enumerate(self.nodes):
            if "skin" not in node or "mesh" not in node:
                continue
            joints = [self.nodes[j].get("name") for j in self.g["skins"][node["skin"]]["joints"]]
            m = self.world(i)
            for prim in self.g["meshes"][node["mesh"]]["primitives"]:
                attrs = prim["attributes"]
                if "JOINTS_0" not in attrs or "WEIGHTS_0" not in attrs:
                    continue
                v = self.accessor(attrs["POSITION"])
                j = self.accessor(attrs["JOINTS_0"]).astype(int)
                w = self.accessor(attrs["WEIGHTS_0"])
                # The bone with the largest weight owns the vertex. Averaging the
                # bones instead would smear the head's outline into the neck.
                owner = j[np.arange(len(j)), w.argmax(axis=1)]
                chunks.append(((m[:3, :3] @ v.T).T + m[:3, 3],
                               np.array([joints[o] for o in owner])))
        if not chunks:
            raise SystemExit("[FAIL] No skinned mesh in this file.")
        return (np.vstack([c[0] for c in chunks]),
                np.concatenate([c[1] for c in chunks]))


HEAD_BONES = ("Head", "HeadTop", "Neck")
TORSO_BONES = ("Spine", "Chest", "Chest2", "Hips", "Shoulder.L", "Shoulder.R")

# name -> (measured value, the constant it feeds)
def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GLB
    if not os.path.isfile(path):
        print(f"[FAIL] No such file: {path}")
        return 1

    rig = Rig(*load_glb(path))
    need = ["UpperArm.L", "UpperArm.R", "LowerArm.L", "Hand.L", "middle.01.L", "middle.04.L"]
    missing = [b for b in need if rig.pos(b) is None]
    if missing:
        print(f"[FAIL] The rig is missing: {', '.join(missing)}")
        print("       This script expects the bone names pose_to_bones.py uses.")
        return 1

    left, right = rig.pos("UpperArm.L"), rig.pos("UpperArm.R")
    unit = abs(left[0] - right[0])          # shoulder width = 1.0
    mid = (left + right) / 2.0              # origin on the shoulder line

    def to_units(p):
        """glTF world -> this file's units: +y DOWN, shoulder width = 1."""
        return np.array([(p[0] - mid[0]), -(p[1] - mid[1]), (p[2] - mid[2])]) / unit

    upper = np.linalg.norm(rig.pos("LowerArm.L") - rig.pos("UpperArm.L"))
    fore = np.linalg.norm(rig.pos("Hand.L") - rig.pos("LowerArm.L"))
    wrist, knuckle, tip = (rig.pos("Hand.L"), rig.pos("middle.01.L"), rig.pos("middle.04.L"))
    palm = (wrist + knuckle) / 2.0

    verts, owner = rig.skinned_vertices()
    units = np.array([to_units(p) for p in verts])

    def extent(bones):
        m = np.isin(owner, bones)
        if not m.any():
            return None
        return units[m]

    head, torso = extent(HEAD_BONES), extent(TORSO_BONES)
    if head is None or torso is None:
        print("[FAIL] Could not find head or torso vertices by bone name.")
        return 1

    arm = (upper + fore) / unit
    values = {
        "RIG_ARM_SPAN":    arm,
        "UPPER_FRAC":      upper / (upper + fore),
        "LOWER_FRAC":      fore / (upper + fore),
        "HAND_REACH":      np.linalg.norm(tip - wrist) / unit,
        "BODY_CLEARANCE":  np.linalg.norm(tip - palm) / unit * 0.92,
        "TORSO_HALF_W":    max(abs(torso[:, 0].min()), abs(torso[:, 0].max())),
        "TORSO_HALF_D":    max(abs(torso[:, 2].min()), abs(torso[:, 2].max())),
        "TORSO_TOP":       torso[:, 1].min(),
        "TORSO_BOTTOM":    torso[:, 1].max(),
        "HEAD_CENTRE_Y":   (head[:, 1].min() + head[:, 1].max()) / 2.0,
        "HEAD_HALF_W":     max(abs(head[:, 0].min()), abs(head[:, 0].max())),
        "HEAD_HALF_H":     (head[:, 1].max() - head[:, 1].min()) / 2.0,
        "HEAD_HALF_D":     max(abs(head[:, 2].min()), abs(head[:, 2].max())),
    }

    print("=" * 68)
    print(f"  {os.path.basename(path)}")
    print("=" * 68)
    print(f"  shoulder width : {unit:.4f} m   (= 1.0 unit below)")
    print(f"  arm            : {upper + fore:.4f} m  upper {upper:.4f} + fore {fore:.4f}")
    print(f"  chin           : y = {head[:, 1].max():+.3f}   "
          f"(a sign that touches the chin must reach here)")

    try:
        sys.path.insert(0, "src")
        from tarjuman_core import pose_to_bones as ptb
    except Exception as exc:                                   # noqa: BLE001
        ptb = None
        print(f"\n  [!] could not import pose_to_bones to compare ({exc})")

    print(f"\n  {'constant':<18s} {'measured':>9s} {'in code':>9s}   ")
    print("  " + "-" * 46)
    drift = 0
    for name, measured in values.items():
        current = getattr(ptb, name, None) if ptb else None
        if current is None:
            print(f"  {name:<18s} {measured:>9.3f} {'-':>9s}")
            continue
        off = abs(measured - current)
        # 0.05 units is about a centimetre on a rig this size - below that the
        # difference is not visible, and chasing it would mean re-exporting for
        # measurement noise.
        mark = "   <- update" if off > 0.05 else ""
        drift += off > 0.05
        print(f"  {name:<18s} {measured:>9.3f} {current:>9.3f}{mark}")

    if ptb is None:
        # Saying "matches" here would be a lie: nothing was compared. This is
        # the same class of mistake the constants themselves fell into.
        print("\n  [!] Nothing was compared - pose_to_bones.py could not be")
        print("      imported. Run this through npm (`npm run measurerig`),")
        print("      which puts src/ on the path.")
        return 1
    if drift:
        print(f"\n  {drift} constant(s) disagree with the rig by more than 0.05.")
        print("  Copy the measured column into pose_to_bones.py, then re-run")
        print("  `npm run export3d` so the avatar's signs are rebuilt with them.")
        return 0
    print("\n  [OK] pose_to_bones.py matches this rig.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
