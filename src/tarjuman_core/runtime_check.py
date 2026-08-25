"""
runtime_check.py - turn dependency mismatches into readable instructions
=======================================================================

The protobuf/mediapipe mismatch surfaced as this, three frames deep inside
a vendored library:

    AttributeError: 'FieldDescriptor' object has no attribute 'label'

Nothing in that names protobuf, names a version, or suggests a fix, and it
appears only when a solution graph is CONSTRUCTED - so it slipped past both
a plain `import mediapipe` and `npm run doctor`.

This module checks the one pairing that is known to break, and does it with
a plain version comparison: no imports beyond metadata, no measurable
startup cost, safe to call from every entry point.
"""

import importlib.metadata as _md

# protobuf 5 removed FieldDescriptor.label and MessageFactory.GetPrototype.
# mediapipe 0.10.x calls both while building a solution graph, so anything
# from 5.0 upwards fails - not at import, but on Hands()/Pose().
_PROTOBUF_MAX_MAJOR = 4


def _major(version: str) -> int:
    try:
        return int(version.split(".", 1)[0])
    except (ValueError, AttributeError):
        return -1


def check_mediapipe_stack(*, fatal: bool = True) -> bool:
    """
    Verify protobuf is old enough for mediapipe. Returns True when usable.

    With fatal=True (the default) an incompatible pair raises SystemExit with
    the exact command that fixes it, which is far more useful at a terminal
    than a traceback pointing into solution_base.py.
    """
    try:
        protobuf_version = _md.version("protobuf")
    except Exception:
        return True          # not installed yet; let the real import complain

    if _major(protobuf_version) <= _PROTOBUF_MAX_MAJOR:
        return True

    message = (
        "\n" + "=" * 68 + "\n"
        f"  INCOMPATIBLE DEPENDENCY: protobuf {protobuf_version}\n"
        + "=" * 68 + "\n"
        "  MediaPipe cannot build its hand/pose graphs with protobuf 5 or\n"
        "  newer - the APIs it calls (FieldDescriptor.label and\n"
        "  MessageFactory.GetPrototype) were removed in protobuf 5.\n"
        "\n"
        "  Note this does NOT break `import mediapipe`, which is why it can\n"
        "  pass a health check and still fail the moment a camera starts.\n"
        "\n"
        "  Fix it with:\n"
        "      npm run fix\n"
        + "=" * 68
    )

    if fatal:
        raise SystemExit(message)
    print(message)
    return False
