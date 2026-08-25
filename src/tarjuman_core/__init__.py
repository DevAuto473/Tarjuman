"""
tarjuman_core - the library the whole project shares
=====================================================
These modules are imported by the server, the trainer, the recorder and the
diagnostics alike. They live in a package rather than loose at the repo root so
that "what is shared code" and "what is a runnable script" is answerable by
looking at the directory tree.

`feature_extractor` in particular is the single source of truth for the feature
layout: the number of landmarks, how they are normalised, and the 4212 values a
sequence turns into. Recording, training and inference must agree on that layout
exactly, and the only way they can is by importing it from one place. A second
copy would not raise an error - it would quietly train a model on one geometry
and run it on another.
"""
