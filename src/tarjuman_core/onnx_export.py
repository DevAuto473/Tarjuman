"""
onnx_export.py - build the ONNX graph directly, without skl2onnx
================================================================
Converts a `Pipeline(StandardScaler, RandomForestClassifier)` into an ONNX file
using nothing but `onnx` and `numpy`.

Why not just use skl2onnx
------------------------
On the development machine `import skl2onnx` terminates the interpreter with
0xC0000005 (access violation) - not an exception, a hard crash inside a native
library, which no amount of error handling in Python can survive. Training could
not reach its export step at all, and there is no way to repair another project's
wheel from inside this one.

`onnx` and `onnxruntime` both work fine there, and the graph in question is
small: a Scaler followed by a TreeEnsembleClassifier, both standard ai.onnx.ml
operators. Writing those two nodes directly removes the broken dependency
entirely, which also means one less package to install on the Raspberry Pi.

Correctness is not taken on trust: `train_model.py` compares this file's output
against scikit-learn's own predict_proba on the held-out test set and refuses
the model if they disagree.

How a RandomForest maps onto TreeEnsembleClassifier
---------------------------------------------------
The operator holds every tree of the forest in flat, parallel arrays - one entry
per node, tagged with which tree it belongs to. Leaves contribute class weights,
which the operator SUMS. scikit-learn AVERAGES its trees instead, so every leaf
weight is divided by the number of trees up front; summing them then reproduces
the mean exactly.
"""

import numpy as np


def _tree_arrays(tree, tree_id: int, n_trees: int, n_classes: int):
    """Flatten one decision tree into TreeEnsembleClassifier arrays."""
    t = tree.tree_
    nodes = {
        "treeids": [], "nodeids": [], "featureids": [], "modes": [],
        "values": [], "truenodeids": [], "falsenodeids": [], "missing": [],
    }
    leaves = {"treeids": [], "nodeids": [], "ids": [], "weights": []}

    for i in range(t.node_count):
        left = int(t.children_left[i])
        right = int(t.children_right[i])
        is_leaf = left == -1

        nodes["treeids"].append(tree_id)
        nodes["nodeids"].append(i)
        nodes["missing"].append(0)

        if is_leaf:
            # A leaf still needs an entry, but it branches nowhere.
            nodes["featureids"].append(0)
            nodes["modes"].append("LEAF")
            nodes["values"].append(0.0)
            nodes["truenodeids"].append(0)
            nodes["falsenodeids"].append(0)

            # sklearn stores class COUNTS at a leaf; a tree's own predict_proba
            # normalises them, and the forest then averages across trees.
            counts = np.asarray(t.value[i][0], dtype=np.float64)
            total = counts.sum()
            proba = counts / total if total > 0 else np.zeros_like(counts)
            for c in range(n_classes):
                leaves["treeids"].append(tree_id)
                leaves["nodeids"].append(i)
                leaves["ids"].append(c)
                leaves["weights"].append(float(proba[c] / n_trees))
        else:
            # sklearn sends X[feature] <= threshold to the LEFT child.
            nodes["featureids"].append(int(t.feature[i]))
            nodes["modes"].append("BRANCH_LEQ")
            nodes["values"].append(float(t.threshold[i]))
            nodes["truenodeids"].append(left)
            nodes["falsenodeids"].append(right)

    return nodes, leaves


def export_pipeline(pipeline, n_features: int, input_name: str = "input",
                    prob_name: str = "probabilities") -> bytes:
    """
    Serialise Pipeline(StandardScaler, RandomForestClassifier) to ONNX bytes.

    The batch dimension is left dynamic so the same file serves single-frame
    inference on the server and batched evaluation offline.
    """
    from onnx import TensorProto, helper

    scaler = pipeline.named_steps.get("scaler")
    clf = pipeline.named_steps.get("clf")
    if clf is None:
        raise ValueError("pipeline has no 'clf' step")

    classes = np.asarray(clf.classes_)
    n_classes = len(classes)
    estimators = clf.estimators_
    n_trees = len(estimators)

    graph_nodes = []
    cursor = input_name

    if scaler is not None:
        # ai.onnx.ml Scaler computes (x - offset) * scale, whereas
        # StandardScaler computes (x - mean_) / scale_. Inverting the scale
        # here is what makes the two agree.
        offset = (np.asarray(scaler.mean_, dtype=np.float32)
                  if getattr(scaler, "with_mean", True) and scaler.mean_ is not None
                  else np.zeros(n_features, dtype=np.float32))
        if getattr(scaler, "with_std", True) and scaler.scale_ is not None:
            scale = (1.0 / np.asarray(scaler.scale_, dtype=np.float64)).astype(np.float32)
        else:
            scale = np.ones(n_features, dtype=np.float32)

        graph_nodes.append(helper.make_node(
            "Scaler", [input_name], ["scaled"], domain="ai.onnx.ml",
            offset=offset.tolist(), scale=scale.tolist(),
        ))
        cursor = "scaled"

    # ── Flatten every tree into one set of parallel arrays ───────────────────
    agg = {k: [] for k in ("treeids", "nodeids", "featureids", "modes",
                           "values", "truenodeids", "falsenodeids", "missing")}
    agg_leaf = {k: [] for k in ("treeids", "nodeids", "ids", "weights")}

    for tid, est in enumerate(estimators):
        nodes, leaves = _tree_arrays(est, tid, n_trees, n_classes)
        for k in agg:
            agg[k].extend(nodes[k])
        for k in agg_leaf:
            agg_leaf[k].extend(leaves[k])

    # Class labels are the LabelEncoder's integer ids; the app maps them back
    # to words through labels.json, exactly as it did with skl2onnx output.
    class_labels = [int(c) for c in classes]

    graph_nodes.append(helper.make_node(
        "TreeEnsembleClassifier",
        [cursor], ["label", prob_name], domain="ai.onnx.ml",
        classlabels_int64s=class_labels,
        nodes_treeids=agg["treeids"],
        nodes_nodeids=agg["nodeids"],
        nodes_featureids=agg["featureids"],
        nodes_modes=agg["modes"],
        nodes_values=agg["values"],
        nodes_truenodeids=agg["truenodeids"],
        nodes_falsenodeids=agg["falsenodeids"],
        nodes_missing_value_tracks_true=agg["missing"],
        class_treeids=agg_leaf["treeids"],
        class_nodeids=agg_leaf["nodeids"],
        class_ids=agg_leaf["ids"],
        class_weights=agg_leaf["weights"],
        # NONE because the weights already are probabilities. Any transform
        # here would quietly distort the confidence the app thresholds on.
        post_transform="NONE",
    ))

    graph = helper.make_graph(
        graph_nodes, "tarjuman_sign_classifier",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT,
                                       [None, n_features])],
        [helper.make_tensor_value_info("label", TensorProto.INT64, [None]),
         helper.make_tensor_value_info(prob_name, TensorProto.FLOAT,
                                       [None, n_classes])],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13),
                       helper.make_opsetid("ai.onnx.ml", 2)],
        producer_name="tarjuman",
    )
    model.ir_version = 8      # accepted by every onnxruntime the project uses
    return model.SerializeToString()
