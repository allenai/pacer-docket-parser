from functools import reduce
import operator
import json

import numpy as np
import layoutparser as lp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def load_json_record(filepath):

    with open(filepath, "r") as fp:
        res = json.load(fp)
    return res

def save_json_record(data, filepath):

    with open(filepath, "w") as fp:
        json.dump(data, fp)

def union(block1, block2):
    x11, y11, x12, y12 = block1.coordinates
    x21, y21, x22, y22 = block2.coordinates

    block = lp.Rectangle(min(x11, x21), min(y11, y21), max(x12, x22), max(y12, y22))
    if isinstance(block1, lp.TextBlock):
        return lp.TextBlock(
            block,
            id=block1.id,
            type=block1.type,
            text=block1.text,
            parent=block1.parent,
            next=block1.next,
        )
    else:
        return block


def union_blocks(blocks):
    return reduce(union, blocks)


def intersect(block1, block2):
    x11, y11, x12, y12 = block1.coordinates
    x21, y21, x22, y22 = block2.coordinates

    x1, y1, x2, y2 = max(x11, x21), max(y11, y21), min(x12, x22), min(y12, y22)
    if x1 > x2 or y1 > y2:
        return None
    block = lp.Rectangle(x1, y1, x2, y2)
    if isinstance(block1, lp.TextBlock):
        return lp.TextBlock(
            block,
            id=block1.id,
            type=block1.type,
            text=block1.text,
            parent=block1.parent,
            next=block1.next,
        )
    else:
        return block


def overlapping_coefficient(block1, block2):

    i = intersect(block1, block2)
    if i is not None:
        o_area = max(block1.area, block2.area)
        return i.area / o_area
    else:
        return 0


def non_maximal_supression(
    threshold,
    sequence,
    scoring_func,
    agg_func,
    bigger_than=True,
):

    op = operator.gt if bigger_than else operator.lt

    length = len(sequence)
    graph = np.zeros((length, length))
    any_connected_components = False

    for i in range(length):
        for j in range(i + 1, length):
            if op(scoring_func(sequence[i], sequence[j]), threshold):
                graph[i][j] = 1
                any_connected_components = True

    if not any_connected_components:
        return sequence

    graph = csr_matrix(graph)
    n_components, labels = connected_components(
        csgraph=graph, directed=False, return_labels=True
    )

    grouped_sequence = []
    for comp_idx in range(n_components):
        element_idx = np.where(labels == comp_idx)[0]
        if len(element_idx) == 1:
            grouped_sequence.append(sequence[element_idx[0]])
        else:
            grouped_sequence.append(agg_func([sequence[i] for i in element_idx]))
    return grouped_sequence
