from typing import List, Union, Dict, Dict, Any, Tuple
from functools import reduce
import operator

import layoutparser as lp
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


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


def to_dataframe(grid):
    return pd.DataFrame([[block.text for block in row] for row in grid])


class TableDetector:

    TABLE_TYPE_NAME = "table"

    MIN_TABLE_WIDTH = 200
    """The minimum valid table width, used for dropping irregular table detections"""
    MIN_TABLE_OVERLAPPING = 0.25
    """The minimum overlapping index for two tables to merge them"""
    MIN_DOCKET_TABLE_WIDTH = 490
    """Threshold for descriminating main docket table or receipt table."""
    MIN_ROW_HEIGHT = 5
    """The minimum row_height, used for separating different rows"""
    ROW_START_SHIFT = 6.5
    """The distance between the row text top to the row header"""

    def __init__(self, model, config=None):
        self.model = model
        self.config = config

    def detect_table_regions(self, pdf_images: List) -> Dict[str, List[lp.TextBlock]]:
        all_tables = {}
        pbar = tqdm(range(len(pdf_images)))
        for idx in pbar:
            pbar.set_description(f"Working on page {idx}")
            res = self.model.detect(pdf_images[idx])
            detected_tables = [e for e in res if e.type == self.TABLE_TYPE_NAME]
            all_tables[idx] = self.post_processing_tables(detected_tables)

        return all_tables

    def post_processing_tables(self, detected_tables: List) -> List:
        """Filter any unreasonable table detection results, includeing:

            1. Extremely narrow blocks
            2. Union block detection results for the same table
                (determined based on their overlapping levels)

        Args:
            detected_tables (List):
                A list of table regions for each page

        Returns:
            List: A list of processed table regions
        """
        detected_tables = [
            e for e in detected_tables if e.width >= self.MIN_TABLE_WIDTH
        ]

        # When less than 2 blocks appear in the detected tables, there
        # could not be overlapping blocks
        if len(detected_tables) <= 1:
            return detected_tables

        return non_maximal_supression(
            self.MIN_TABLE_OVERLAPPING,
            detected_tables,
            overlapping_coefficient,
            lambda blocks: reduce(union, blocks),
        )

    def identify_table_columns(
        self, table: lp.TextBlock, table_tokens: List[lp.TextBlock]
    ) -> List[lp.Rectangle]:
        """A rule-based methods for identifying table columns:
            It separates the docket table into three columns
            based on the column widths.

        Args:
            table (lp.TextBlock): The table block

        Returns:
            List[lp.Rectangle]:
                A list of rectangles indicating for each column
        """
        if table.width <= self.MIN_DOCKET_TABLE_WIDTH:
            # If it's a receipt table, we don't generate any columns
            return [table.block]
        else:
            # Some simple rules for determining column boundaries
            x_1, y_1, x_2, y_2 = table.coordinates

    def identify_table_rows(self, table, columns, table_tokens):

        x_1, y_1, x_2, y_2 = table.coordinates

        column_one_tokens = table_tokens.filter_by(columns[0], center=True)
        row_coordinate_candidates = [tok.coordinates[1] for tok in column_one_tokens]

        row_starts = non_maximal_supression(
            5,
            sorted(row_coordinate_candidates),
            lambda x1, x2: x2 - x1,
            np.mean,
            bigger_than=False,
        )

        row_starts = [ele - self.ROW_START_SHIFT for ele in row_starts]
        rows = []

        for row_start, row_end in zip(row_starts, row_starts[1:] + [y_2]):
            rows.append(lp.Rectangle(x_1, row_start, x_2, row_end))

        return rows

    def create_table_grid(self, table, table_tokens, filter_text=True):

        columns = self.identify_table_columns(table, table_tokens)
        rows = self.identify_table_rows(table, columns, table_tokens)
        grid = [[intersect(row, column) for column in columns] for row in rows]
        if filter_text:
            for row in grid:
                for block in row:
                    block.text = " ".join(
                        table_tokens.filter_by(block, center=True).get_texts()
                    )
        return grid

    # def parse_table_structure(self, table, tokens):
