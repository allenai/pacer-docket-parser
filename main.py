from typing import List, Union, Dict, Dict, Any, Tuple
from functools import reduce
from collections import defaultdict
from dataclasses import dataclass
import operator

import layoutparser as lp
import cv2
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


def construct_filter(df, column, value, eps=0.1):
    return df[column].between(value - eps, value + eps)


def obtain_line_id(df, filter, diff_th=6, rho_min=5):
    lines = df[filter].copy().sort_values(by="rho")
    lines["line_id"] = (lines["rho"].diff(1) > diff_th).cumsum()
    lines = lines[lines["rho"] > rho_min]
    return lines.groupby("line_id")[["rho", "theta"]].median()


def is_valid_table(table_image):
    height, width = table_image.shape[:2]

    table_image_grey = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
    _, table_image_bin = cv2.threshold(table_image_grey, 210, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(table_image_bin, 50, 30)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, int(height * 0.75), None, 0, 0)
    if lines is None or len(lines) == 0:
        return False

    return True


def detect_vertical_lines(table_image):

    height, width = table_image.shape[:2]

    table_image_grey = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
    _, table_image_bin = cv2.threshold(table_image_grey, 210, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(table_image_bin, 50, 30)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, int(height * 0.75), None, 0, 0)
    if lines is None or len(lines) == 0:
        return None
    line_table = pd.DataFrame(np.squeeze(lines), columns=["rho", "theta"]).sort_values(
        by="theta"
    )

    line_table.loc[
        (line_table["rho"] < 0) | (line_table["theta"] > np.pi / 2), "theta"
    ] -= np.pi
    line_table.loc[line_table["rho"] < 0, "rho"] *= -1

    vertical_filter = construct_filter(line_table, "theta", 0)
    vertical_lines = obtain_line_id(line_table, vertical_filter)
    return vertical_lines


@dataclass
class Table:
    boundary: lp.elements.BaseLayoutElement
    tokens: List[lp.TextBlock]
    columns: List[lp.elements.BaseLayoutElement]
    rows: List[lp.elements.BaseLayoutElement]
    grid: List[List[lp.TextBlock]]

    @classmethod
    def from_columns_and_rows(cls, table, tokens, columns, rows):
        grid = [[intersect(row, column) for column in columns] for row in rows]
        for row in grid:
            for block in row:
                block.text = " ".join(tokens.filter_by(block, center=True).get_texts())
        return cls(table, tokens, columns, rows, grid)

    def to_dataframe(self):
        return pd.DataFrame([[block.text for block in row] for row in self.grid])


class TableDetector:

    TABLE_TYPE_NAME = "table"

    MIN_TABLE_WIDTH = 350
    """The minimum valid table width, used for dropping irregular table detections"""
    MIN_TABLE_OVERLAPPING = 0.25
    """The minimum overlapping index for two tables to merge them"""
    MIN_DOCKET_TABLE_WIDTH = 420
    """Threshold for descriminating main docket table or receipt table."""
    MIN_ROW_HEIGHT = 5
    """The minimum row_height, used for separating different rows"""
    ROW_START_SHIFT = 6.5
    """The distance between the row text top to the row header"""
    HEADER_HEIGHT = 25
    """The height of headers, used for trimming table proposal detections"""
    FOOTER_HEIGHT = 15
    """The height of headers, used for trimming table proposal detections"""

    GAP_BETWEEN_TOP_ROW_AND_TABLE_BOUNDARY = 15
    """If top_row.y_1 - table.y_1 is larger than this value, add an additional row"""

    def __init__(self, model, pdf_extractor, config=None):
        self.model = model
        self.pdf_extractor = pdf_extractor
        self.config = config

    def detect_table_region_proposals(
        self, pdf_images: List
    ) -> Dict[str, List[lp.TextBlock]]:
        """Detect possible table regions using DL models
         and process the detected regions based on self.post_process_table_region_proposals

        Args:
            pdf_images (List):
                A list of PDF page images

        Returns:
            Dict[str, List[lp.TextBlock]]:

                {
                    "page_id" : [ A list of table table proposals on this page ]
                }
        """
        all_table_region_proposals = {}
        pbar = tqdm(range(len(pdf_images)))
        for idx in pbar:
            pbar.set_description(f"Working on page {idx}")

            res = self.model.detect(pdf_images[idx])
            detected_tables = [e for e in res if e.type == self.TABLE_TYPE_NAME]
            all_table_region_proposals[idx] = self.post_process_table_region_proposals(
                detected_tables, pdf_images[idx]
            )

        return all_table_region_proposals

    def post_process_table_region_proposals(
        self, table_region_proposals: List, pdf_image
    ) -> List:
        """Filter any unreasonable table detection results, includeing:

            1. Extremely narrow blocks
            2. Union block detection results for the same table
                (determined based on their overlapping levels)
            3. Trim tables if they appear within the
        Args:
            detected_tables (List):
                A list of table regions for each page

        Returns:
            List: A list of processed table regions
        """

        if isinstance(pdf_image, np.ndarray):
            canvas_height, canvas_width = pdf_image.shape[:2]
        else:
            canvas_width, canvas_height = pdf_image.size

        table_region_proposals = [
            e
            for e in table_region_proposals
            if e.width >= self.MIN_TABLE_WIDTH
            and is_valid_table(e.crop_image(pdf_image))
        ]

        # When less than 2 blocks appear in the detected tables, there
        # could not be overlapping blocks

        for table_region_proposal in table_region_proposals:
            table_region_proposal.block.y_1 = max(
                table_region_proposal.block.y_1, self.HEADER_HEIGHT
            )
            table_region_proposal.block.y_2 = min(
                table_region_proposal.block.y_2, canvas_height - self.FOOTER_HEIGHT
            )

        if len(table_region_proposals) <= 1:
            return table_region_proposals

        table_region_proposals = non_maximal_supression(
            self.MIN_TABLE_OVERLAPPING,
            table_region_proposals,
            overlapping_coefficient,
            lambda blocks: reduce(union, blocks),
        )

        return table_region_proposals

    def identify_table_columns(
        self, table: lp.TextBlock, table_image: "Image"
    ) -> List[lp.Rectangle]:
        """A rule-based methods for identifying table columns:
            It uses line detection methods to search for vertical_lines
            inside the table region image.

        Args:
            table (lp.TextBlock): The table block

        Returns:
            List[lp.Rectangle]:
                A list of rectangles for each column, ordered from right to left.
        """
        if table.width <= self.MIN_DOCKET_TABLE_WIDTH:
            # If it's a receipt table, we don't generate any columns
            return []
        else:
            vertical_lines = detect_vertical_lines(table_image)
            if vertical_lines is None:
                return None
            else:
                x_1, y_1, x_2, y_2 = table.coordinates
                col_1_end = vertical_lines.iloc[0, 0]
                col_2_end = vertical_lines.iloc[1, 0]
                return lp.Layout(
                    [
                        lp.Interval(0, col_1_end, axis="x"),
                        lp.Interval(col_1_end, col_2_end, axis="x"),
                        lp.Interval(col_2_end, table.width, axis="x"),
                    ]
                ).condition_on(table)

    def identify_table_rows(
        self,
        table: lp.TextBlock,
        columns: List[lp.elements.BaseLayoutElement],
        table_tokens: List[lp.TextBlock],
    ) -> List[lp.Rectangle]:
        """A rule-based methods for identifying table rows:
        It identifies row_start based on y_1 of tokens inside the
        left-most columns.

        Args:
            table (lp.TextBlock): The table block
            columns (List[lp.elements.BaseLayoutElement]):
                Table columns detected by self.identify_table_columns
            table_tokens (List[lp.TextBlock]):
                All the tokens within the table

        Returns:
            List[lp.Rectangle]:
                A list of rectangles for each row, ordered from top to bottom
        """
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

        if row_starts[0] - y_1 > self.GAP_BETWEEN_TOP_ROW_AND_TABLE_BOUNDARY:
            row_starts.insert(0, y_1)

        for row_start, row_end in zip(row_starts, row_starts[1:] + [y_2]):
            rows.append(lp.Rectangle(x_1, row_start, x_2, row_end))

        return rows

    def detect_table_from_pdf_page(
        self,
        page_tokens: List[lp.TextBlock],
        page_image: Union["Image", np.ndarray],
        table_region_proposals: List[lp.TextBlock] = None,
    ) -> List[Table]:
        """Detects the tables for a given PDF page

        Args:
            page_tokens (List[lp.TextBlock]):
                All the tokens within the given page.
            page_image (Union[PIL.Image, np.ndarray]):
                The image of the given page.
            table_region_proposals (List[lp.TextBlock], optional):
                The table regions proposals of this page detected by
                self.detect_table_region_proposals.
                If not set, it will automatically call detect_table_region_proposals
                to detect the table region proposals.

        Returns:
            List[Table]:
                A list of `Table`s.
        """
        if not isinstance(page_image, np.ndarray):
            page_image = np.array(page_image)

        if table_region_proposals is None:
            table_region_proposals = self.detect_table_region_proposals([page_image])[0]

        tables = []
        for table in table_region_proposals:
            table_tokens = page_tokens.filter_by(table, center=True)
            table_image = np.array(table.crop_image(page_image))
            table = union_blocks(table_tokens)
            # Slightly rectify the table region based on the contained tokens

            columns = self.identify_table_columns(table, table_image)
            if columns is None:  # This is not a valid table, drop it.
                continue

            rows = self.identify_table_rows(table, columns, table_tokens)
            table = Table.from_columns_and_rows(table, table_tokens, columns, rows)
            tables.append(table)
        return tables

    def detect_tables_from_pdf(self, pdf_filename: str) -> Dict[str, List[Table]]:
        """Detect tables for all pages from the given pdf_filename

        Args:
            pdf_filename (str):
                The pdf filename.

        Returns:
            Dict[str, List[Table]]:
                {page_index: all possible tables on this page}
        """
        pdf_page_tokens, pdf_images = self.pdf_extractor.load_tokens_and_image(
            pdf_filename, resize_image=True
        )

        table_region_proposals = self.detect_table_region_proposals(pdf_images)

        all_tables = {}
        for idx, tables in table_region_proposals.items():
            page_tokens = pdf_page_tokens[idx]["tokens"]
            page_image = pdf_images[idx]

            all_tables[idx] = self.detect_table_from_pdf_page(
                page_tokens, page_image, tables
            )

        return all_tables