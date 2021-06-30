from typing import List, Union, Dict, Any, Tuple
from copy import deepcopy
import operator
from functools import reduce, partial
from dataclasses import dataclass
from collections import defaultdict
import itertools

import pdfplumber
import numpy as np
import layoutparser as lp
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from ..datamodel import PDFPage
from ..pdftools import PDFExtractor
from ..utils import *


def analyze_right_tokens(right_tokens: List) -> Tuple[List, List]:

    title_tokens = []
    regular_tokens = []
    for ele in right_tokens:
        if ele.text in ["represented", "by"]:
            continue
        if "bold" in ele.type.lower():
            title_tokens.append(ele)
        else:
            regular_tokens.append(ele)

    title_lines = non_maximal_supression(
        5,
        title_tokens,
        scoring_func=lambda x1, x2: abs(x1.coordinates[-1] - x2.coordinates[-1]),
        agg_func=lambda seq: seq,
        bigger_than=False,
    )

    return title_lines, regular_tokens


def split_page_tokens(boundary, page_tokens):

    width = page_tokens.width
    tokens = page_tokens.tokens

    start, end = boundary

    left_tokens = tokens.filter_by(lp.Rectangle(0, start, width / 2, end), center=True)

    right_tokens = tokens.filter_by(
        lp.Rectangle(width / 2, start, width, end), center=True
    )

    title_lines, regular_tokens = analyze_right_tokens(right_tokens)

    return left_tokens, right_tokens, title_lines, regular_tokens


@dataclass
class PlantiffBlock:
    title_tokens: lp.TextBlock
    name_tokens: lp.TextBlock
    representatives: List

    def get_info(self, role_mapping: Dict = None):
        block_type = " ".join(ele.text for ele in self.title_tokens)
        party_name = " ".join(ele.text for ele in self.name_tokens)
        if role_mapping is None:
            role_mapping = {}

        return {
            "block_type": role_mapping.get(block_type, block_type),
            "block_content": {
                party_name: [[ele.get_info()] for ele in self.representatives]
            },
        }

    @classmethod
    def from_page_tokens(cls, title_tokens, boundary, page_tokens):

        start, end = boundary

        left_tokens, right_tokens, title_lines, regular_tokens = split_page_tokens(
            boundary, page_tokens
        )

        representatives = []

        if len(title_lines) > 0:
            # Sometimes it can be empty on the right
            title_separators = [
                [a[0].coordinates[-1], b[0].coordinates[1]]
                for a, b in zip(title_lines, title_lines[1:])
            ] + [[title_lines[-1][0].coordinates[-1], end]]

            for rep_id in range(len(title_lines)):
                representatives.append(
                    RepresentativeBlock(
                        title_tokens=title_lines[rep_id],
                        info_tokens=right_tokens.filter_by(
                            lp.Interval(*title_separators[rep_id], axis="y")
                        ),
                    )
                )

        return cls(
            title_tokens=title_tokens,
            name_tokens=left_tokens,
            representatives=representatives,
        )

    def include_next_page_token(self, boundary, page_tokens):

        start, end = boundary

        left_tokens, right_tokens, title_lines, regular_tokens = split_page_tokens(
            boundary, page_tokens
        )

        representatives = []
        if len(title_lines) > 0:

            title_separators = [
                [a[0].coordinates[-1], b[0].coordinates[1]]
                for a, b in zip(title_lines, title_lines[1:])
            ] + [[title_lines[-1][0].coordinates[-1], end]]

            for rep_id in range(len(title_lines)):
                representatives.append(
                    RepresentativeBlock(
                        title_tokens=title_lines[rep_id],
                        info_tokens=right_tokens.filter_by(
                            lp.Interval(*title_separators[rep_id], axis="y")
                        ),
                    )
                )

            pre_title_separator = [0, title_lines[0][0].coordinates[1]]
            remaining_tokens = right_tokens.filter_by(
                lp.Interval(*pre_title_separator, axis="y")
            )

        else:
            remaining_tokens = regular_tokens

        if self.representatives:
            # Somtimes it might be empty
            self.representatives[-1].info_tokens.extend(remaining_tokens)
        self.representatives.extend(representatives)


@dataclass
class RepresentativeBlock:
    title_tokens: List[lp.TextBlock]
    info_tokens: List[lp.TextBlock]

    def get_info(self):
        name = " ".join([ele.text for ele in self.title_tokens])
        info = " ".join(
            [tok.text for tok in self.info_tokens if "italic" not in tok.type.lower()]
        )
        feature = " ".join(
            [tok.text for tok in self.info_tokens if "italic" in tok.type.lower()]
        )
        return "\n".join([name, info, feature])


class PageStructureParser:

    HEADER_RATIO = 0.035

    FOOTER_RATIO = 1 - 0.035

    FRONT_PAGE_HEADER_RATIO = 0.2
    # For the front page, the "header" of the document before the possible plantiff

    TOKEN_UNDERLINE_DISTANCE_MAX = 10
    # Used for match token with possible underlines

    VALID_UNDERLINED_BOLD_TOKEN_LENGTH_MIN = 5

    def __init__(self, pdf_extractor=None):

        if pdf_extractor is None:
            self.pdf_extractor = PDFExtractor("pdfplumber")
        else:
            self.pdf_extractor = pdf_extractor

    def remove_header_and_footer_tokens(self, pdf_tokens: List) -> List:

        new_pdf_tokens = []
        for pdf_token in pdf_tokens:

            height = pdf_token.height
            new_pdf_tokens.append(
                PDFPage(
                    height = pdf_token.height,
                    width = pdf_token.width,
                    tokens = pdf_token.tokens.filter_by(
                        lp.Interval(
                            height * self.HEADER_RATIO, height * self.FOOTER_RATIO, axis="y"
                        )
                    ),
                    lines = pdf_token.lines.filter_by(
                        lp.Interval(
                            height * self.HEADER_RATIO, height * self.FOOTER_RATIO, axis="y"
                        )
                    ),
                )
            )
        return new_pdf_tokens

    def find_bold_tokens(self, pdf_tokens: List) -> List:

        all_bold_tokens = []
        for page_idx in range(len(pdf_tokens)):
            tokens = pdf_tokens[page_idx].tokens
            height = pdf_tokens[page_idx].height
            if page_idx == 0:
                tokens = tokens.filter_by(
                    lp.Interval(height * self.FRONT_PAGE_HEADER_RATIO, height, axis="y")
                )
            bold_tokens = [ele for ele in tokens if "bold" in ele.type.lower()]
            all_bold_tokens.append(bold_tokens)

        return all_bold_tokens

    def find_matched_token_underlines(
        self, bold_tokens: List, possible_underlines: List
    ) -> List[Tuple]:

        assert len(bold_tokens) >= 1

        token_centers = np.array([ele.block.center for ele in bold_tokens])
        underline_centers = np.array([ele.center for ele in possible_underlines])
        token_underline_dist = cdist(
            token_centers, underline_centers, metric="cityblock"
        )
        tids, uids = np.where(token_underline_dist < self.TOKEN_UNDERLINE_DISTANCE_MAX)

        matched = []
        for tid, uid in zip(tids, uids):
            matched.append([bold_tokens[tid], possible_underlines[uid]])

        return matched

    def create_plantiff_structure(
        self,
        pdf_tokens: List,
        all_underlined_bold_tokens: List,
        table_regions: Dict = None,
    ) -> Dict[str, List[PlantiffBlock]]:

        all_plantiffs_blocks = {}

        for idx, underlined_bold_tokens in all_underlined_bold_tokens.items():

            # separators = [ele.coordinates[-1] for ele in underlined_bold_tokens]
            if len(underlined_bold_tokens) < 1:
                continue

            separators = [
                [a.coordinates[-1], b.coordinates[1]]
                for a, b in zip(underlined_bold_tokens, underlined_bold_tokens[1:])
            ]

            if table_regions[idx]:
                separators.append(
                    [
                        underlined_bold_tokens[-1].coordinates[-1],
                        table_regions[idx][0].boundary.coordinates[1],
                    ]
                )
            else:
                separators.append(
                    [
                        underlined_bold_tokens[-1].coordinates[-1],
                        pdf_tokens[idx].height,
                    ]
                )

            # width = pdf_tokens[idx]["width"]
            # tokens = pdf_tokens[idx]["tokens"]

            page_plantiff_blocks = []

            if idx >= 1 and idx - 1 in all_plantiffs_blocks:
                pre_separator = [0, separators[0][0]]
                all_plantiffs_blocks[idx - 1][-1].include_next_page_token(
                    boundary=pre_separator, page_tokens=pdf_tokens[idx]
                )

            for plantiff_block_id in range(len(separators)):

                if separators[plantiff_block_id][0] > separators[plantiff_block_id][1]:
                    print(f"Weird separator distribution {separators}")
                    continue

                page_plantiff_blocks.append(
                    PlantiffBlock.from_page_tokens(
                        title_tokens=[underlined_bold_tokens[plantiff_block_id]],
                        boundary=separators[plantiff_block_id],
                        page_tokens=pdf_tokens[idx],
                    )
                )

            all_plantiffs_blocks[idx] = page_plantiff_blocks

        return all_plantiffs_blocks

    def parse_page_structure_from_pdf(self, filename, table_regions=None):

        pdf_tokens = self.pdf_extractor.pdf_extractor.extract(filename)
        return self.parse_page_structure(pdf_tokens, table_regions)

    def parse_page_structure(self, pdf_tokens, table_regions=None):

        pdf_tokens = self.remove_header_and_footer_tokens(pdf_tokens)

        if table_regions is None:
            table_regions = {}

        all_possible_underlines = [page.lines for page in pdf_tokens]
        all_bold_tokens = self.find_bold_tokens(pdf_tokens)

        all_underlined_bold_tokens = {}

        for idx in range(len(pdf_tokens)):

            bold_tokens = all_bold_tokens[idx]
            if len(bold_tokens) < 1:
                # Skip for pages without bold tokens
                continue

            possible_underlines = all_possible_underlines[idx]
            if len(possible_underlines) < 1:
                # Skip for pages without underlines
                continue

            matched_token_underlines = self.find_matched_token_underlines(
                bold_tokens, possible_underlines
            )

            if len(matched_token_underlines) > 0:
                all_underlined_bold_tokens[idx] = [
                    ele[0]
                    for ele in matched_token_underlines
                    if len(ele[0].text) >= self.VALID_UNDERLINED_BOLD_TOKEN_LENGTH_MIN
                ]

        all_plantiffs_blocks = self.create_plantiff_structure(
            pdf_tokens, all_underlined_bold_tokens, table_regions
        )

        return all_plantiffs_blocks

    def fetch_case_flags(self, pdf_tokens, upper_rule_ratio=0.045, bottom_rule_ratio=0.07):
        """A simple rule-based method for fetching the case flags. 
        Right now it selects tokens inside a manually specified "band" on the first page, and 
        treat them as the case_flags. 
        """
        pdf_page = pdf_tokens[0] # Only check the first page
        height = pdf_page.height

        case_flag_region = lp.Interval(int(height*upper_rule_ratio),int(height*bottom_rule_ratio), axis='y')
        case_flag_tokens = pdf_page.tokens.filter_by(case_flag_region, center=True)
        case_flags = [token.text.strip().rstrip(", ") for token in case_flag_tokens]

        case_flags = list(itertools.chain.from_iterable([case_flag.split(",") for case_flag in case_flags]))
        # Sometimes the pdf-parser does not split the text based at ",", and we have to manually do it

        return case_flags
        