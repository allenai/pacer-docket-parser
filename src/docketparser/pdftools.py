from typing import List, Union, Dict, Any, Tuple
from abc import ABC, abstractmethod

import pandas as pd
import pdfplumber
import layoutparser as lp
import pdf2image

from .datamodel import *


def convert_token_dict_to_layout(tokens):
    return lp.Layout(
        [
            lp.TextBlock(
                lp.Rectangle(
                    x_1=token["x"],
                    y_1=token["y"],
                    x_2=token["x"] + token["width"],
                    y_2=token["y"] + token["height"],
                ),
                text=token["text"],
                type=token.get("type"),
            )
            for token in tokens
        ]
    )


def load_page_data_from_dict(source_data: Dict[str, Any]) -> List[Dict]:

    return [
        PDFPage(
            height=page_data["page"]["height"],
            width=page_data["page"]["width"],
            tokens=convert_token_dict_to_layout(page_data["tokens"]),
            url_tokens=convert_token_dict_to_layout(page_data["url_tokens"]),
            lines=[
                lp.Rectangle(
                    x_1=line["x"],
                    y_1=line["y"],
                    x_2=line["x"] + line["width"],
                    y_2=line["y"] + line["height"],
                ) for line in page_data["lines"]
            ]
        )
        for page_data in source_data
    ]


class BasePDFTokenExtractor(ABC):
    """PDF token extractors will load all the *tokens* and save using pdfstructure service."""

    def __call__(self, pdf_path: str):
        return self.extract(pdf_path)

    @abstractmethod
    def extract(self, pdf_path: str):
        """Extract PDF Tokens from the input pdf_path

        Args:
            pdf_path (str):
                The path to a PDF file

        Returns:
        """
        pass


class PDFPlumberTokenExtractor(BasePDFTokenExtractor):
    NAME = "pdfplumber"

    UNDERLINE_HEIGHT_THRESHOLD = 3
    UNDERLINE_WIDTH_THRESHOLD = 100
    # Defines what a regular underline should look like

    @staticmethod
    def convert_to_pagetoken(row: pd.Series) -> Dict:
        """Convert a row in a DataFrame to pagetoken"""
        return dict(
            text=row["text"],
            x=row["x0"],
            width=row["width"],
            y=row["top"],
            height=row["height"],
            type=row.get("fontname"),
        )

    def obtain_word_tokens(self, cur_page: pdfplumber.page.Page) -> List[Dict]:
        """Obtain all words from the current page.
        Args:
            cur_page (pdfplumber.page.Page):
                the pdfplumber.page.Page object with PDF token information
        Returns:
            List[PageToken]:
                A list of page tokens stored in PageToken format.
        """
        words = cur_page.extract_words(
            x_tolerance=1.5,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True,
            horizontal_ltr=True,
            vertical_ttb=True,
            extra_attrs=["fontname", "size"],
        )

        if len(words) == 0:
            return []

        df = pd.DataFrame(words)

        # Avoid boxes outside the page
        df[["x0", "x1"]] = (
            df[["x0", "x1"]].clip(lower=0, upper=int(cur_page.width)).astype("float")
        )
        df[["top", "bottom"]] = (
            df[["top", "bottom"]]
            .clip(lower=0, upper=int(cur_page.height))
            .astype("float")
        )

        df["height"] = df["bottom"] - df["top"]
        df["width"] = df["x1"] - df["x0"]

        word_tokens = df.apply(self.convert_to_pagetoken, axis=1).tolist()
        return word_tokens

    def obtain_page_hyperlinks(self, cur_page: pdfplumber.page.Page) -> List[Dict]:

        if len(cur_page.hyperlinks) == 0:
            return []

        df = pd.DataFrame(cur_page.hyperlinks)
        df[["x0", "x1"]] = (
            df[["x0", "x1"]].clip(lower=0, upper=int(cur_page.width)).astype("float")
        )
        df[["top", "bottom"]] = (
            df[["top", "bottom"]]
            .clip(lower=0, upper=int(cur_page.height))
            .astype("float")
        )
        df[["height", "width"]] = df[["height", "width"]].astype("float")

        hyperlink_tokens = (
            df.rename(columns={"uri": "text"})
            .apply(self.convert_to_pagetoken, axis=1)
            .tolist()
        )
        return hyperlink_tokens

    def obtain_page_lines(self, cur_page: pdfplumber.page.Page) -> List[Dict]:

        height = float(cur_page.height)
        page_objs = cur_page.rects + cur_page.lines
        possible_underlines = [
            dict(
                x = float(ele["x0"]),
                y = height - float(ele["y0"]),
                height = float(ele['height']),
                width = float(ele['width']),
            )
            for ele in filter(
                lambda obj: obj["height"] < self.UNDERLINE_HEIGHT_THRESHOLD
                and obj["width"] < self.UNDERLINE_WIDTH_THRESHOLD,
                page_objs,
            )
        ]
        return possible_underlines

    def extract(self, pdf_path: str) -> List[Dict]:
        """Extracts token text, positions, and style information from a PDF file.
        Args:
            pdf_path (str): the path to the pdf file.
            include_lines (bool, optional): Whether to include line tokens. Defaults to False.
            target_data (str, optional): {"token", "hyperlink"}
        Returns:
            PdfAnnotations: A `PdfAnnotations` containing all the paper token information.
        """
        plumber_pdf_object = pdfplumber.open(pdf_path)

        pages = []
        for page_id in range(len(plumber_pdf_object.pages)):
            cur_page = plumber_pdf_object.pages[page_id]

            tokens = self.obtain_word_tokens(cur_page)
            url_tokens = self.obtain_page_hyperlinks(cur_page)
            lines = self.obtain_page_lines(cur_page)

            page = dict(
                page=dict(
                    width=float(cur_page.width),
                    height=float(cur_page.height),
                    index=page_id,
                ),
                tokens=tokens,
                url_tokens=url_tokens,
                lines=lines,
            )
            pages.append(page)

        return load_page_data_from_dict(pages)


class PDFExtractor:
    """PDF Extractor will load both images and layouts for PDF documents for downstream processing."""

    def __init__(self, pdf_extractor_name, **kwargs):

        self.pdf_extractor_name = pdf_extractor_name.lower()

        if self.pdf_extractor_name == PDFPlumberTokenExtractor.NAME:
            self.pdf_extractor = PDFPlumberTokenExtractor(**kwargs)
        else:
            raise NotImplementedError(
                f"Unknown pdf_extractor_name {pdf_extractor_name}"
            )

    def load_tokens_and_image(
        self, pdf_path: str, resize_image=False, resize_layout=False, **kwargs
    ):

        pdf_tokens = self.pdf_extractor(pdf_path, **kwargs)

        page_images = pdf2image.convert_from_path(pdf_path, dpi=72)

        assert not (
            resize_image and resize_layout
        ), "You could not resize image and layout simultaneously."

        if resize_layout:
            for image, page in zip(page_images, pdf_tokens):
                width, height = image.size
                resize_factor = width / page.width, height / page.height
                page.tokens = page.tokens.scale(resize_factor)
                page.image_height = height
                page.image_width = width

        elif resize_image:
            page_images = [
                image.resize((int(page.width), int(page.height)))
                if page.width != image.size[0]
                else image
                for image, page in zip(page_images, pdf_tokens)
            ]

        return pdf_tokens, page_images