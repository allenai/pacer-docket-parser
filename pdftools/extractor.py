from pdf2image import convert_from_path

from .grobid import GrobidTokenExtractor
from .pdfplumber import PDFPlumberTokenExtractor


class PDFExtractor:
    """PDF Extractor will load both images and layouts for PDF documents for downstream processing."""

    def __init__(self, pdf_extractor_name, **kwargs):

        self.pdf_extractor_name = pdf_extractor_name.lower()

        if self.pdf_extractor_name == GrobidTokenExtractor.NAME:
            self.pdf_extractor = GrobidTokenExtractor(**kwargs)
        elif self.pdf_extractor_name == PDFPlumberTokenExtractor.NAME:
            self.pdf_extractor = PDFPlumberTokenExtractor(**kwargs)
        else:
            raise NotImplementedError(
                f"Unknown pdf_extractor_name {pdf_extractor_name}"
            )

        self.use_lp = True

    def load_tokens_and_image(
        self, pdf_path: str, resize_image=False, resize_layout=False, **kwargs
    ):

        pdf_layouts = self.pdf_extractor(pdf_path, **kwargs)

        page_images = convert_from_path(pdf_path)

        assert not (
            resize_image and resize_layout
        ), "You could not resize image and layout simultaneously."

        if self.use_lp:
            if resize_layout:
                for image, page in zip(page_images, pdf_layouts):
                    width, height = image.size
                    resize_factor = width / page["width"], height / page["height"]
                    page["layout"] = page["layout"].scale(resize_factor)
                    page["image_height"] = height
                    page["image_width"] = width

            elif resize_image:
                page_images = [
                    image.resize((int(page["width"]), int(page["height"])))
                    for image, page in zip(page_images, pdf_layouts)
                ]
        else:

            if resize_layout:
                for image, page in zip(page_images, pdf_layouts):
                    width, height = image.size
                    resize_factor = width / page.page.width, height / page.page.height
                    page.tokens.scale(resize_factor)

            elif resize_image:
                page_images = [
                    image.resize((int(page.page.width), int(page.page.width)))
                    for image, page in zip(page_images, pdf_layouts)
                ]

        return pdf_layouts, page_images
