from collections import defaultdict

from ..pdftools import PDFExtractor
from . import *


class MainParser:
    def __init__(
        self, pdf_extractor=None, table_parser=None, page_parser=None, text_parser=None
    ):

        pdf_extractor = (
            PDFExtractor("pdfplumber") if pdf_extractor is None else pdf_extractor
        )
        self.table_parser = (
            TableParser(pdf_extractor=pdf_extractor)
            if table_parser is None
            else table_parser
        )
        self.page_parser = (
            PageStructureParser(pdf_extractor=pdf_extractor)
            if page_parser is None
            else page_parser
        )
        self.text_parser = (
            RawTextParser(pdf_extractor=pdf_extractor)
            if text_parser is None
            else text_parser
        )
        self.pdf_extractor = pdf_extractor

    @staticmethod
    def collect_genererated_table_data(tables, pdf_tokens):
        all_table_df = []
        for page_id, table in tables.items():
            if len(table) == 0:
                continue

            if len(table) > 1:
                print(f"Warning: page {page_id} has {len(table)} tables)")
            table = table[0]  # we assume there's only one table per page
            table_df = table.to_dataframe()

            hyperlinks = pdf_tokens[page_id].url_tokens

            if len(hyperlinks) == 0:
                table_df[3] = None
            else:
                all_possible_links = [None] * len(table.rows)
                for idx, rows in enumerate(table.grid):
                    col = rows[1]
                    all_possible_links[idx] = " ".join(
                        hyperlinks.filter_by(col, center=True).get_texts()
                    )
                table_df[3] = all_possible_links
            all_table_df.append(table_df)

        combined_df = all_table_df[0]
        for table_df in all_table_df[1:]:
            if table_df.iloc[0, 0] != "":
                combined_df = combined_df.append(table_df)
            else:
                combined_df.iloc[-1, 2] += " " + table_df.iloc[0, 2]  # combine text
                combined_df.iloc[-1, 3] += table_df.iloc[0, 3]  # combine link
                combined_df = combined_df.append(table_df.iloc[1:])

        combined_df.iloc[0, 3] = "URL"
        combined_df.columns = combined_df.iloc[0].tolist()
        return combined_df.drop([0])

    @staticmethod
    def collect_all_plantiff_data(all_plantiffs_blocks):
        all_plantiff_data = defaultdict(dict)

        for page_plantiffs_blocks in all_plantiffs_blocks.values():
            for plantiff_block in page_plantiffs_blocks:
                plantiff_data = plantiff_block.get_info()
                all_plantiff_data[plantiff_data["block_type"].lower()].update(
                    plantiff_data["block_content"]
                )
        return all_plantiff_data

    def parse(self, pdf_file):

        pdf_tokens, pdf_images = self.pdf_extractor.load_tokens_and_image(
            pdf_file, resize_image=True
        )

        all_tables = self.table_parser.parse_tables_from_pdf_data(
            pdf_tokens, pdf_images
        )
        docket_df = self.collect_genererated_table_data(all_tables, pdf_tokens)

        all_plantiffs_blocks = self.page_parser.parse_page_structure_from_pdf_data(
            pdf_tokens, table_regions=all_tables
        )
        all_plantiffs_data = self.collect_all_plantiff_data(all_plantiffs_blocks)

        case_flags = self.page_parser.parse_case_flags(pdf_tokens)
        case_json = self.text_parser.parse_text_from_pdf_data(pdf_tokens, case_flags)

        case_json.update(all_plantiffs_data)

        return docket_df, case_json