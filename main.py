import argparse

from table import *
from text_analysis import *
from page_structure import *

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, help="PDF filenames for parsing", nargs="+")


def collect_all_plantiff_data(all_plantiffs_blocks):
    all_plantiff_data = defaultdict(dict)

    for page_plantiffs_blocks in all_plantiffs_blocks.values():
        for plantiff_block in page_plantiffs_blocks:
            plantiff_data = plantiff_block.get_info()
            all_plantiff_data[plantiff_data["block_type"].lower()].update(
                plantiff_data["block_content"]
            )
    return all_plantiff_data


def collect_genererated_table_data(tables):
    all_dfs = [
        table.to_dataframe()
        for table in sum([tb for tb in tables.values() if tb != []], [])
    ]

    columns = all_dfs[0].iloc[0, :].tolist()
    all_dfs[0].drop([0], inplace=True)
    df = pd.concat(all_dfs)
    df.columns = columns
    return df


if __name__ == "__main__":
    args = parser.parse_args()

    model = lp.Detectron2LayoutModel(
        config_path="models/publaynet/mask_rcnn_R_50_FPN_3x/config.yaml",
        model_path="models/publaynet/mask_rcnn_R_50_FPN_3x/model_final.pth",
        extra_config=[
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
            0.2,
            "MODEL.ROI_HEADS.NMS_THRESH_TEST",
            0.5,
        ],
        label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"},
    )

    pdf_extractor = PDFExtractor("pdfplumber")
    table_detector = TableDetector(model, pdf_extractor)
    page_parser = PageStructureParser()

    for filename in args.filename:
        print(f"Processing {filename}")
        tables = table_detector.detect_tables_from_pdf(filename)
        df = collect_genererated_table_data(tables)

        all_plantiffs_blocks = page_parser.parse_page_structure(filename=filename, table_regions=tables)
        all_plantiffs_data = collect_all_plantiff_data(all_plantiffs_blocks)

        case_json = raw_text_parse(filename)
        case_json['docket'] = df.apply(lambda row: row.to_list() + [{}], axis=1).to_list()
        case_json.update(all_plantiffs_data)

        table_save_name = filename.replace(".pdf", ".csv")
        df.to_csv(table_save_name, index=None)
        print(f"Saved to {table_save_name}")

        json_save_name = filename.replace(".pdf", ".json")
        with open(json_save_name, 'w') as fp:
            json.dump(case_json, fp, indent=4)
        print(f"Saved to {json_save_name}")