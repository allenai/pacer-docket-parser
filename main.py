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


def collect_genererated_table_data(tables, hyperlink_tokens):
    all_table_df = []
    for page_id, table in tables.items():
        if len(table)==0: continue
            
        if len(table)>1: print(f"Warning: page {page_id} has {len(table)} tables)")
        table = table[0] # we assume there's only one table per page
        table_df = table.to_dataframe()
        
        hyperlinks = hyperlink_tokens[page_id]["tokens"]
        
        if len(hyperlinks)==0: 
            table_df[3] = None 
        else:
            all_possible_links = [None]* len(table.rows)
            for idx, rows in enumerate(table.grid):
                col = rows[1]
                all_possible_links[idx] = ' '.join(hyperlinks.filter_by(col, center=True).get_texts())
            table_df[3] = all_possible_links
        all_table_df.append(table_df)
        
    combined_df = all_table_df[0]
    for table_df in all_table_df[1:]:
        if table_df.iloc[0,0]!='':
            combined_df = combined_df.append(table_df)
        else:
            combined_df.iloc[-1, 2] += ' ' + table_df.iloc[0, 2] # combine text
            combined_df.iloc[-1, 3] += table_df.iloc[0, 3] # combine link
            combined_df = combined_df.append(table_df.iloc[1:])

    combined_df.iloc[0, 3] = "URL"
    combined_df.columns = combined_df.iloc[0].tolist()
    return combined_df.drop([0])


if __name__ == "__main__":
    args = parser.parse_args()

    model = lp.Detectron2LayoutModel(
        "lp://TableBank/faster_rcnn_R_101_FPN_3x/config",
        extra_config=[
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
            0.2,
            "MODEL.ROI_HEADS.NMS_THRESH_TEST",
            0.5,
        ],
        label_map={0: "table"},
    )

    pdf_extractor = PDFExtractor("pdfplumber")
    table_detector = TableDetector(model, pdf_extractor)
    page_parser = PageStructureParser()

    pbar = tqdm(args.filename)
    for filename in pbar:
        pbar.set_description(f"Processing {filename}")
        try:
            tables = table_detector.detect_tables_from_pdf(filename)
            hyperlink_tokens = pdf_extractor.pdf_extractor.extract(filename, target_data="hyperlink")
            df = collect_genererated_table_data(tables, hyperlink_tokens)

            all_plantiffs_blocks = page_parser.parse_page_structure(filename=filename, table_regions=tables)
            all_plantiffs_data = collect_all_plantiff_data(all_plantiffs_blocks)

            case_json = raw_text_parse(filename)
            case_json['docket'] = df.apply(lambda row: row.to_list(), axis=1).to_list()
            case_json.update(all_plantiffs_data)
            case_json['case_flags'] = page_parser.fetch_case_flags(filename=filename)

            table_save_name = filename.replace(".pdf", ".csv")
            df.to_csv(table_save_name, index=None)
            pbar.set_description(f"Saved to {table_save_name}")

            json_save_name = filename.replace(".pdf", ".json")
            with open(json_save_name, 'w') as fp:
                json.dump(case_json, fp, indent=4)
            pbar.set_description(f"Saved to {json_save_name}")
        except KeyboardInterrupt:
            exit()
        except:
            print(f"Serious issue for {filename}")