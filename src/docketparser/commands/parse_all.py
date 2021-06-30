import os
import click
from click import UsageError, BadArgumentUsage
import json

from tqdm import tqdm 

from ..parsers import MainParser
from .utils import *

@click.command(context_settings={"help_option_names": ["--help", "-h"]})
@click.argument(
    "pdf-files",
    type=click.Path(exists=True, file_okay=True),
    nargs=-1,
)
@click.argument(
    "save-path",
    type=click.Path(file_okay=False, dir_okay=True),
)
def parse_all(
    pdf_files: click.Path,
    save_path: click.Path = None,
):
    pdf_files = directory_check(pdf_files)
    print(pdf_files)
    if len(pdf_files) == 0: exit()
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print("Initializing Models for Main DocketParser")
    docket_parser = MainParser()

    pbar = tqdm(pdf_files)
    for pdf_file in pbar:
        
        pbar.set_description(f"Processing {pdf_file}")
        if os.path.exists(pdf_file.replace(".pdf", ".json")):
            continue
        
        try:
            docket_df, case_json = docket_parser.parse(pdf_file)
        
            case_json["docket"] = docket_df.apply(
                lambda row: row.to_list(), axis=1
            ).to_list()

            table_save_name = pdf_file.replace(".pdf", ".csv")
            json_save_name = pdf_file.replace(".pdf", ".json")
            if save_path is not None:
                table_save_name = os.path.join(save_path, os.path.basename(table_save_name))
                json_save_name = os.path.join(save_path, os.path.basename(json_save_name))
            pbar.set_description(f"Saved to {table_save_name}")
            
            docket_df.to_csv(table_save_name, index=None)
            with open(json_save_name, 'w') as fp:
                json.dump(case_json, fp, indent=4)
            pbar.set_description(f"Saved to {json_save_name}")

        except KeyboardInterrupt:
            exit()
        except:
            print(f"Serious issue for {pdf_file}")

if __name__ == '__main__':
    parse_all()