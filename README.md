# PACER Docket Parser - Parsing PACER Docket PDF files with ease

This project aims to extract structured information from PACER docket PDF files and store in JSON formats specified [here](https://github.com/scales-okn/PACER-tools/tree/master/code/parsers#json-schema).

## Installation

1. Install the `docketparser` library via:
    ```bash
    git clone https://github.com/allenai/pacer-docket-parser.git
    cd ./pacer-docket-parser 
    pip install -e .
    pip install 'git+https://github.com/facebookresearch/detectron2.git#egg=detectron2' 
    ```
    You might find more instructions on Detectron2 installation [here](https://github.com/Layout-Parser/layout-parser/blob/master/installation.md#optional-install-detectron2-for-using-layout-models).

2. We use Poppler to render PDF documents as images - the installation methods are different based on your platform:
    1. Mac: `brew install poppler`
    2. Ubuntu: `sudo apt-get install -y poppler-utils`
    3. Windows: See [this post](https://stackoverflow.com/questions/18381713/how-to-install-poppler-on-windows)

## Usage

### Docket Table detection and extraction for PDF
Use the following command to extract docket tables from a pdf file:
```bash
docketparser parse-all [PDF_FILES] [SAVE_PATH]
```
It will save the extracted table and metadata json for each PDF_FILE as `filename.csv` and `filename.json`. Please check the exemplar outputs [here](https://drive.google.com/drive/folders/1iG84OfOZ-U9oUiFw75pmyBUO4-6jq01D?usp=sharing). For each PDF, we generate a single csv file, which merges the tables from each page according. 
