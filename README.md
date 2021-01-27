# The Docket PDF Parsing Project 

This project aims to extract information from docket PDF files and store in JSON formats specified [here](https://github.com/scales-okn/PACER-tools/tree/master/code/parsers#json-schema).

## Installation

```bash
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/detectron2.git#egg=detectron2' 
```

And we use Poppler to render PDF documents as images - the installation methods are different based on your platform:
1. Mac: `brew install poppler`
2. Ubuntu: `sudo apt-get install -y poppler-utils`
3. Windows: See [this post](https://stackoverflow.com/questions/18381713/how-to-install-poppler-on-windows)


## Note

The PDF extraction utility is directly copied from [allenai/scienceparseplus](https://github.com/allenai/scienceparseplus/tree/main/src/scienceparseplus/pdftools).