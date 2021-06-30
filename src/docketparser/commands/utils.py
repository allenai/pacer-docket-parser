import os 
import glob 

def directory_check(pdf_files):
    if len(pdf_files) > 1: 
        assert all(os.path.isfile(pdf_file) for pdf_file in pdf_files)
    elif len(pdf_files) == 1 and os.path.isdir(pdf_files[0]):
        pdf_files = glob.glob(f"{pdf_files[0]}/*.pdf")
    
    return pdf_files