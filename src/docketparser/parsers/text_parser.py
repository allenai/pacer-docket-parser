import re

from . import text_helper
from ..pdftools import PDFExtractor


def generic_re_existence_helper(obj, split_text, index):
    if obj != None:
        return line_detagger(obj.group()).split(split_text)[index]
    else:
        return None


def re_existence_helper(obj):
    if obj != None:
        return line_detagger(obj.group()).split(": ")[-1]
    else:
        return None


def line_detagger(obj):
    if obj != None:
        while (
            obj.count("***") > 1
        ):  # sometimes Pacer does, e.g., "***DO NOT FILE IN THIS CASE***"
            obj = obj.split("***")[0] + "***".join(obj.split("***")[2:])
        return re.sub("\<[^<]+?>", "", obj).strip("<>?! ")
    else:
        return None


def line_cleaner(string):
    if string != None:
        string = (
            string.replace("&amp;", "&")
            .replace("&nbsp;", " ")
            .replace("\\'", "'")
            .replace("\\n", "")
        )
        string = string.lstrip(")").rstrip("(")
        string = " ".join(string.split()).strip()
        return string
    else:
        return None


def get_mdl_code(case_data):
    """
    Get the mdl code if present
    The same as https://github.com/scales-okn/PACER-tools/blob/5a170f900df41e6395507023d52069bd24a3ce5f/code/parsers/parse_pacer.py#L362

    Inputs:
        - case_data (dict): the rest of the case_data
    Outputs:
        - mdl_code (int): the case no for the mdl
        - mdl_id_source (str): the source of the identification of the code
    """

    # Check lead case
    lead_case_id = case_data.get("lead_case_id")
    if lead_case_id:
        code = text_helper.mdl_code_from_casename(lead_case_id)
        if code:
            return (code, "lead_case_id")

    # Check flags
    for flag in case_data.get("case_flags") or []:
        code = text_helper.mdl_code_from_string(flag)
        if code:
            return (code, "flag")

    return (None, None)


class RegexGroup:
    def __init__(self):
        # Resuing functions from  https://github.com/scales-okn/PACER-tools/blob/5a170f900df41e6395507023d52069bd24a3ce5f/code/parsers/parse_pacer.py
        self.re_fdate = re.compile("Date Filed: [0-9]{2}/[0-9]{2}/[0-9]{4}")
        self.re_tdate = re.compile("Date Terminated: [0-9]{2}/[0-9]{2}/[0-9]{4}")
        self.re_judge = re.compile("Assigned to: [ A-Za-z'\.\,\-\\\\]{1,100}")
        self.re_referred_judge = re.compile("Referred to: [ A-Za-z'\.\,\-\\\\]{1,100}")
        self.re_nature = re.compile("Nature of Suit: [A-Za-z0-9 :()\.]{1,100}")
        self.re_jury = re.compile("Jury Demand: [A-Za-z0-9 :(\.)]{1,100}")
        self.re_cause = re.compile("Cause: [A-Za-z0-9 :(\.)]{1,100}")
        self.re_jurisdiction = re.compile("Jurisdiction: [A-Za-z0-9 :(\.)]{1,100}")
        self.re_lead_case_id = re.compile("Lead case: [A-Za-z0-9:-]{1,100}")
        self.re_demand = re.compile("Demand: [0-9\,\$]{1,100}")
        self.re_other_court = re.compile(
            "Case in other court: [A-Za-z0-9 :;()\.\,\-]{1,100}"
        )  # brittle but functional
        self.re_header_case_id = re.compile("DOCKET FOR CASE #: [A-Za-z0-9 :\-]{1,100}")

        # Used
        self.re_party = re.compile("<b><u>([A-Za-z\- ]{1,100})(?:</u|\()")
        self.re_cr_title = re.compile(
            "Case title: [ A-Za-z0-9#&;()'\.\,\-\$\/\\\\]{1,100}"
        )
        self.re_cv_title = re.compile(
            "(\<br( \/)?\>|\))([^(][ A-Za-z0-9#&;()'\.\,\-\$\/\\\\]{1,100} v.? |(I|i)n (R|r)e:?)[ A-Za-z0-9#&;()'\.\,\-\$\/\\\\]{1,100}(\<|\()"
        )


class RawTextParser:
    def __init__(self, pdf_extractor=None):

        if pdf_extractor is None:
            self.pdf_extractor = PDFExtractor("pdfplumber")
        else:
            self.pdf_extractor = pdf_extractor

        self.regex = RegexGroup()

    def parse_text_from_pdf(self, filename, case_flags=None):
        pdf_tokens = self.pdf_extractor.pdf_extractor.extract(filename)
        return self.parse_text_from_pdf_data(pdf_tokens)

    def parse_text_from_pdf_data(self, pdf_tokens, case_flags=None):

        if case_flags is None:
            case_flags = []

        doc_text = "\n".join([page_tokens.get_text() for page_tokens in pdf_tokens])

        case_data = {}
        case_data["case_flags"] = case_flags
        # fmt:off
        # Some fields are universal (judge, filing date, terminating date...)
        case_data["header_case_id"] = re_existence_helper(self.regex.re_header_case_id.search(doc_text))
        case_data["filing_date"] = re_existence_helper(self.regex.re_fdate.search(doc_text))
        case_data["terminating_date"] = re_existence_helper(self.regex.re_tdate.search(doc_text))
        
        if case_data["terminating_date"] == None:
            case_data["case_status"] = "open"
        else:
            case_data["case_status"] = "closed"
        
        case_data["judge"] = line_cleaner(re_existence_helper(self.regex.re_judge.search(doc_text)))
        case_data["referred_judge"] = line_cleaner(re_existence_helper(self.regex.re_referred_judge.search(doc_text)))
        case_data["nature_suit"] = generic_re_existence_helper(self.regex.re_nature.search(doc_text), "Suit: ", -1)
        case_data["jury_demand"] = generic_re_existence_helper(self.regex.re_jury.search(doc_text), "Jury Demand: ", -1)
        case_data["cause"] = generic_re_existence_helper(self.regex.re_cause.search(doc_text), "Cause: ", -1)
        case_data["jurisdiction"] = generic_re_existence_helper(self.regex.re_jurisdiction.search(doc_text), "Jurisdiction: ", -1)
        case_data["monetary_demand"] = generic_re_existence_helper(self.regex.re_demand.search(doc_text), "Demand: ", -1)
        case_data["lead_case_id"] = generic_re_existence_helper(self.regex.re_lead_case_id.search(doc_text), "Lead case: ", -1)
        case_data["other_court"] = generic_re_existence_helper(self.regex.re_other_court.search(doc_text), "&nbsp;", -1)

        # add mdl extraction
        case_data["mdl_code"], case_data["mdl_id_source"] = get_mdl_code(case_data)
        # Is an mdl if we have a code OR if an 'MDL' or 'MDL_<description>' flag exists
        case_data["is_mdl"] = bool(case_data["mdl_code"]) or any(f.lower().startswith("mdl") for f in case_data["case_flags"])
        case_data['is_multi'] = any((case_data['is_mdl'], bool(case_data['lead_case_id']), bool(case_data['other_court']))) # Add bool(case_data['member_case_key'])
        # fmt:on

        return case_data