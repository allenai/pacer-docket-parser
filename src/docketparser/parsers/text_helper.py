import sys
import re

import pandas as pd

re_com = {
    'office': r"[0-9A-Za-z]",
    'year': r"[0-9]{2}",
    'case_type': r"[A-Za-z]{1,4}",
    'case_no': r"[0-9]{3,10}",
    'def_no': r"(?:-[0-9]{1,3})?",
    'href': r'''["']/cgi-bin/DktRpt.pl\?[0-9]{1,10}['"]''',
    'judge_names': r"(?:-{1,2}[A-Za-z]{1,4}){0,3}",
    'update_ind': r"\d*"
}
rg = lambda k: rf"(?P<{k}>{re_com[k]})"
re_mdl = r"MDL\s*(no.|[-_])?\s*(?P<code>\d{2,5})"
re_case_no_gr = rf"{rg('office')}:{rg('year')}-{rg('case_type')}-{rg('case_no')}{rg('judge_names')}{rg('def_no')}_?{rg('update_ind')}"
re_mdl_caseno_condensed = rf"{rg('year')}-?{rg('case_type')}-?{rg('case_no')}"

def colonize(case_no):
    ''' Puts colon after office in case_no if it doesn't exist'''
    if ':' not in case_no:
        case_no = case_no.replace('-', ':', 1)
    return case_no

def decompose_caseno(case_no, pattern=re_case_no_gr):
    ''' Decompose a case no. of the fomrat "2:16-cv-01002-ROS" '''
    case_no = colonize(case_no)
    match =  re.search(pattern, case_no)
    if not match:
        raise ValueError(f"case_no supplied ({case_no})was not in the expected format, see re_case_no_gr")
    else:
        data = match.groupdict()
        judges = data['judge_names'].strip('-').replace('--','-').split('-') if data.get('judge_names','') != '' else None
        data['judge_names'] = judges
        data['def_no'] = data['def_no'].lstrip('-') if data.get('def_no','') != '' else None
        return data

def mdl_code_from_string(string):
    ''' Search for mdl_code in a certain string'''
    mdl_match = re.search(re_mdl, string, re.IGNORECASE)
    if mdl_match:
        return int(mdl_match.groupdict()['code'])

def mdl_code_from_casename(casename):
    try:
        casename_data = decompose_caseno(casename)
    except ValueError:
        pass

    try:
        casename_data = decompose_caseno(casename, pattern=re_mdl_caseno_condensed)
    except ValueError:
        return None

    if casename_data['case_type'].lower() in ['md','ml', 'mdl']:
        return int(casename_data['case_no'])