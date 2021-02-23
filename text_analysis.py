import re
import fitz

# Resuing functions from  https://github.com/scales-okn/PACER-tools/blob/5a170f900df41e6395507023d52069bd24a3ce5f/code/parsers/parse_pacer.py
re_fdate = re.compile('Date Filed: [0-9]{2}/[0-9]{2}/[0-9]{4}')
re_tdate = re.compile('Date Terminated: [0-9]{2}/[0-9]{2}/[0-9]{4}')
re_judge = re.compile('Assigned to: [ A-Za-z\'\.\,\-\\\\]{1,100}')
re_referred_judge = re.compile('Referred to: [ A-Za-z\'\.\,\-\\\\]{1,100}')
re_cr_title = re.compile('Case title: [ A-Za-z0-9#&;()\'\.\,\-\$\/\\\\]{1,100}')
re_cv_title = re.compile('(\<br( \/)?\>|\))([^(][ A-Za-z0-9#&;()\'\.\,\-\$\/\\\\]{1,100} v.? |(I|i)n (R|r)e:?)[ A-Za-z0-9#&;()\'\.\,\-\$\/\\\\]{1,100}(\<|\()')
re_nature = re.compile('Nature of Suit: [A-Za-z0-9 :()\.]{1,100}')
re_jury = re.compile('Jury Demand: [A-Za-z0-9 :(\.)]{1,100}')
re_cause = re.compile('Cause: [A-Za-z0-9 :(\.)]{1,100}')
re_jurisdiction = re.compile('Jurisdiction: [A-Za-z0-9 :(\.)]{1,100}')
re_lead_case_id = re.compile('Lead case: <a href=[^>]*>[A-Za-z0-9:-]{1,100}')
re_demand = re.compile('Demand: [0-9\,\$]{1,100}')
re_other_court = re.compile('Case in other court: [A-Za-z0-9 :;()\.\,\-]{1,100}') # brittle but functional
re_party = re.compile('<b><u>([A-Za-z\- ]{1,100})(?:</u|\()')
re_header_case_id = re.compile('DOCKET FOR CASE #: [A-Za-z0-9 :\-]{1,100}')


def generic_re_existence_helper(obj, split_text, index):
    if obj != None:
        return line_detagger(obj.group()).split(split_text)[index]
    else:
        return None

def re_existence_helper(obj):
    if obj != None:
        return line_detagger(obj.group()).split(': ')[-1]
    else:
        return None

def line_detagger(obj):
    if obj != None:
        while obj.count('***') > 1: # sometimes Pacer does, e.g., "***DO NOT FILE IN THIS CASE***"
            obj = obj.split('***')[0] + '***'.join(obj.split('***')[2:])
        return re.sub('\<[^<]+?>', '', obj).strip('<>?! ')
    else:
        return None

def line_cleaner(string):
    if string != None:
        string = string.replace('&amp;','&').replace('&nbsp;',' ').replace('\\\'','\'').replace('\\n','')
        string = string.lstrip(')').rstrip('(')
        string = ' '.join(string.split()).strip()
        return string
    else:
        return None


def raw_text_parse(filename):
    
    case_data = {}

    doc = fitz.open(filename)
    doc_text = '\n'.join([subdoc.get_text() for subdoc in doc])
    doc.close()
    
    # Some fields are universal (judge, filing date, terminating date...)
    case_data['header_case_id'] = re_existence_helper( re_header_case_id.search(doc_text) )
    case_data['filing_date'] = re_existence_helper( re_fdate.search(doc_text) )
    case_data['terminating_date'] = re_existence_helper( re_tdate.search(doc_text) )
    if case_data['terminating_date'] == None:
        case_data['case_status'] = 'open'
    else:
        case_data['case_status'] = 'closed'
    case_data['judge'] = line_cleaner(re_existence_helper( re_judge.search(doc_text) ))
    case_data['referred_judge'] = line_cleaner(re_existence_helper( re_referred_judge.search(doc_text) ))
    case_data['nature_suit'] = generic_re_existence_helper( re_nature.search(doc_text), 'Suit: ', -1)
    case_data['jury_demand'] = generic_re_existence_helper( re_jury.search(doc_text), 'Jury Demand: ', -1)
    case_data['cause'] = generic_re_existence_helper( re_cause.search(doc_text), 'Cause: ', -1)
    case_data['jurisdiction'] = generic_re_existence_helper( re_jurisdiction.search(doc_text), 'Jurisdiction: ', -1)
    case_data['monetary_demand'] = generic_re_existence_helper( re_demand.search(doc_text), 'Demand: ', -1)
    case_data['lead_case_id'] = generic_re_existence_helper( re_lead_case_id.search(doc_text), 'Lead case: ', -1)
    case_data['other_court'] = generic_re_existence_helper( re_other_court.search(doc_text), '&nbsp;', -1)

    return case_data