from dataclasses import dataclass
from typing import Union, List

import layoutparser as lp


@dataclass
class PDFPage:
    height: Union[float, int]
    width: Union[float, int]
    tokens: lp.Layout
    url_tokens: lp.Layout

