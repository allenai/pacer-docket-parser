from dataclasses import dataclass
from typing import Union, List

import layoutparser as lp


@dataclass
class PDFPage:
    height: Union[float, int]
    width: Union[float, int]
    tokens: lp.Layout
    url_tokens: lp.Layout
    lines: List[lp.Rectangle]

    def get_text_segments(self, x_tolerance=10, y_tolerance=10) -> List[List]:
        """Get text segments from the current page.
        It will automatically add new lines for 1) line breaks
        2) big horizontal gaps
        """
        prev_y = None
        prev_x = None

        lines = []
        token_in_this_line = []
        n = 0

        for token in self.tokens:
            cur_y = token.block.center[1]
            cur_x = token.coordinates[0]

            if prev_y is None:
                prev_y = cur_y
                prev_x = cur_x

            if abs(cur_y - prev_y) <= y_tolerance and cur_x - prev_x <= x_tolerance:

                token_in_this_line.append(token)
                if n == 0:
                    prev_y = cur_y
                else:
                    prev_y = (prev_y * n + cur_y) / (n + 1)
                n += 1

            else:
                lines.append(token_in_this_line)
                token_in_this_line = [token]
                n = 1
                prev_y = cur_y

            prev_x = token.coordinates[2]

        if token_in_this_line:
            lines.append(token_in_this_line)

        return lines

    def get_text(self, x_tolerance=10, y_tolerance=10) -> str:
        """Returns the page text by instering '\n' between text segments
        returned by `self.get_text_segments` .
        """

        return "\n".join(
            [
                " ".join([e.text for e in ele])
                for ele in self.get_text_segments(x_tolerance, y_tolerance)
            ]
        )
