from abc import ABC, abstractmethod


class BasePDFTokenExtractor(ABC):
    """PDF token extractors will load all the *tokens* and save using pdfstructure service."""

    def __call__(self, pdf_path):
        return self.extract(pdf_path)

    @abstractmethod
    def extract(self, pdf_path):
        """Extract PDF Tokens from the input pdf_path

        Args:
            pdf_path (str):
                The path to a downloaded PDF file or a pdf SHA,
                e.g., sha://004cff2a0ed89f5f3855690f3fd2cc2778dc1a8e

        Returns:
            (PdfAnnotations):
                The PDF document structure are saved in the pdf-structure-service format
                https://github.com/allenai/s2-pdf-structure-service/tree/master/clients/python
        """
        pass

    @property
    @abstractmethod
    def NAME(self):
        """The name of the TokenExtractor"""
        pass
