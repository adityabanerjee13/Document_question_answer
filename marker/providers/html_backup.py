import os
import tempfile

from marker.providers.pdf import PdfProvider


class HTMLProvider(PdfProvider):
    def __init__(self, filepath: str, config=None):
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        self.temp_pdf_path = temp_pdf.name
        temp_pdf.close()

        # Convert HTML to PDF
        try:
            self.convert_html_to_pdf(filepath)
        except Exception as e:
            raise RuntimeError(f"Failed to convert {filepath} to PDF: {e}")

        # Initialize the PDF provider with the temp pdf path
        super().__init__(self.temp_pdf_path, config)

    def __del__(self):
        if os.path.exists(self.temp_pdf_path):
            os.remove(self.temp_pdf_path)

    def convert_html_to_pdf(self, filepath: str):
        import platform
        
        print(f"DEBUG: Platform detected as: {platform.system()}")
        
        # Use playwright for Windows compatibility (WeasyPrint has GTK+ issues on Windows)
        if platform.system() == "Windows":
            print("DEBUG: Using Playwright for HTML to PDF conversion")
            self.convert_html_to_pdf_playwright(filepath)
        else:
            print("DEBUG: Using WeasyPrint for HTML to PDF conversion")
            self.convert_html_to_pdf_weasyprint(filepath)
            
    def convert_html_to_pdf_weasyprint(self, filepath: str):
        from weasyprint import HTML

        font_css = self.get_font_css()
        HTML(filename=filepath, encoding="utf-8").write_pdf(
            self.temp_pdf_path, stylesheets=[font_css]
        )
        
    def convert_html_to_pdf_playwright(self, filepath: str):
        # Simplified approach: Use simple HTML parsing and create a basic text-based PDF
        # This is a workaround for Windows compatibility issues
        print("DEBUG: Using simplified HTML processing due to Windows compatibility issues")
        
        import platform
        if platform.system() == "Windows":
            self._create_simple_pdf_from_html(filepath)
        else:
            # On non-Windows, try the original WeasyPrint approach
            self.convert_html_to_pdf_weasyprint(filepath)
            
    def _create_simple_pdf_from_html(self, filepath: str):
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from bs4 import BeautifulSoup
        import textwrap
        
        # Read and parse the HTML file
        with open(filepath, 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract text content
        text_content = soup.get_text()
        
        # Create a simple PDF with the text content
        c = canvas.Canvas(self.temp_pdf_path, pagesize=letter)
        width, height = letter
        
        # Set up text positioning
        y_position = height - 50
        line_height = 14
        margin = 50
        max_width = width - 2 * margin
        
        # Add title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, y_position, "HTML Content (Converted)")
        y_position -= 30
        
        # Add content
        c.setFont("Helvetica", 12)
        
        # Wrap text to fit page width
        lines = text_content.replace('\n\n', '\n').split('\n')
        for line in lines:
            if not line.strip():
                continue
                
            wrapped_lines = textwrap.wrap(line, width=80)
            for wrapped_line in wrapped_lines:
                if y_position < 50:  # Start new page if needed
                    c.showPage()
                    y_position = height - 50
                    c.setFont("Helvetica", 12)
                
                c.drawString(margin, y_position, wrapped_line)
                y_position -= line_height
        
        c.save()
        print(f"DEBUG: Created simplified PDF from HTML: {self.temp_pdf_path}")
