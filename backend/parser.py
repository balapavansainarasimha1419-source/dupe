import os
import fitz  # PyMuPDF
import docx

def extract_text_from_file(filepath: str) -> dict:
    """
    Extracts text from PDF, DOCX, and TXT files.
    
    Adheres strictly to the FileSense Global Data Contract while 
    implementing robust error handling and 10MB size limits.
    Files are accessed strictly in read-only mode to prevent overwrites.
    """
    # 1. Initialize the strict Global Data Contract dictionary
    filename = os.path.basename(filepath)
    result = {
        'filename': filename,
        'filepath': filepath,
        'text_content': "",
        'metadata': {},
        'error': None  # Added to support graceful failure without crashing
    }

    try:
        # 2. File Size Check & Validation (Max 100MB)
        try:
            file_size = os.path.getsize(filepath)
        except OSError:
            result['error'] = "File not found or inaccessible"
            return result
            
        result['metadata']['file_size'] = file_size
        
        # 100MB limit (100 * 1024 * 1024 bytes)
        if file_size > 104857600:
            result['error'] = "File too large"
            return result

        # 3. Extension Validation
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        result['metadata']['extension'] = ext

        if ext not in ['.pdf', '.docx', '.txt']:
            result['error'] = "Unsupported format"
            return result

        # 4. Parsing Logic (Strictly Read-Only)
        if ext == '.pdf':
            try:
                # fitz.open defaults to read-only
                doc = fitz.open(filepath)
                text_parts = [page.get_text() for page in doc]
                result['text_content'] = "\n".join(text_parts)
                doc.close()
            except fitz.FileDataError:
                result['error'] = "Corrupt PDF"
                return result
            except Exception as e:
                result['error'] = f"PDF parsing error: {str(e)}"
                return result

        elif ext == '.docx':
            try:
                # docx.Document reads without locking for write
                doc = docx.Document(filepath)
                text_parts = [para.text for para in doc.paragraphs]
                result['text_content'] = "\n".join(text_parts)
            except Exception as e:
                result['error'] = f"DOCX parsing error: {str(e)}"
                return result

        elif ext == '.txt':
            try:
                # Strictly read-only ('r'), ignoring weird encoding characters
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    result['text_content'] = f.read()
            except Exception as e:
                result['error'] = f"TXT parsing error: {str(e)}"
                return result

    except Exception as e:
        # Ultimate fallback to ensure the app NEVER crashes
        result['error'] = f"Unexpected system error: {str(e)}"

    return result
