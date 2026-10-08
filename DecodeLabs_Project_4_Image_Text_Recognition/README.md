# DecodeLabs Project 4 – Image & Text Recognition (OCR)

This project implements **Path 1: Optical Character Recognition (OCR)** from DecodeLabs Project 4.

## What it demonstrates
- `pytesseract` integration with the Tesseract OCR engine.
- OpenCV pre-processing:
  1. grayscale conversion
  2. Gaussian blur
  3. adaptive thresholding
  4. optional deskew
- OCR confidence filtering with the Project 4 **80% minimum confidence gate**.
- Visual confirmation: accepted OCR text is drawn on the processed image.
- Saves both the recognized text and an annotated image.

## Project structure
```text
DecodeLabs_Project_4_Image_Text_Recognition/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── input/
│   └── sample_text.png        # add your own image here
└── output/
```

## 1. Install Python packages
```bash
pip install -r requirements.txt
```

## 2. Install Tesseract OCR
`pytesseract` is a Python wrapper; the Tesseract executable must also be installed.

### Windows
Install Tesseract OCR, then either add it to PATH or set the path in `main.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install tesseract-ocr
```

## 3. Run
Place an image in `input/` and run:
```bash
python main.py --image input/sample_text.png
```

Or:
```bash
python main.py --image path/to/your/image.jpg --output-dir output
```

## Output
- `output/processed.png` – pre-processed image
- `output/annotated.png` – visual confirmation with accepted OCR text
- `output/recognized.txt` – final text accepted by the 80% confidence gate
- `output/ocr_data.csv` – OCR words and confidence values

## Confidence rule
The script accepts an OCR word only when:
```text
confidence >= 80
```
Low-confidence detections are discarded rather than displayed as trusted output.

> Note: OCR confidence is the confidence returned by Tesseract. It is not a guarantee that the text is objectively correct.
