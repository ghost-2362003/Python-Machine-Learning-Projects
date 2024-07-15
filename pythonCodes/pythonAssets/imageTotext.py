import pytesseract as tess
tess.pytesseract.tesseract_cmd = r'C:\Users\shubh\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
from PIL import Image

image = Image.open('pythonCodes/pythonAssets/image2.jpg')
text = tess.image_to_string(image)

print(text)