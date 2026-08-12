# OCR model benchmark

| Model | OK | Field exact | MRZ exact | MRZ valid | Accepted | Mean ms | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek OCR (`deepseek_ocr`) | 0/2 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0 | 2 |
| Marker OCR (`marker`) | 0/2 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0 | 2 |
| RapidOCR ONNX (`rapidocr`) | 2/2 | 64.28% | 50.0% | 0.0% | 0.0% | 4049 | 0 |
| Replit OCR endpoint (`replit`) | 0/2 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0 | 2 |

Best overall: `rapidocr` by field exact, MRZ validity, then latency.

Per-field exact-match:
- `deepseek_ocr`: document_number=0.0%, surname=0.0%, given_names=0.0%, nationality=0.0%, birth_date=0.0%, sex=0.0%, expiry_date=0.0%
- `marker`: document_number=0.0%, surname=0.0%, given_names=0.0%, nationality=0.0%, birth_date=0.0%, sex=0.0%, expiry_date=0.0%
- `rapidocr`: document_number=100.0%, surname=0.0%, given_names=50.0%, nationality=100.0%, birth_date=100.0%, sex=0.0%, expiry_date=100.0%
- `replit`: document_number=0.0%, surname=0.0%, given_names=0.0%, nationality=0.0%, birth_date=0.0%, sex=0.0%, expiry_date=0.0%
