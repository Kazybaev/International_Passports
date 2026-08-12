# OCR model benchmark

| Model | OK | Field exact | MRZ exact | MRZ valid | Accepted | Mean ms | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| RapidOCR ONNX (`rapidocr`) | 300/300 | 65.62% | 85.5% | 0.0% | 0.0% | 3948.31 | 0 |
| Replit OCR endpoint (`replit`) | 0/300 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0 | 300 |

Best overall: `rapidocr` by field exact, MRZ validity, then latency.

Per-field exact-match:
- `rapidocr`: document_number=96.0%, surname=32.67%, given_names=36.0%, nationality=100.0%, birth_date=98.33%, sex=1.0%, expiry_date=95.33%
- `replit`: document_number=0.0%, surname=0.0%, given_names=0.0%, nationality=0.0%, birth_date=0.0%, sex=0.0%, expiry_date=0.0%
