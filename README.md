# Passport Lens — Streamlit MVP

Локальный MRZ-first MVP для загранпаспортов Китая, Узбекистана, России, Турции, Казахстана и Таджикистана. Реализует quality gate, OCR, строгий ICAO TD3 parser, checksum-guided коррекцию, причины решения и экспорт JSON.

## Быстрый запуск

```bash
make setup
make test
make run
```

Откройте `http://localhost:8501`. Первый OCR-запуск может быть медленнее из-за инициализации ONNX-моделей.

Docker-вариант:

```bash
docker compose up --build
```

## Что уже работает

- JPEG/PNG до 12 МБ и 30 Мп, EXIF orientation;
- blur / brightness / glare / resolution checks;
- три варианта MRZ preprocessing;
- локальный RapidOCR через ONNX Runtime;
- ICAO TD3 2×44 parser и пять checksum-проверок;
- консервативная O/0, I/1, B/8, S/5, G/6 коррекция только по checksum;
- country hint и контроль шести целевых issuing-state кодов;
- `accepted / review / retry_capture`, reason codes, provenance и JSON.

## Честные ограничения MVP

Универсальный OCR не гарантирует production-accuracy на реальных паспортах без consented golden set. Национальные VIZ-поля, классификация поколений, NFC и authenticity checks требуют отдельных country packs/SDK и реальных образцов. Этот MVP не заявляет проверку подлинности: валидная MRZ может быть напечатана на копии.

Для пилота соберите обезличенный holdout по каждой стране/версии/условию съёмки и измеряйте exact-match номера, DOB, expiry, ФИО, долю retry/review и latency p95.
