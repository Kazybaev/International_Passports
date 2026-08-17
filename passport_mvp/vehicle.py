from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def vehicle_brands() -> list[dict[str, str]]:
    """Official vehicle-make classifier, shipped locally with the app."""
    path = Path(__file__).with_name("vehicle_brands.json")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def vehicle_types() -> list[dict[str, str]]:
    path = Path(__file__).with_name("vehicle_types.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _value_right_of_label(ocr_lines: list[dict], labels: tuple[str, ...]) -> str:
    """Find the nearest OCR object to the right of a printed field label."""
    objects = []
    for item in ocr_lines:
        points = item.get("box") or []
        if len(points) < 4:
            continue
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        objects.append({
            "text": str(item.get("text", "")).strip(), "left": min(xs), "right": max(xs),
            "cx": sum(xs) / len(xs), "cy": sum(ys) / len(ys), "height": max(ys) - min(ys),
        })
    for label in objects:
        if not any(marker in label["text"].upper() for marker in labels):
            continue
        candidates = [
            item for item in objects
            if item is not label and item["cx"] > label["cx"]
            and abs(item["cy"] - label["cy"]) <= max(45, label["height"] * 2.5)
            and not any(marker in item["text"].upper() for marker in labels)
        ]
        if candidates:
            best = min(candidates, key=lambda item: abs(item["cy"] - label["cy"]) * 3 + max(0, item["left"] - label["right"]))
            return best["text"].strip().upper()
    return ""


def _chinese_plate(value: str) -> str:
    compact = re.sub(r"\s", "", value.upper())
    match = re.fullmatch(r"[\u4e00-\u9fff][A-Z][A-Z0-9]{4,6}[挂学警港澳]?", compact)
    return match.group(0) if match else ""


def _vehicle_segments(lines: list[str]) -> list[list[str]]:
    """Split concatenated OCR text into distinct vehicle-license records."""
    chinese_headers = [index for index, line in enumerate(lines) if "中华人民共和国机动车行驶证" in line]
    starts = chinese_headers
    if len(starts) < 2:
        english_headers = []
        for index, line in enumerate(lines):
            compact = re.sub(r"[^A-Z]", "", line.upper())
            if "VEHICLELICENSEOFTHEPEOPLESREPUBLICOFCHINA" in compact:
                if not english_headers or index - english_headers[-1] > 8:
                    english_headers.append(index)
        starts = english_headers
    if len(starts) >= 2:
        return [lines[start:end] for start, end in zip(starts, [*starts[1:], len(lines)]) if start < end]

    plate_indexes = [index for index, line in enumerate(lines) if _chinese_plate(line)]
    if len(plate_indexes) < 2:
        return [lines]
    boundaries = [0]
    boundaries.extend((left + right) // 2 for left, right in zip(plate_indexes, plate_indexes[1:]))
    boundaries.append(len(lines))
    return [lines[start:end] for start, end in zip(boundaries, boundaries[1:]) if start < end]


def extract_vehicle_fields(lines: list[str], ocr_lines: list[dict] | None = None) -> dict[str, str]:
    """Conservative OCR extraction for vehicle registration documents."""
    text = "\n".join(lines).upper()
    compact_text = re.sub(r"[^A-ZА-ЯЁ0-9]", "", text)
    vin_match = re.search(r"(?<![A-Z0-9])[A-HJ-NPR-Z0-9]{17}(?![A-Z0-9])", text)
    fields = {
        "vin": vin_match.group(0) if vin_match else "",
        "make_code": "", "make": "", "type_code": "", "type": "",
        "registration_number": "", "registration_date": "",
        "body_number": "", "chassis_number": "", "model": "",
    }
    for brand in sorted(vehicle_brands(), key=lambda item: len(item["name"]), reverse=True):
        if re.search(rf"(?<![A-ZА-Я0-9]){re.escape(brand['name'])}(?![A-ZА-Я0-9])", text):
            fields.update(make_code=brand["code"], make=brand["name"])
            break
    # Commercial-vehicle OCR commonly glues the make to the model. These
    # aliases are limited to distinctive manufacturer names; short names such
    # as MAN/JAC must retain word boundaries to avoid false positives.
    glued_make_aliases = (("SHACMAN", "561", "SHAANXI"),)
    if not fields["make"]:
        glued_make = next((item for item in glued_make_aliases if item[0] in compact_text), None)
        if glued_make:
            fields.update(make_code=glued_make[1], make=glued_make[2])
    chinese_brands = (
        ("解放", "173", "FAW"), ("东风", "149", "DONGFENG"),
        ("福田", "188", "FOTON"), ("中国重汽", "261", "HOWO"), ("豪沃", "261", "HOWO"),
        ("陕汽", "561", "SHAANXI"), ("江淮", "294", "JAC"), ("宇通", "693", "YUTONG"),
        ("金龙", "331", "KING LONG"), ("比亚迪", "093", "BYD"), ("吉利", "200", "GEELY"),
        ("豪瀚", "261", "HOWO"),
        ("长城", "217", "GREAT WALL"), ("奇瑞", "121", "CHERY"),
        ("现代", "272", "HYUNDAI"), ("丰田", "628", "TOYOTA"), ("大众", "665", "VOLKSWAGEN"),
    )
    if not fields["make"]:
        chinese_brand = next((item for item in chinese_brands if item[0] in text), None)
        if chinese_brand:
            fields.update(make_code=chinese_brand[1], make=chinese_brand[2])

    aliases = (
        ("ВОДНОЕ СУДНО", "100"), ("СЕДЕЛЬНЫЙ ТЯГАЧ", "307"),
        ("АВТОМОБИЛЬ-ТЯГАЧ", "306"), ("МИКРОАВТОБУС", "324"),
        ("СОЧЛЕНЕННЫЙ АВТОБУС", "323"), ("СПЕЦИАЛЬНЫЙ АВТОБУС", "322"),
        ("АВТОБУС", "321"), ("ГРУЗОВОЙ ПОЛУПРИЦЕП", "319"),
        ("СПЕЦИАЛЬНЫЙ ПОЛУПРИЦЕП", "320"), ("ПОЛУПРИЦЕП", "319"),
        ("СПЕЦИАЛЬНЫЙ ПРИЦЕП", "313"), ("ГРУЗОВОЙ ПРИЦЕП", "312"),
        ("КАРАВАН", "314"), ("СПЕЦИАЛЬНЫЙ ГРУЗОВОЙ", "304"),
        ("ГРУЗОПАССАЖИРСКИЙ", "305"), ("ГРУЗОВОЙ АВТОМОБИЛЬ", "303"),
        ("СПЕЦИАЛЬНЫЙ ЛЕГКОВОЙ", "302"), ("ЛЕГКОВОЙ АВТОМОБИЛЬ", "301"),
        ("КОНТЕЙНЕР", "901"), ("ВОЗДУШНОЕ СУДНО", "400"),
        ("重型半挂牵引车", "307"),
        ("牵引车", "307"), ("专项作业车", "304"), ("半挂车", "319"),
        ("挂车", "312"), ("货车", "303"), ("轿车", "301"), ("客车", "321"),
    )
    type_by_code = {item["code"]: item for item in vehicle_types()}
    type_code = next((code for marker, code in aliases if marker in text), "")
    if not type_code:
        matched_type = next(
            (item for item in sorted(vehicle_types(), key=lambda item: len(item["name"]), reverse=True)
             if item["name"].upper() in text),
            None,
        )
        type_code = matched_type["code"] if matched_type else ""
    if type_code:
        fields.update(type_code=type_code, type=type_by_code[type_code]["name"])

    def value_after(*labels: str) -> str:
        for index, raw_line in enumerate(lines):
            upper = raw_line.upper().strip()
            for label in labels:
                label_match = re.search(
                    rf"(?<![A-ZА-ЯЁ0-9]){re.escape(label)}(?![A-ZА-ЯЁ0-9])",
                    upper,
                )
                if label_match:
                    value = upper[label_match.end():].lstrip(" :№#-").strip()
                    if value and value != upper:
                        return value
                    if index + 1 < len(lines):
                        return lines[index + 1].strip().upper()
        return ""

    def numbered_values(number: str) -> list[str]:
        values = []
        for raw_line in lines:
            compact = re.sub(r"\s+", "", raw_line.upper())
            match = re.match(rf"^{number}[.)-]?([A-ZА-ЯЁ0-9][A-ZА-ЯЁ0-9.'/-]{{3,}})$", compact)
            if match:
                values.append(match.group(1).strip(". '-/"))
        return values

    fields["vin"] = fields["vin"] or value_after(
        "VEHICLE IDENTIFICATION NUMBER", "ИДЕНТИФИКАЦИОННЫЙ НОМЕР (VIN)",
        "ИДЕНТИФИКАЦИОННЫЙ НОМЕР", "车辆识别代号", "车辆识别代码", "VIN",
    )
    fields["model"] = value_after("МОДЕЛЬ", "MODEL", "品牌型号")
    fields["body_number"] = value_after("НОМЕР КУЗОВА", "BODY NO")
    fields["chassis_number"] = value_after("НОМЕР ШАССИ", "CHASSIS NO")
    fields["registration_number"] = value_after("ГОСУДАРСТВЕННЫЙ НОМЕР", "РЕГИСТРАЦИОННЫЙ НОМЕР", "REGISTRATION NO")
    fields["registration_date"] = value_after("ДАТА РЕГИСТРАЦИИ", "DATE OF REGISTRATION", "REGISTRATION DATE")
    if not fields["registration_number"]:
        plate_label = next((index for index, line in enumerate(lines) if "号牌号码" in line), -1)
        if plate_label >= 0:
            for candidate in lines[plate_label + 1:plate_label + 6]:
                compact = re.sub(r"\s", "", candidate.upper())
                # Mainland plate: province ideograph + Latin letter + 5–6
                # alphanumerics. Captions and arbitrary Chinese text cannot fit.
                plate = _chinese_plate(compact)
                if plate:
                    fields["registration_number"] = plate
                    break
    # Uzbek vehicle certificates use fixed numbered rows and often place the
    # value before the multilingual caption. OCR may remove the dot after the
    # number, so accept both `1.10A123BC` and `1 10A123BC` forms.
    def valid_registration(value: str) -> bool:
        compact = re.sub(r"[^A-Z0-9]", "", value)
        return 6 <= len(compact) <= 12 and bool(re.search(r"[A-Z]", value)) and bool(re.search(r"\d", value))

    def valid_model(value: str) -> bool:
        compact = re.sub(r"[^A-Z0-9]", "", value)
        return 5 <= len(compact) <= 32 and sum(character.isalpha() for character in value) >= 2 and bool(re.search(r"\d", value))

    # Keep values from repeated certificates together. A multi-document scan
    # must never combine field 1 from one certificate with field 2 from the
    # next certificate merely because the latter has a more recognizable make.
    records: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw_line in lines:
        compact = re.sub(r"\s+", "", raw_line.upper())
        match = re.match(r"^([126])[.)-]?([A-ZА-ЯЁ0-9][A-ZА-ЯЁ0-9.'/-]{3,})$", compact)
        if not match:
            continue
        number, value = match.group(1), match.group(2).strip(". '-/")
        if number == "1" and valid_registration(value):
            current = {"registration_number": value}
            records.append(current)
        elif current is not None and number == "2" and valid_model(value):
            current.setdefault("model", value)
        elif current is not None and number == "6" and re.fullmatch(r"\d{2}[./-]\d{2}[./-]\d{4}", value):
            current.setdefault("registration_date", value)

    if records:
        selected_record = max(
            records,
            key=lambda record: (
                len(record),
                int("SHACMAN" in record.get("model", "")),
            ),
        )
        for key, value in selected_record.items():
            if not fields[key]:
                fields[key] = value
    else:
        registration_candidates = [value for value in numbered_values("1") if valid_registration(value)]
        model_candidates = [value for value in numbered_values("2") if valid_model(value)]
        date_candidates = [value for value in numbered_values("6") if re.fullmatch(r"\d{2}[./-]\d{2}[./-]\d{4}", value)]
        if registration_candidates and not fields["registration_number"]:
            fields["registration_number"] = registration_candidates[0]
        if model_candidates and not fields["model"]:
            fields["model"] = model_candidates[0]
        if date_candidates and not fields["registration_date"]:
            fields["registration_date"] = date_candidates[0]
    if ocr_lines:
        fields["vin"] = fields["vin"] or _value_right_of_label(
            ocr_lines, ("VEHICLE IDENTIFICATION NUMBER", "ИДЕНТИФИКАЦИОННЫЙ НОМЕР", "车辆识别代号", "车辆识别代码", "VIN")
        )
        fields["registration_number"] = fields["registration_number"] or _value_right_of_label(
            ocr_lines, ("PLATE NO", "REGISTRATION NO", "号牌号码", "号牌")
        )
        fields["model"] = fields["model"] or _value_right_of_label(ocr_lines, ("MODEL",))
        fields["registration_date"] = fields["registration_date"] or _value_right_of_label(
            ocr_lines, ("REGISTER DATE", "REGISTERDATE", "REGISTERDALE", "REGISTERDAIE", "REGISTRATION DATE", "注册日期")
        )
    vin_match = re.search(r"(?<![A-Z0-9])[A-HJ-NPR-Z0-9]{17}(?![A-Z0-9])", fields["vin"].upper())
    fields["vin"] = vin_match.group(0) if vin_match else ""
    model_candidates = [
        candidate for candidate in re.findall(r"[A-Z0-9.-]{2,}", fields["model"].upper())
        if (
            candidate.isalpha()
            or (
                len(candidate) >= 5
                and any(character.isalpha() for character in candidate)
                and any(character.isdigit() for character in candidate)
            )
        )
    ]
    fields["model"] = max(model_candidates, key=len) if model_candidates else ""
    date_match = re.search(r"(?:20\d{2}[-./]\d{1,2}[-./]\d{1,2}|\d{1,2}[-./]\d{1,2}[-./]20\d{2})", fields["registration_date"])
    if not date_match:
        date_match = re.search(r"(?:20\d{2}[-./]\d{1,2}[-./]\d{1,2}|\d{1,2}[-./]\d{1,2}[-./]20\d{2})", text)
    if date_match:
        fields["registration_date"] = date_match.group(0).replace(".", "-").replace("/", "-")
    return fields


def extract_vehicle_records(lines: list[str], ocr_lines: list[dict] | None = None) -> list[dict[str, str]]:
    """Extract every distinct vehicle document without mixing their fields."""
    segments = _vehicle_segments(lines)
    records: list[dict[str, str]] = []
    for segment in segments:
        # Geometry belongs to the full page and is safe only when no split was
        # needed. Segmented text must not borrow a value from another record.
        fields = extract_vehicle_fields(segment, ocr_lines if len(segments) == 1 else None)
        if not (fields["registration_number"] or fields["vin"] or is_vehicle_document(segment)):
            continue
        identity = (fields["registration_number"], fields["vin"])
        existing = next(
            (record for record in records if identity != ("", "") and identity == (record["registration_number"], record["vin"])),
            None,
        )
        if existing:
            for key, value in fields.items():
                if value and not existing[key]:
                    existing[key] = value
        else:
            records.append(fields)
    return records


def is_vehicle_document(lines: list[str]) -> bool:
    text = " ".join(lines).upper()
    compact = re.sub(r"[^A-ZА-ЯЁ0-9]", "", text)
    score = sum(marker in text for marker in (
        "VIN", "ТРАНСПОРТ", "РЕГИСТРАЦ", "КУЗОВ", "ШАССИ",
        "VEHICLE LICENSE", "PLATE NO", "REGISTER DATE", "车辆", "号牌",
    ))
    score += sum(marker in compact for marker in (
        "AVTOMOTOTRANSPORTVOSITASI", "DAVLATRAQAMBELGISI",
        "ROYXATDANOTKAZILGANLIGI", "VEHICLELICENCE", "VEICLELICENCE",
        "RUSUMIMODELI", "BERILGANSANASI",
    ))
    return score >= 2 or bool(re.search(r"(?<![A-Z0-9])[A-HJ-NPR-Z0-9]{17}(?![A-Z0-9])", text))
