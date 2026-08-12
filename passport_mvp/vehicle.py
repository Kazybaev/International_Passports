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


def extract_vehicle_fields(lines: list[str], ocr_lines: list[dict] | None = None) -> dict[str, str]:
    """Conservative OCR extraction for vehicle registration documents."""
    text = "\n".join(lines).upper()
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
    chinese_brands = (
        ("解放", "173", "FAW"), ("东风", "149", "DONGFENG"),
        ("福田", "188", "FOTON"), ("中国重汽", "261", "HOWO"), ("豪沃", "261", "HOWO"),
        ("陕汽", "561", "SHAANXI"), ("江淮", "294", "JAC"), ("宇通", "693", "YUTONG"),
        ("金龙", "331", "KING LONG"), ("比亚迪", "093", "BYD"), ("吉利", "200", "GEELY"),
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
                if label in upper:
                    value = re.sub(rf"^.*?{re.escape(label)}\s*[:№#-]?\s*", "", upper).strip()
                    if value and value != upper:
                        return value
                    if index + 1 < len(lines):
                        return lines[index + 1].strip().upper()
        return ""

    fields["vin"] = fields["vin"] or value_after(
        "VEHICLE IDENTIFICATION NUMBER", "ИДЕНТИФИКАЦИОННЫЙ НОМЕР (VIN)",
        "ИДЕНТИФИКАЦИОННЫЙ НОМЕР", "车辆识别代号", "车辆识别代码", "VIN",
    )
    fields["model"] = value_after("МОДЕЛЬ", "MODEL")
    fields["body_number"] = value_after("НОМЕР КУЗОВА", "BODY NO")
    fields["chassis_number"] = value_after("НОМЕР ШАССИ", "CHASSIS NO")
    fields["registration_number"] = value_after("ГОСУДАРСТВЕННЫЙ НОМЕР", "РЕГИСТРАЦИОННЫЙ НОМЕР", "REGISTRATION NO")
    fields["registration_date"] = value_after("ДАТА РЕГИСТРАЦИИ", "DATE OF REGISTRATION", "REGISTRATION DATE")
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
    model_match = re.search(r"[A-Z]{1,5}[A-Z0-9.-]{4,}", fields["model"])
    if model_match:
        fields["model"] = model_match.group(0)
    date_match = re.search(r"20\d{2}[-./]\d{1,2}[-./]\d{1,2}", fields["registration_date"])
    if date_match:
        fields["registration_date"] = date_match.group(0).replace(".", "-").replace("/", "-")
    return fields


def is_vehicle_document(lines: list[str]) -> bool:
    text = " ".join(lines).upper()
    score = sum(marker in text for marker in (
        "VIN", "ТРАНСПОРТ", "РЕГИСТРАЦ", "КУЗОВ", "ШАССИ",
        "VEHICLE LICENSE", "PLATE NO", "REGISTER DATE", "车辆", "号牌",
    ))
    return score >= 2 or bool(re.search(r"(?<![A-Z0-9])[A-HJ-NPR-Z0-9]{17}(?![A-Z0-9])", text))
