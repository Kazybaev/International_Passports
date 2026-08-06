from __future__ import annotations

import re
from datetime import date
from itertools import product

WEIGHTS = (7, 3, 1)
ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
AMBIGUITIES = {"O": "0", "0": "O", "I": "1", "1": "I", "B": "8", "8": "B", "S": "5", "5": "S", "G": "6", "6": "G"}


def char_value(ch: str) -> int:
    if ch == "<": return 0
    if ch.isdigit(): return int(ch)
    if "A" <= ch <= "Z": return ord(ch) - 55
    raise ValueError(f"Недопустимый символ MRZ: {ch!r}")


def check_digit(value: str) -> str:
    return str(sum(char_value(ch) * WEIGHTS[i % 3] for i, ch in enumerate(value)) % 10)


def clean_line(value: str) -> str:
    value = value.upper().replace(" ", "<").replace("«", "<").replace("‹", "<")
    return "".join(ch for ch in value if ch in ALLOWED)


def normalize_lines(lines: list[str]) -> list[str]:
    cleaned = [clean_line(x) for x in lines if len(clean_line(x)) >= 20]
    exact = [x for x in cleaned if len(x) == 44]
    if len(exact) >= 2:
        selected = exact[-2:]
        passport_line = next((line for line in selected if line.startswith("P")), None)
        if passport_line:
            other = next(line for line in selected if line is not passport_line)
            return [passport_line, other]
        return selected
    joined = "".join(cleaned)
    if len(joined) >= 88:
        # Favor the final 88 characters: OCR often prefixes labels above the MRZ.
        return [joined[-88:-44], joined[-44:]]
    return cleaned[-2:]


def _date_yyMMdd(value: str, kind: str) -> str | None:
    if not re.fullmatch(r"\d{6}", value): return None
    yy, mm, dd = int(value[:2]), int(value[2:4]), int(value[4:])
    today = date.today()
    if kind == "birth":
        year = 2000 + yy if yy <= today.year % 100 else 1900 + yy
        if year > today.year: year -= 100
    else:
        # Passports seen by an MVP are unlikely to be >20 years expired or >20 years ahead.
        candidates = [1900 + yy, 2000 + yy, 2100 + yy]
        year = min(candidates, key=lambda y: abs(y - today.year))
    try: return date(year, mm, dd).isoformat()
    except ValueError: return None


def _name(value: str) -> str:
    return " ".join(part for part in value.replace("<", " ").split() if part)


def extract_raw_identity(lines: list[str]) -> dict[str, str]:
    """Recover names from a readable passport-like MRZ even if TD3 checks fail."""
    if not lines:
        return {}
    line = clean_line(lines[0])
    match = re.search(r"(?:^|<)P<?[A-Z]{3}(.+)$", line)
    if not match or "<<" not in match.group(1):
        return {}
    surname, given_names = match.group(1).split("<<", 1)
    result = {
        "surname": _name(surname),
        "given_names": _name(given_names),
    }
    return {key: value for key, value in result.items() if value}


def extract_raw_document(lines: list[str]) -> dict[str, str]:
    """Recover core TD3-positioned values without claiming checksum validity."""
    cleaned = [clean_line(line) for line in lines]
    name_line = next((line for line in cleaned if line.startswith("P")), None)
    data_line = next((line for line in cleaned if re.match(
        r"^[A-Z0-9<]{9}[0-9O][A-Z<]{3}\d{6}[0-9O][MFX<]\d{6}", line
    )), None)
    if not name_line or not data_line:
        return {}
    result = {
        "document_type": "TD3",
        "issuing_state": name_line[2:5],
        "document_number": data_line[0:9].replace("<", ""),
        "nationality": data_line[10:13].replace("<", ""),
        "birth_date": _date_yyMMdd(data_line[13:19], "birth") or "",
        "sex": data_line[20].replace("<", "X"),
        "expiry_date": _date_yyMMdd(data_line[21:27], "expiry") or "",
    }
    return {key: value for key, value in result.items() if value}


def _valid(value: str, digit: str) -> bool:
    return digit.isdigit() and check_digit(value) == digit


def parse_td3(lines: list[str]) -> dict:
    lines = normalize_lines(lines)
    if len(lines) != 2 or any(len(x) != 44 for x in lines):
        raise ValueError("MRZ должна содержать ровно 2 строки по 44 символа")
    l1, l2 = lines
    if not l1.startswith("P"):
        raise ValueError("Не найден тип документа P (TD3 паспорт)")
    names = l1[5:].split("<<", 1)
    checks = {
        "document_number": _valid(l2[0:9], l2[9]),
        "birth_date": _valid(l2[13:19], l2[19]),
        "expiry_date": _valid(l2[21:27], l2[27]),
        "optional_data": _valid(l2[28:42], l2[42]),
        "composite": _valid(l2[0:10] + l2[13:20] + l2[21:43], l2[43]),
    }
    return {
        "lines": lines,
        "format": "TD3",
        "document_type": l1[0:2].replace("<", ""),
        "issuing_state": l1[2:5],
        "surname": _name(names[0]),
        "given_names": _name(names[1] if len(names) > 1 else ""),
        "document_number": l2[0:9].replace("<", ""),
        "nationality": l2[10:13],
        "birth_date": _date_yyMMdd(l2[13:19], "birth"),
        "birth_date_raw": l2[13:19],
        "sex": l2[20].replace("<", "X"),
        "expiry_date": _date_yyMMdd(l2[21:27], "expiry"),
        "expiry_date_raw": l2[21:27],
        "optional_data": l2[28:42].rstrip("<"),
        "checks": checks,
        "all_required_valid": all(checks[k] for k in ("document_number", "birth_date", "expiry_date", "composite")),
    }


def repair_line2(line: str, max_ambiguous: int = 8) -> tuple[str, list[dict]]:
    """Checksum-guided conservative repair; never changes more than ambiguous glyphs."""
    line = clean_line(line)
    if len(line) != 44: return line, []
    positions = [i for i, ch in enumerate(line) if ch in AMBIGUITIES]
    # Check digits and nationality/sex are constrained separately; cap combinatorics.
    positions = positions[:max_ambiguous]
    try:
        if parse_td3(["P<UTO" + "TEST<<USER".ljust(39, "<"), line])["all_required_valid"]:
            return line, []
    except ValueError: pass
    best = None
    for mask in product((0, 1), repeat=len(positions)):
        if not any(mask): continue
        chars = list(line)
        changes = []
        for use, pos in zip(mask, positions):
            if use:
                old = chars[pos]; chars[pos] = AMBIGUITIES[old]
                changes.append({"position": pos, "from": old, "to": chars[pos], "reason": "CHECKSUM_GUIDED"})
        candidate = "".join(chars)
        try:
            parsed = parse_td3(["P<UTO" + "TEST<<USER".ljust(39, "<"), candidate])
        except ValueError: continue
        score = sum(parsed["checks"].values()) * 10 - len(changes)
        if best is None or score > best[0]: best = (score, candidate, changes)
    return (best[1], best[2]) if best and best[0] >= 39 else (line, [])
