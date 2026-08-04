COUNTRIES = {
    "AUTO": {"name": "Автоопределение", "scripts": "MRZ / Latin"},
    "CHN": {"name": "Китай", "scripts": "Chinese + Latin", "risk": "Порядок имени и romanization"},
    "UZB": {"name": "Узбекистан", "scripts": "Latin", "risk": "Апострофы и поколения шаблонов"},
    "RUS": {"name": "Россия", "scripts": "Cyrillic + Latin", "risk": "Расхождение транслитерации VIZ/MRZ"},
    "TUR": {"name": "Турция", "scripts": "Turkish Latin", "risk": "Диакритика в VIZ"},
    "KAZ": {"name": "Казахстан", "scripts": "Kazakh/Russian + Latin", "risk": "Казахские буквы и ИИН"},
    "TJK": {"name": "Таджикистан", "scripts": "Tajik Cyrillic + Latin", "risk": "Дефицит шаблонов, строгий review"},
}


# Operator-facing names for identifiers that can be printed in a passport's
# visual zone. They are deliberately not called "ИНН": a tax identifier is not
# a standard ICAO TD3 passport field.
PERSONAL_NUMBER_LABELS = {
    "CHN": "Персональный номер",
    "UZB": "ПИНФЛ / JSHSHIR",
    "RUS": "Персональный номер",
    "TUR": "T.C. Kimlik No",
    "KAZ": "ИИН",
    "TJK": "Персональный номер",
}


def personal_number_label(country_code: str | None) -> str:
    """Return the passport-appropriate label for a national identifier."""
    return PERSONAL_NUMBER_LABELS.get(country_code or "", "Персональный номер")
