from passport_mvp.viz import extract_viz


def item(text, x1, y1, x2, y2, score=.95):
    return {"text": text, "box": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], "score": score}


def test_extracts_inline_and_spatial_values():
    objects = [
        item("Surname / Фамилия", 10, 10, 180, 35),
        item("ИВАНОВ", 220, 10, 350, 35),
        item("Date of birth: 01.02.1990", 10, 60, 320, 85),
        item("Place of birth", 10, 110, 180, 135),
        item("MOSCOW", 10, 145, 180, 170),
    ]
    fields, text = extract_viz(objects)
    assert fields["surname_viz"].value == "ИВАНОВ"
    assert fields["birth_date"].value == "01.02.1990"
    assert fields["birth_place"].value == "MOSCOW"
    assert len(text) == 5
