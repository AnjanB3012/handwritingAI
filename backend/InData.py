class InData:
    def __init__(self, id: int, entry_text: str):
        self.id = id
        self.entry_text = entry_text
        self.stroke_data = None

    def get_id(self):
        return self.id

    def get_entry_text(self):
        return self.entry_text

    def set_id(self, id: int):
        self.id = id

    def set_entry_text(self, entry_text: str):
        self.entry_text = entry_text

    def get_stroke_data(self):
        return self.stroke_data

    def set_stroke_data(self, stroke_data: dict):
        self.stroke_data = stroke_data

    def to_json(self):
        return {
            "id": self.id,
            "entry_text": self.entry_text,
            "stroke_data": self.stroke_data
        }

    @classmethod
    def from_json(cls, json_data: dict):
        obj = cls(json_data["id"], json_data["entry_text"])
        obj.stroke_data = json_data.get("stroke_data")
        return obj