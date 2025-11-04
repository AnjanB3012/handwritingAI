import json
import random
from InData import InData

class SystemManager:
    def __init__(self, config_path: str, input_data_path: str, needed_data_file: str):
        self.id_counter = 1
        self.config_path = config_path
        self.input_data_path = input_data_path
        self.needed_data_file = needed_data_file
        self.config = self.load_config()
        self.input_data = [] # list of InData objects
        self.input_data = self.load_input_data()
        self.needed_data = self.load_needed_data()
        self.active_state = self.config.get("active_state", False)
        self.name = self.config.get("name")
        self.active_entry_id = None

    def get_name(self):
        return self.name

    def save_system(self):
        self.save_config()
        self.save_input_data()
        self.save_needed_data()

    def set_active_state(self, active_state: bool):
        self.active_state = active_state
        self.config["active_state"] = active_state
        self.save_config()
    def get_active_state(self):
        return self.active_state

    def get_id_counter(self):
        self.increment_id_counter()
        return self.id_counter - 1

    def set_id_counter(self, id_counter: int):
        self.id_counter = id_counter

    def increment_id_counter(self):
        self.id_counter += 1

    def save_input_data(self):
        with open(self.input_data_path, 'w') as file:
            json.dump([data.to_json() for data in self.input_data], file)

    def load_input_data(self):
        try:
            with open(self.input_data_path, 'r') as file:
                content = file.read().strip()
                if not content:
                    return []
                data = [InData.from_json(data) for data in json.loads(content)]
                if len(data) > 0:
                    self.id_counter = data[-1].get_id() + 1
                return data
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def load_config(self):
        with open(self.config_path, 'r') as file:
            return json.load(file)

    def save_config(self):
        with open(self.config_path, 'w') as file:
            json.dump(self.config, file)

    def load_needed_data(self):
        try:
            with open(self.needed_data_file, 'r') as file:
                content = file.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def save_needed_data(self):
        with open(self.needed_data_file, 'w') as file:
            json.dump(self.needed_data, file)
    
    def get_needed_data(self):
        return self.needed_data
    
    def remove_needed_data(self, id):
        # Handle both string and integer keys
        if isinstance(id, int):
            id = str(id)
        del self.needed_data[id]
        self.save_system()
    
    def get_next_needed_data(self):
        if len(self.needed_data) == 0:
            return None
        
        # If we have an active entry and it still exists in needed_data, return it
        # This prevents text from changing when frontend/app polls
        if self.active_entry_id is not None:
            # Convert active_entry_id to string for dictionary lookup
            active_key = str(self.active_entry_id)
            if active_key in self.needed_data:
                entry_text = self.needed_data[active_key]
                return {
                    "id": self.active_entry_id,
                    "entry_text": entry_text
                }
        
        # Otherwise, get a random key from needed_data (new entry)
        keys = list(self.needed_data.keys())
        random_key = random.choice(keys)
        entry_text = self.needed_data[random_key]
        
        # Set this as the active entry id
        try:
            id_value = int(random_key)
        except (ValueError, TypeError):
            id_value = random_key
        
        self.active_entry_id = id_value
        
        return {
            "id": id_value,
            "entry_text": entry_text
        }
    
    def update_InData(self, stroke_data: dict):
        # Use the active entry id
        if self.active_entry_id is None:
            raise ValueError("No active entry id set. Call get_next_needed_data first.")
        
        # Convert active_entry_id to string if it's an int (JSON keys are strings)
        key = str(self.active_entry_id) if isinstance(self.active_entry_id, int) else self.active_entry_id
        
        # Find the entry text from needed_data using the entry_id
        if key not in self.needed_data:
            raise ValueError(f"Entry with id {self.active_entry_id} not found in needed_data")
        
        entry_text = self.needed_data[key]
        
        # Create InData object with stroke data
        in_data = InData(self.get_id_counter(), entry_text)
        in_data.set_stroke_data(stroke_data)
        
        # Add to input_data
        self.input_data.append(in_data)
        
        # Remove from needed_data (this also saves needed_data)
        # This ensures the entry is removed and won't appear again
        if key in self.needed_data:
            del self.needed_data[key]
        
        # Clear active entry id after removal
        self.active_entry_id = None
        
        # Save system (this will save both input_data and needed_data)
        self.save_system()
    
    def get_active_entry_id(self):
        return self.active_entry_id