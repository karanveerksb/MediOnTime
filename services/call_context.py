from typing import List, Optional

class CallContext:
    def __init__(self, system_message: str, medication: str, dosage: str, schedule: dict):
        self.system_message = system_message
        self.medication = medication
        self.dosage = dosage
        self.schedule = schedule
        self.user_context = []