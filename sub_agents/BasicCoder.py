class BasicCoder:
    def __init__(self):
        self.messages = []
        self.prev_codes = []

    def refine(self, error_message):
        return None

    def _sanitize(self, code):
        start_token = "```python"
        end_token = "```"

        start_idx = code.find(start_token)
        if start_idx == -1:
            return code

        sub_str = code[start_idx + len(start_token):]

        end_idx = sub_str.find(end_token)
        if end_idx == -1:
            return sub_str.strip()

        return sub_str[:end_idx].strip()
