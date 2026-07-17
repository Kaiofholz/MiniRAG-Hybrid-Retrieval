class AnswerCache:
    def __init__(self):
        self.cache = {}

    def get(self, question):
        return self.cache.get(question)

    def save(self, question, result):
        self.cache[question] = result

    def clear(self):
        self.cache.clear()

    def __len__(self):
        return len(self.cache)

class LLMCache:
    def __init__(self):
        self.cache = {}

    def make_key(self, prompt):
        return prompt.strip()

    def get(self, prompt):
        key = self.make_key(prompt)
        return self.cache.get(key)

    def save(self, prompt, answer):
        key = self.make_key(prompt)
        self.cache[key] = answer

    def clear(self):
        self.cache.clear()

    def __len__(self):
        return len(self.cache)
