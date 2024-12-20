from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer


class Labeller:
    """
    Класс для генерации названия темы на основе ключевых слов.
    """
    def __init__(self, model_name: str):
        """
        Инициализация с базовой моделью для генерации текста.

        :param model_name: Название модели для загрузки из Transformers.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def label(self, words: List[str]) -> str:
        """
        Генерация названия темы на основе списка слов.

        :param words: Массив слов для анализа.
        :return: Название темы.
        """
        prompt = f"Name this topic based on these keywords: {', '.join(words)}"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=50, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
