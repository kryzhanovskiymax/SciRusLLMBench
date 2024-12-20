from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any
import random


class TopicBenchmark:
    """
    Класс для тестирования языковых моделей на задаче распознавания тем.
    """
    def __init__(self, dataset: List[Dict[str, Any]]):
        """
        Инициализация с данными для тестирования.

        :param dataset: Список объектов с текстом, основной темой и альтернативными темами.
        """
        self.dataset = dataset

    def _create_qa(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Создание промпта для LLM из объекта данных.

        :param data: Объект данных с текстом, основной и альтернативными темами.
        :return: Словарь с промптом и правильным ответом.
        """
        text = data["text"]
        all_topics = [data["dominant_topic"]] + data["negative_topics"]
        random.shuffle(all_topics)
        options = "\n".join([f"{i + 1}. {topic}" for i, topic in enumerate(all_topics)])

        prompt = (
            f"К какой теме относится данный текст:\n\n"
            f"{text}\n\n"
            f"Варианты ответа:\n{options}"
        )

        correct_answer = str(all_topics.index(data["dominant_topic"]) + 1)

        return {
            "prompt": prompt,
            "answer": correct_answer
        }

    def create_eval_dataset(self) -> List[Dict[str, str]]:
        """
        Создает массив промптов и правильных ответов для оценки.

        :return: Список объектов с промптом и правильным ответом.
        """
        eval_dataset = [self._create_qa(data) for data in self.dataset]
        return eval_dataset

    def evaluate(self, model_name: str) -> float:
        """
        Оценивает языковую модель на основе метрики accuracy.

        :param model_name: Название модели для загрузки из Transformers.
        :return: Значение accuracy.
        """
        # Загрузка модели
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Создание датасета для оценки
        eval_dataset = self.create_eval_dataset()
        correct = 0

        for qa in eval_dataset:
            prompt = qa["prompt"]
            correct_answer = qa["answer"]

            # Генерация ответа
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Проверка правильности ответа
            if correct_answer in generated_text:
                correct += 1

        # Расчет accuracy
        accuracy = correct / len(eval_dataset)
        return accuracy
