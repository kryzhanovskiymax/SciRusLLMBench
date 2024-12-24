from typing import List, Dict, Optional, Any
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import random
from bench.labeller import Labeller


class TopicInstructs:
    """
    Класс для тематического моделирования и анализа текстов.
    """
    def __init__(self,
                 texts: List[str],
                 num_topics: int = 5,
                 max_features: int = 1000,
                 labeller: Optional[Labeller] = None):
        """
        Инициализация модели и выполнение тематического моделирования.

        :param texts: Список текстов для анализа.
        :param num_topics: Количество тем для моделирования.
        :param max_features: Максимальное количество уникальных слов для анализа (для CountVectorizer).
        :param labeller: Объект класса Labeller для генерации названий тем (опционально).
        """
        self.texts = texts
        self.num_topics = num_topics
        self.labeller = labeller
        self.vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
        self.model = LatentDirichletAllocation(n_components=num_topics, random_state=42)

        # Выполнение обучения модели
        self.document_term_matrix = self.vectorizer.fit_transform(texts)
        self.model.fit(self.document_term_matrix)
        self.topics = self._extract_topics()

    def _extract_topics(self, num_words: int = 10) -> Dict[int, List[str]]:
        """
        Получение топ-слов для каждой темы.

        :param num_words: Количество слов, описывающих каждую тему.
        :return: Словарь, где ключ — номер темы, значение — список слов.
        """
        words = self.vectorizer.get_feature_names_out()
        topics = {
            i: [words[idx] for idx in topic.argsort()[-num_words:][::-1]]
            for i, topic in enumerate(self.model.components_)
        }
        return topics

    def sample_object(self, k_negatives: int = 3) -> Dict[str, Any]:
        """
        Возвращает случайный текст, соответствующую ему тему и k_negatives наименее вероятных тем.

        :param k_negatives: Количество наименее вероятных тем.
        :return: Словарь с информацией о тексте, темах и их названиях.
        """

        random_idx = random.randint(0, len(self.texts) - 1)
        random_text = self.texts[random_idx]
        text_topic_distribution = self.model.transform(self.document_term_matrix[random_idx])
        dominant_topic_idx = np.argmax(text_topic_distribution)
        least_likely_topics = np.argsort(text_topic_distribution)[0][:-k_negatives - 1:-1]
        dominant_topic_words = self.topics[dominant_topic_idx]
        negative_topics_words = [self.topics[idx] for idx in least_likely_topics]
        result = {
            "text": random_text,
            "dominant_topic": dominant_topic_words,
            "negative_topics": negative_topics_words
        }

        if self.labeller:
            result["dominant_topic_label"] = self.labeller.label(dominant_topic_words)
            result["negative_topics_labels"] = [self.labeller.label(words) for words in negative_topics_words]

        return result
