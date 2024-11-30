import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

from src.service.models.rag_information.interfaces import ITagsRetriever
from src.service.utils.exceptions.service_error import ServiceError
from src.service.utils.logging.loggers.console_logger import ConsoleLogger


class ZeroShotRagTagsRetriever(ITagsRetriever):
    _TAGS_SOURCE_URL = "https://easyoffer.ru/analytic"

    _PROFESSIONS = [
        "Machine Learning Engineer",
        "Python Developer",
        "Data Scientist",
        "Frontend Developer",
        "Java Developer",
    ]

    _MODEL_PATH = Path(__file__).parents[4] / "models/bart"

    _NON_CLASSIFIED = "Non classified"

    def __init__(self):
        self._classifier: Optional[pipeline] = None

        self._logger = ConsoleLogger()

    def load_model(self) -> None:
        try:
            self._classifier = pipeline(
                "zero-shot-classification",
                model=self._MODEL_PATH.as_posix(),
                device="gpu",
            )
            self._logger.info("Zero-shot classification model loaded successfully.")
        except Exception as e:
            self._logger.critical(f"Error loading model: {e}")
            raise ServiceError("Failed to load the zero-shot classification model.") from e

    def classify_cv(self, cv_text: str) -> str:
        if not self._classifier:
            raise ValueError("Model not loaded. Call load_model() before classify_cv().")
        try:
            result = self._classifier(cv_text, self._PROFESSIONS)
            predicted_class: str = result["labels"][0]
            self._logger.info(f"CV classified as: {predicted_class}")
            return predicted_class
        except Exception as e:
            self._logger.error(f"Error during classification: {e}")
            return self._NON_CLASSIFIED

    def retrieve_tags_by_class(self, cv_class: str) -> List[str]:
        if cv_class not in self._PROFESSIONS or cv_class == self._NON_CLASSIFIED:
            return []

        response = requests.get(f"{self._TAGS_SOURCE_URL}/{self._to_snake_case(cv_class)}")
        soup = BeautifulSoup(response.text, "html.parser")

        try:
            table = soup.find_all("table", {"class": "table table-sm table-bordered table-hover"})[0]
            tables = pd.read_html(table.prettify())
            if tables:
                df = tables[0]
                if "Навык" in df.columns:
                    return df["Навык"].tolist()

            return []
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return []

    @staticmethod
    def _to_snake_case(name: str) -> str:
        return re.sub("(?!^)([A-Z]+)", r"_\1", name).lower().replace(" ", "")
