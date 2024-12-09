from abc import ABC, abstractmethod
from typing import List


class ITagsRetriever(ABC):
    @abstractmethod
    def load_model(self) -> None:
        """
        Loads the model from disk or other storage. This method is called before generating tags.
        """

    @abstractmethod
    def classify_cv(self, cv_text: str) -> str:
        """
        Applies class to CV to retrieve tags relevant for this particular class in the future.
        :param cv_text: text of the input cv.
        :return: str representation of the class.
        """

    @abstractmethod
    def retrieve_tags_by_class(self, cv_class: str) -> List[str]:
        """
        Retrieves from API / scrapes relevant tags.
        :param cv_class: str representation of the class.
        :return: list of tags retrieved (if class is unknown or unable to retrieve return [])
        """
