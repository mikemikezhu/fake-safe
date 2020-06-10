from abc import ABC, abstractmethod

"""
Abstract Model Creator
"""


class AbstractModelCreator(ABC):

    @abstractmethod
    def create_model(self):
        raise NotImplementedError('Abstract class shall not be implemented')
