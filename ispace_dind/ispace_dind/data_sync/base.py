from abc import ABC, abstractmethod

class DataSync(ABC):

    def __init__(self):
        pass
    
    @abstractmethod
    def assosiate_data(self, data):
        pass