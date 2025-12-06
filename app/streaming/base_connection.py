from abc import ABC, abstractmethod
from typing import Any, Optional

class BaseConnection(ABC):
    """
    Absztrakt base osztály message broker kapcsolatokhoz.
    
    SOLID elvek:
    - DIP: Magas szintű modulok függnek ettől az absztrakciótól
    - OCP: Új connection típusok hozzáadhatók módosítás nélkül
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """Kapcsolódás a message broker-hez."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Lecsatlakozás a message broker-től."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Ellenőrzi, hogy van-e aktív kapcsolat."""
        pass
    
    @abstractmethod
    def get_channel(self) -> Any:
        """Visszaadja a channel objektumot."""
        pass
