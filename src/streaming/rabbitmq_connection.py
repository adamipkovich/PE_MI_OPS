import pika
import time
import logging
from typing import Optional
from .base_connection import BaseConnection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RabbitMQConnection(BaseConnection):
    """
    RabbitMQ specifikus connection implementáció.
    
    SOLID elvek:
    - SRP: Csak a kapcsolat kezelése
    - LSP: Helyettesíthető a BaseConnection-nel
    """
    
    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 5672,
        username: str = "guest",
        password: str = "guest",
        retry_delay: float = 0.5,
        max_retries: int = 10
    ):
        """
        RabbitMQ connection inicializálása.
        
        Args:
            host: RabbitMQ server címe
            port: RabbitMQ server portja
            username: Felhasználónév
            password: Jelszó
            retry_delay: Újrapróbálkozás késleltetése (másodperc)
            max_retries: Maximum újrapróbálkozások száma
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        
        self._connection: Optional[pika.BlockingConnection] = None
        self._channel: Optional[pika.channel.Channel] = None
    
    def connect(self) -> bool:
        """
        Kapcsolódás RabbitMQ-hoz retry logikával.
        
        Returns:
            True ha sikeres, False ha nem
        """
        credentials = pika.PlainCredentials(self.username, self.password)
        parameters = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            credentials=credentials,
            heartbeat=600,  # 10 perc heartbeat
            blocked_connection_timeout=300  # 5 perc timeout
        )
        
        retries = 0
        while retries < self.max_retries:
            try:
                self._connection = pika.BlockingConnection(parameters)
                self._channel = self._connection.channel()
                self._channel.basic_qos(prefetch_count=1)
                
                logger.info(f"✅ Sikeres kapcsolat RabbitMQ-hoz: {self.host}:{self.port}")
                return True
                
            except pika.exceptions.AMQPConnectionError as e:
                retries += 1
                logger.warning(
                    f"⚠️ RabbitMQ kapcsolat sikertelen (próba {retries}/{self.max_retries}): {e}"
                )
                if retries < self.max_retries:
                    time.sleep(self.retry_delay)
        
        logger.error(f"❌ RabbitMQ kapcsolat véglegesen sikertelen {self.max_retries} próbálkozás után")
        return False
    
    def disconnect(self) -> None:
        """Kapcsolat lezárása."""
        try:
            if self._channel and not self._channel.is_closed:
                self._channel.close()
            if self._connection and not self._connection.is_closed:
                self._connection.close()
            logger.info("🔌 RabbitMQ kapcsolat lezárva")
        except Exception as e:
            logger.error(f"❌ Hiba a kapcsolat lezárásakor: {e}")
    
    def is_connected(self) -> bool:
        """Kapcsolat állapotának ellenőrzése."""
        return (
            self._connection is not None and 
            not self._connection.is_closed and
            self._channel is not None and
            not self._channel.is_closed
        )
    
    def get_channel(self) -> Optional[pika.channel.Channel]:
        """Channel objektum visszaadása."""
        if not self.is_connected():
            logger.warning("⚠️ Nincs aktív kapcsolat!")
            return None
        return self._channel
