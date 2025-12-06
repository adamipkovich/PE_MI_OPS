import json
import logging
import pika
from typing import Dict, Any
from .base_connection import BaseConnection

logger = logging.getLogger(__name__)

class Producer:
    """
    Üzeneteket küld a RabbitMQ queue-ba.
    
    SOLID elvek:
    - SRP: Csak üzenetküldés
    - DIP: Függ a BaseConnection absztrakciótól
    """
    
    def __init__(self, connection: BaseConnection):
        """
        Producer inicializálása.
        
        Args:
            connection: Message broker kapcsolat (dependency injection)
        """
        self.connection = connection
    
    def declare_queue(self, queue_name: str, durable: bool = True) -> bool:
        """
        Queue létrehozása/deklarálása.
        
        Args:
            queue_name: A queue neve
            durable: Megőrzi-e az üzeneteket restart esetén
            
        Returns:
            True ha sikeres
        """
        try:
            channel = self.connection.get_channel()
            if channel is None:
                logger.error("❌ Nincs aktív kapcsolat!")
                return False
            
            channel.queue_declare(queue=queue_name, durable=durable)
            logger.info(f"📦 Queue deklarálva: {queue_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Queue deklarálás sikertelen: {e}")
            return False
    
    def send_message(
        self, 
        queue_name: str, 
        message: Dict[str, Any],
        declare_queue: bool = True
    ) -> bool:
        """
        Üzenet küldése a queue-ba.
        
        Args:
            queue_name: A queue neve
            message: Az üzenet (dictionary)
            declare_queue: Automatikusan deklarálja-e a queue-t
            
        Returns:
            True ha sikeres
        """
        try:
            if declare_queue:
                self.declare_queue(queue_name)
            
            channel = self.connection.get_channel()
            if channel is None:
                logger.error("❌ Nincs aktív kapcsolat!")
                return False
            
            # JSON serialization
            message_json = json.dumps(message)
            
            # Üzenet küldése
            channel.basic_publish(
                exchange='',  # Default exchange
                routing_key=queue_name,
                body=message_json.encode('utf-8'),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Persistent message
                )
            )
            
            logger.info(f"📤 Üzenet elküldve: {queue_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Üzenet küldés sikertelen: {e}")
            return False
