import json
import logging
from typing import Dict, Any, Callable, Optional
from .base_connection import BaseConnection

logger = logging.getLogger(__name__)

class Consumer:
    """
    Üzeneteket fogad a RabbitMQ queue-ból.
    
    SOLID elvek:
    - SRP: Csak üzenetfogadás
    - DIP: Függ a BaseConnection absztrakciótól
    """
    
    def __init__(self, connection: BaseConnection):
        """
        Consumer inicializálása.
        
        Args:
            connection: Message broker kapcsolat (dependency injection)
        """
        self.connection = connection
    
    def get_message(self, queue_name: str, auto_ack: bool = False) -> Optional[Dict[str, Any]]:
        """
        Egyetlen üzenet lekérése a queue-ból (non-blocking).
        
        Args:
            queue_name: A queue neve
            auto_ack: Automatikus acknowledgment
            
        Returns:
            Az üzenet dictionary-ként, vagy None ha nincs üzenet
        """
        try:
            channel = self.connection.get_channel()
            if channel is None:
                logger.error("❌ Nincs aktív kapcsolat!")
                return None
            
            # Egyetlen üzenet lekérése
            method_frame, header_frame, body = channel.basic_get(queue_name, auto_ack=auto_ack)
            
            if method_frame is None:
                logger.info("📭 Nincs üzenet a queue-ban")
                return None
            
            # JSON deserialization
            message = json.loads(body.decode('utf-8'))
            
            # Manual ACK ha nem auto
            if not auto_ack:
                channel.basic_ack(delivery_tag=method_frame.delivery_tag)
            
            logger.info(f"📥 Üzenet fogadva: {queue_name}")
            return message
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parse hiba: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Üzenet fogadás sikertelen: {e}")
            return None
    
    def start_consuming(
        self, 
        queue_name: str, 
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Folyamatos üzenet figyelés (blocking).
        
        Args:
            queue_name: A queue neve
            callback: Feldolgozó függvény (message-t kap paraméterként)
        """
        try:
            channel = self.connection.get_channel()
            if channel is None:
                logger.error("❌ Nincs aktív kapcsolat!")
                return
            
            def on_message(ch, method, properties, body):
                """Belső callback wrapper."""
                try:
                    message = json.loads(body.decode('utf-8'))
                    logger.info(f"📥 Üzenet fogadva: {queue_name}")
                    
                    # User callback meghívása
                    callback(message)
                    
                    # ACK
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                    
                except Exception as e:
                    logger.error(f"❌ Üzenet feldolgozás sikertelen: {e}")
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            
            # Queue fogyasztás indítása
            channel.basic_consume(
                queue=queue_name,
                on_message_callback=on_message
            )
            
            logger.info(f"👂 Figyelés indítva: {queue_name}")
            logger.info("⏹️ Leállítás: CTRL+C")
            channel.start_consuming()
            
        except KeyboardInterrupt:
            logger.info("🛑 Fogyasztás leállítva")
        except Exception as e:
            logger.error(f"❌ Consuming hiba: {e}")
