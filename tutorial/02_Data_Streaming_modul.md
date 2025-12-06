# 02 - Data Streaming Modul (RabbitMQ)

## Mi a Data Streaming és miért van rá szükség?

### A Probléma

Amikor egy gépi tanulási rendszert építünk, gyakran találkozunk ezekkel a kihívásokkal:

1. **Nagy adatmennyiség** - A frontend nem tudja egyszerre a memóriába tölteni az összes adatot
2. **Aszinkron feldolgozás** - A predikció időigényes lehet, nem akarjuk blokkolni a felhasználót
3. **Skálázhatóság** - Több backend instance dolgozhat párhuzamosan ugyanazon az adaton
4. **Megbízhatóság** - Ha a backend leáll, az üzenetek nem vesznek el

### A Megoldás: Message Queue (RabbitMQ)

A **RabbitMQ** egy message broker, ami producer (küldő) és consumer (fogadó) között közvetít üzeneteket.

```
Frontend (Producer) → RabbitMQ Queue → Backend (Consumer)
```

---

## RabbitMQ Alapfogalmak

### 1. **Producer (Küldő)**
- Üzeneteket küld a queue-ba
- Nem tudja ki fogja fogadni az üzenetet
- Példa: Frontend elküldi a prediktálandó adatokat

### 2. **Queue (Sor)**
- Tárolja az üzeneteket
- FIFO (First In, First Out) elven működik
- Persistence: megőrzi az üzeneteket még restart esetén is

### 3. **Consumer (Fogadó)**
- Fogadja és feldolgozza az üzeneteket
- Visszajelez (ACK), ha feldolgozta
- Példa: Backend fogadja az adatokat és prediktál

### 4. **Exchange**
- Routing logika
- Alapértelmezett esetben nem használjuk (direkt routing)

---

## SOLID Elvek Alkalmazása

### Melyik SOLID elveket használjuk?

#### 1. **Single Responsibility Principle (SRP)**
```python
# ❌ ROSSZ - Egy osztály mindent csinál
class RabbitMQ:
    def connect(self): ...
    def send_message(self): ...
    def receive_message(self): ...
    def disconnect(self): ...

# ✅ JÓ - Minden osztály egy felelősséggel
class RabbitMQConnection:
    def connect(self): ...
    def disconnect(self): ...

class Producer:
    def send(self, message): ...

class Consumer:
    def receive(self): ...
```

#### 2. **Dependency Inversion Principle (DIP)**
```python
# Dependency injection - a connection-t kívülről kapja
class Producer:
    def __init__(self, connection: BaseConnection):
        self.connection = connection
```

#### 3. **Open/Closed Principle (OCP)**
```python
# Ha később más message broker-t akarunk (pl. Kafka),
# csak új implementációt írunk, a kódot nem módosítjuk
class BaseConnection(ABC):
    @abstractmethod
    def connect(self): pass

class RabbitMQConnection(BaseConnection):
    def connect(self): ...

class KafkaConnection(BaseConnection):  # Jövőbeli bővítés
    def connect(self): ...
```

---

## Implementáció Lépésről Lépésre

### Lépés 1: Base Connection Interfész

**Fájl:** `app/streaming/base_connection.py`

**Miért csináljuk?**
- Absztrakció létrehozása (SOLID: DIP)
- Könnyű tesztelhetőség (mock objektumok)
- Rugalmasság (más message broker támogatása)

```python
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
```

**Mit tanultunk?**
- Abstract base class definiálása Python-ban
- Type hints használata (→ tisztább kód)
- Docstring írása (→ dokumentáció)

---

### Lépés 2: RabbitMQ Connection Implementáció

**Fájl:** `app/streaming/rabbitmq_connection.py`

**Miért csináljuk?**
- Konkrét RabbitMQ implementáció
- Hibakezelés (reconnection logic)
- Konfigurálhatóság (host, port, credentials)

```python
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
```

**Tesztelés:**

Hozzuk létre a `app/streaming/__init__.py` fájlt:
```python
from .base_connection import BaseConnection
from .rabbitmq_connection import RabbitMQConnection

__all__ = ['BaseConnection', 'RabbitMQConnection']
```

Teszteljük a terminálban:
```python
# Python interpreter
from app.streaming import RabbitMQConnection

conn = RabbitMQConnection(host="localhost", port=5672)
if conn.connect():
    print("Sikeres kapcsolat!")
    conn.disconnect()
```

**Fontos:** RabbitMQ-nak futnia kell! Docker-rel:
```bash
docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management
```

---

### Lépés 3: Producer (Üzenetküldő)

**Fájl:** `app/streaming/producer.py`

**Miért csináljuk?**
- Üzenetek küldése a queue-ba
- Queue deklarálás (persistence)
- Típusos API (JSON serialization)

```python
import json
import logging
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
```

**Tesztelés:**

```python
from app.streaming import RabbitMQConnection, Producer

# Kapcsolat létrehozása
conn = RabbitMQConnection()
conn.connect()

# Producer létrehozása
producer = Producer(conn)

# Üzenet küldése
message = {
    "data": [[5.1, 3.5, 1.4, 0.2]],
    "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
}
producer.send_message("predictions_queue", message)

conn.disconnect()
```

---

### Lépés 4: Consumer (Üzenetfogadó)

**Fájl:** `app/streaming/consumer.py`

**Miért csináljuk?**
- Üzenetek fogadása a queue-ból
- Acknowledgment (ACK) kezelése
- Callback pattern (feldolgozó függvény)

```python
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
```

**Tesztelés:**

Terminal 1 - Producer:
```python
from app.streaming import RabbitMQConnection, Producer

conn = RabbitMQConnection()
conn.connect()
producer = Producer(conn)

for i in range(5):
    producer.send_message("test_queue", {"id": i, "value": i * 10})

conn.disconnect()
```

Terminal 2 - Consumer:
```python
from app.streaming import RabbitMQConnection, Consumer

conn = RabbitMQConnection()
conn.connect()
consumer = Consumer(conn)

def process_message(message):
    print(f"Feldolgozás: {message}")

# Egyetlen üzenet
msg = consumer.get_message("test_queue")
print(msg)

# Vagy folyamatos figyelés
# consumer.start_consuming("test_queue", process_message)

conn.disconnect()
```

---

## Frissítés: `app/streaming/__init__.py`

```python
from .base_connection import BaseConnection
from .rabbitmq_connection import RabbitMQConnection
from .producer import Producer
from .consumer import Consumer

__all__ = [
    'BaseConnection',
    'RabbitMQConnection',
    'Producer',
    'Consumer'
]
```

---

## SOLID Elvek Összefoglalása

### ✅ Mit értünk el?

1. **Single Responsibility (SRP)**
   - `RabbitMQConnection` → csak kapcsolat
   - `Producer` → csak küldés
   - `Consumer` → csak fogadás

2. **Open/Closed (OCP)**
   - Új message broker (pl. Kafka) hozzáadható módosítás nélkül
   - Csak új `BaseConnection` implementáció kell

3. **Liskov Substitution (LSP)**
   - Bármelyik `BaseConnection` implementáció használható

4. **Interface Segregation (ISP)**
   - Kis, specifikus interfészek (connect, disconnect, get_channel)

5. **Dependency Inversion (DIP)**
   - Producer és Consumer függ `BaseConnection` absztrakciótól
   - Dependency injection használata

---

## Következő Lépés

Most, hogy van egy tiszta, SOLID alapú data streaming modulunk, a következő tutorialban létrehozzuk a **ML Backend modult**, ahol:
- Model training service-t építünk
- Prediction service-t készítünk
- FastAPI endpoints-okat definiálunk
- RabbitMQ-t integrálunk

👉 Folytasd a `03_ML_Backend_modul.md` fájllal!
