# 01 - SOLID Alapelvek Bevezetése

## Miért van szükségünk SOLID elvekre?

### Problémák a jelenlegi kóddal

Ha megnézzük a korábbi `back_end.py` fájlunkat, több problémát is találunk:

```python
# Globális változók mindenhol
model = None
client = None
signature = None
rabbit_connection = None
channel = None

# Egy fájl sok mindent csinál:
# - API végpontokat definiál
# - MLflow-hoz csatlakozik
# - RabbitMQ-hoz csatlakozik
# - Modellt tölt be
# - Predikciót végez
```

**Problémák:**
1. ❌ **Nehéz tesztelni** - Globális változók miatt unit tesztek írása nehézkes
2. ❌ **Tight coupling** - Minden szorosan össze van kötve
3. ❌ **Nehéz módosítani** - Ha változtatunk valamit, sok mindent érint
4. ❌ **Nem újrafelhasználható** - Nem tudjuk a komponenseket külön használni
5. ❌ **Nehéz skálázni** - Új funkcionalitás hozzáadása bonyolult

### Mi a megoldás? SOLID Elvek!

A SOLID elvek 5 objektum-orientált tervezési alapelvet foglalnak össze, amelyek segítenek **karbantartható, bővíthető és tesztelhető** kódot írni.

---

## Az 5 SOLID Elv

### 1. **S**ingle Responsibility Principle (Egyszeres Felelősség Elve)

> "Egy osztálynak csak egy oka legyen a változásra"

**Mit jelent?**
- Minden osztály/modul csak egy jól meghatározott feladatot lát el
- Ha változtatni kell, csak egy ok miatt változik

**Példa:**
```python
# ❌ ROSSZ - Egy osztály sok mindent csinál
class ModelHandler:
    def connect_to_mlflow(self): ...
    def load_model(self): ...
    def predict(self, data): ...
    def connect_to_rabbitmq(self): ...
    def send_to_queue(self, data): ...

# ✅ JÓ - Minden osztály egy felelősséggel
class MLflowConnection:
    def connect(self): ...

class ModelService:
    def load_model(self): ...
    def predict(self, data): ...

class RabbitMQConnection:
    def connect(self): ...
    def send(self, data): ...
```

**Előnyök:**
- Könnyebb megérteni a kódot
- Könnyebb tesztelni
- Változtatások csak egy helyen szükségesek

---

### 2. **O**pen/Closed Principle (Nyitva/Zárt Elv)

> "A kód legyen nyitott a bővítésre, de zárt a módosításra"

**Mit jelent?**
- Új funkcionalitást úgy adjunk hozzá, hogy ne kelljen a meglévő kódot módosítani
- Használjunk absztrakciót és öröklődést

**Példa:**
```python
# ❌ ROSSZ - Új model típusnál módosítani kell a kódot
def load_model(model_type):
    if model_type == "sklearn":
        return load_sklearn_model()
    elif model_type == "tensorflow":
        return load_tensorflow_model()
    # Minden új típusnál módosítani kell ezt a függvényt!

# ✅ JÓ - Új model típus hozzáadása módosítás nélkül
class BaseModelLoader(ABC):
    @abstractmethod
    def load(self, model_path): pass

class SklearnModelLoader(BaseModelLoader):
    def load(self, model_path):
        return mlflow.sklearn.load_model(model_path)

class TensorFlowModelLoader(BaseModelLoader):
    def load(self, model_path):
        return mlflow.tensorflow.load_model(model_path)

# Új loader hozzáadása nem érinti a meglévő kódot
```

**Előnyök:**
- Biztonságosabb bővíteni a rendszert
- Kevesebb bug-ot okozunk
- Rugalmasabb architektúra

---

### 3. **L**iskov Substitution Principle (Liskov Helyettesítési Elv)

> "A leszármazott osztályok helyettesíthetők legyenek az ősosztállyal anélkül, hogy a program működése megváltozna"

**Mit jelent?**
- Ha egy osztály leszármazik egy másikból, akkor az ősosztály helyére mindig be lehet helyettesíteni
- A leszármazottnak ugyanúgy kell viselkednie, mint az ősnek

**Példa:**
```python
# ✅ JÓ - Minden connection ugyanúgy működik
class BaseConnection(ABC):
    @abstractmethod
    def connect(self) -> bool: pass
    @abstractmethod
    def disconnect(self) -> None: pass
    @abstractmethod
    def is_connected(self) -> bool: pass

class MLflowConnection(BaseConnection):
    def connect(self) -> bool:
        # MLflow specifikus kapcsolódás
        return True
    
    def disconnect(self) -> None:
        # MLflow specifikus lecsatlakozás
        pass
    
    def is_connected(self) -> bool:
        return self._connected

class RabbitMQConnection(BaseConnection):
    def connect(self) -> bool:
        # RabbitMQ specifikus kapcsolódás
        return True
    
    def disconnect(self) -> None:
        # RabbitMQ specifikus lecsatlakozás
        pass
    
    def is_connected(self) -> bool:
        return self._connection is not None

# Mindkét connection ugyanúgy használható
def setup_connection(conn: BaseConnection):
    if conn.connect():
        print("Sikeres kapcsolat!")
```

**Előnyök:**
- Biztonságos helyettesíthetőség
- Könnyebb refaktorálás
- Kevesebb meglepetés a kódban

---

### 4. **I**nterface Segregation Principle (Interfész Szegregációs Elv)

> "Ne kényszerítsünk egy klienst olyan interfészek implementálására, amit nem használ"

**Mit jelent?**
- Kisebb, specifikus interfészek jobbak, mint egy nagy, mindent tartalmazó
- Minden kliens csak azt kapja, amire szüksége van

**Példa:**
```python
# ❌ ROSSZ - Nagy interfész, amit nem mindenki használ
class ModelService(ABC):
    @abstractmethod
    def train(self): pass
    @abstractmethod
    def predict(self): pass
    @abstractmethod
    def save(self): pass
    @abstractmethod
    def load(self): pass
    @abstractmethod
    def evaluate(self): pass

# Egy predikciós service-nek nem kell train() vagy save()

# ✅ JÓ - Kis, specifikus interfészek
class Trainable(ABC):
    @abstractmethod
    def train(self, X, y): pass

class Predictable(ABC):
    @abstractmethod
    def predict(self, X): pass

class Saveable(ABC):
    @abstractmethod
    def save(self, path): pass

# Service-ek csak azt implementálják, amire szükségük van
class TrainingService(Trainable, Saveable):
    def train(self, X, y): ...
    def save(self, path): ...

class PredictionService(Predictable):
    def predict(self, X): ...
```

**Előnyök:**
- Egyszerűbb implementációk
- Kevesebb szükségtelen kód
- Tisztább API-k

---

### 5. **D**ependency Inversion Principle (Függőség Megfordítási Elv)

> "Függj absztrakciótól, ne konkrét implementációtól"

**Mit jelent?**
- A magas szintű modulok ne függjenek alacsony szintű moduloktól
- Mindkettő függjön absztrakcióktól (interfészektől)
- Az absztrakciók ne függjenek részletektől

**Példa:**
```python
# ❌ ROSSZ - Közvetlen függőség konkrét implementációtól
class PredictionService:
    def __init__(self):
        self.mlflow_client = MlflowClient("http://localhost:5000")  # Hard-coded!
        self.model = None
    
    def predict(self, data):
        # Közvetlenül használja az MLflowClient-et
        self.model = self.mlflow_client.load_model(...)

# ✅ JÓ - Dependency Injection, interfészre támaszkodás
class PredictionService:
    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader  # Absztrakció!
    
    def predict(self, data):
        model = self.model_loader.load()
        return model.predict(data)

# A konkrét implementációt kívülről adjuk át
mlflow_loader = MLflowModelLoader(client)
service = PredictionService(mlflow_loader)
```

**Előnyök:**
- Könnyű tesztelni (mock objektumokkal)
- Rugalmas, cserélhető komponensek
- Kevésbé töredékeny kód

---

## Hogyan alkalmazzuk a projektünkben?

A következő lépésekben **modulonként** fogjuk felépíteni a rendszert, minden lépésben alkalmazva a SOLID elveket:

1. **FastAPI Modul** → *SRP, DIP*
2. **Connection Modul** → *SRP, OCP, LSP, DIP*
3. **Training Modul** → *SRP, ISP, DIP*
4. **Prediction Modul** → *SRP, ISP, DIP*
5. **UI Modul** → *SRP, ISP*

### Előnyök, amiket el fogunk érni:

✅ **Tesztelhető kód** - Unit tesztek írása egyszerű lesz  
✅ **Karbantartható** - Változtatások lokalizáltak  
✅ **Bővíthető** - Új funkciók hozzáadása nem tör meg semmit  
✅ **Újrafelhasználható** - Modulok más projektekben is használhatók  
✅ **Olvasható** - Világos felelősségek, könnyű megérteni  

---

## Következő Lépés

A következő tutorialban elkezdjük építeni a **FastAPI modult**, ahol az API végpontokat **tisztán** fogjuk definiálni, külön választva az üzleti logikától.

👉 Folytasd a `02_FastAPI_modul.md` fájllal!
