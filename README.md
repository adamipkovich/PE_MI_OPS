# 🤖 ML Platform - SOLID Elvekkel Tervezett Machine Learning Rendszer

## 📖 Projekt Áttekintés

Ez a projekt egy **átfogó 2 órás tutorial**, amely bemutatja, hogyan építsünk egy production-ready machine learning platformot a **SOLID tervezési elvek** alkalmazásával.

### 🎯 Főbb Technológiák

- **FastAPI** - REST API backend
- **RabbitMQ** - Message broker (aszinkron kommunikáció)
- **Streamlit** - Interaktív frontend
- **Scikit-learn** - Machine learning
- **Docker** - Konténerizáció
- **UV** - Gyors Python package management

### 🏗️ Architektúra

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Frontend   │────▶│   RabbitMQ   │────▶│   Backend    │
│  (Streamlit) │◀────│  (Messages)  │◀────│  (FastAPI)   │
└──────────────┘     └──────────────┘     └──────────────┘
       │                                          │
       └──────────────────┬───────────────────────┘
                     ML Models
                   (Persistence)
```

---

## 📚 Tutorial Anyagok

A projekt 5 részletes tutorialt tartalmaz:

1. **[01_SOLID_bevezetes.md](tutorial/01_SOLID_bevezetes.md)** - SOLID alapelvek bemutatása
2. **[02_Data_Streaming_modul.md](tutorial/02_Data_Streaming_modul.md)** - RabbitMQ messaging
3. **[03_ML_Backend_modul.md](tutorial/03_ML_Backend_modul.md)** - FastAPI + ML services
4. **[04_Frontend_modul.md](tutorial/04_Frontend_modul.md)** - Streamlit UI komponensek
5. **[05_Docker_setup.md](tutorial/05_Docker_setup.md)** - Docker konténerizáció

Minden tutorial tartalmazza:
- ✅ Problémameghatározás
- ✅ SOLID elvek alkalmazása
- ✅ Lépésről-lépésre implementáció
- ✅ Kód példák
- ✅ Tesztelési útmutató

---

## 🚀 Gyors Start (Docker)

### Előfeltételek

- Docker
- Docker Compose

### Indítás

```bash
# 1. Klónozd a repository-t
git clone <repo-url>
cd MLOPS

# 2. Build és indítás
docker-compose up -d --build

# 3. Ellenőrzés
docker-compose ps
```

### Elérések

- **Frontend (Streamlit)**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Dokumentáció**: http://localhost:8000/docs
- **RabbitMQ Management**: http://localhost:15672 (guest/guest)

---

## 📂 Projekt Struktúra

```
MLOPS/
├── app/                          # Data Streaming modul
│   └── streaming/
│       ├── base_connection.py    # Absztrakt interfész (DIP)
│       ├── rabbitmq_connection.py # RabbitMQ implementáció
│       ├── producer.py           # Üzenetek küldése
│       └── consumer.py           # Üzenetek fogadása
│
├── backend/                      # ML Backend modul
│   ├── main.py                   # FastAPI app
│   ├── services/
│   │   ├── model_storage.py      # Model I/O (SRP)
│   │   ├── training_service.py   # Training logika (SRP, DIP)
│   │   └── prediction_service.py # Prediction logika (SRP, DIP)
│   └── models/
│       └── saved/                # Mentett modellek
│
├── frontend/                     # Streamlit Frontend
│   ├── app.py                    # Main app
│   ├── components/
│   │   ├── model_selector.py    # Model választó (SRP)
│   │   ├── data_uploader.py     # Adat feltöltő (SRP)
│   │   └── results_display.py   # Eredmények (SRP)
│   └── utils/
│       └── api_client.py         # Backend kommunikáció (SRP)
│
├── docker/                       # Docker konfiguráció
│   ├── Dockerfile.backend        # Backend image
│   └── Dockerfile.frontend       # Frontend image
│
├── tutorial/                     # Tutorial anyagok
│   ├── 01_SOLID_bevezetes.md
│   ├── 02_Data_Streaming_modul.md
│   ├── 03_ML_Backend_modul.md
│   ├── 04_Frontend_modul.md
│   └── 05_Docker_setup.md
│
├── data/                         # Adatok
│   └── cars.csv
│
├── docker-compose.yml            # Orchestration
├── requirements-backend.txt      # Backend dependencies
├── requirements-frontend.txt     # Frontend dependencies
└── .dockerignore                 # Docker build optimization
```

---

## 🎓 SOLID Elvek Alkalmazása

### Single Responsibility Principle (SRP)

Minden osztály egy felelősséggel:
- `ModelStorage` → csak I/O műveletek
- `TrainingService` → csak model tanítás
- `PredictionService` → csak predikció
- `Producer` → csak üzenetküldés
- `Consumer` → csak üzenetfogadás

### Open/Closed Principle (OCP)

Rendszer nyitott bővítésre, zárt módosításra:
- Új message broker támogatása (pl. Kafka) új class-szal
- Új model típusok hozzáadása módosítás nélkül

### Liskov Substitution Principle (LSP)

Interfészek helyettesíthetők:
- `BaseConnection` implementációk cserélhetők
- Service-ek mock objektumokkal tesztelhetők

### Interface Segregation Principle (ISP)

Kis, specifikus interfészek:
- `Trainable` és `Predictable` külön interfészek
- UI komponensek csak szükséges metódusokat használnak

### Dependency Inversion Principle (DIP)

Függés absztrakciótól:
- Service-ek dependency injection-nel kapják függőségeiket
- FastAPI függőségek absztrakciók (Depends)

---

## 🔧 Fejlesztés

### Lokális Futtatás (Python környezet)

```bash
# 1. Virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Dependencies telepítése
pip install -r requirements-backend.txt
pip install -r requirements-frontend.txt

# 3. RabbitMQ indítása (Docker)
docker run -d -p 5672:5672 -p 15672:15672 rabbitmq:3-management

# 4. Backend indítása
python -m uvicorn backend.main:app --reload

# 5. Frontend indítása (új terminál)
streamlit run frontend/app.py
```

### Docker Fejlesztés

```bash
# Build
docker-compose build

# Indítás detached mode-ban
docker-compose up -d

# Logok követése
docker-compose logs -f

# Egy service újraindítása
docker-compose restart backend

# Leállítás
docker-compose down
```

---

## 📊 Használat

### 1. Model Tanítás

1. Nyisd meg: http://localhost:8501
2. Menj a **"Model Training"** oldalra
3. Töltsd fel a `data/cars.csv` fájlt
4. Válaszd ki a **target column**-t (pl. `Origin`)
5. Kattints a **"Train"** gombra
6. Model mentésre kerül: `backend/models/saved/`

### 2. Model Betöltés

1. Menj a **"Prediction"** oldalra
2. Válaszd ki a tanított modelt a listából
3. Kattints a **"Load Model"** gombra

### 3. Predikció

1. Tölts fel új CSV adatokat
2. Kattints az **"Adatok Küldése és Predikció"** gombra
3. Adatok RabbitMQ-n keresztül küldésre kerülnek
4. Backend feldolgozza → eredmények megjelennek
5. Metrikák, confusion matrix vizualizáció

---

## 🧪 Tesztelés

### API Tesztelés (Backend)

```bash
# Health check
curl http://localhost:8000/

# Modellek listázása
curl http://localhost:8000/models

# Model betöltése
curl -X POST http://localhost:8000/models/{model_id}/load

# Swagger UI
# http://localhost:8000/docs
```

### RabbitMQ Tesztelés

```bash
# RabbitMQ Management UI
# http://localhost:15672 (guest/guest)

# Queue-k listázása
docker-compose exec rabbitmq rabbitmqctl list_queues
```

### Docker Health Check

```bash
# Service állapotok
docker-compose ps

# Konténer health check
docker inspect --format='{{.State.Health.Status}}' mlops-backend
```

---

## 🐛 Troubleshooting

### RabbitMQ nem érhető el

```bash
# RabbitMQ logs
docker-compose logs rabbitmq

# Health check
docker-compose exec rabbitmq rabbitmq-diagnostics ping
```

### Backend nem indul

```bash
# Backend logs
docker-compose logs backend

# Shell a konténerben
docker-compose exec backend bash

# Dependencies ellenőrzése
docker-compose exec backend pip list
```

### Frontend nem kapcsolódik

```bash
# Network ellenőrzés
docker network inspect mlops_ml-network

# Backend elérhetőség frontend-ből
docker-compose exec frontend curl http://backend:8000/
```

---

## 📈 Production Ajánlások

### 1. Environment Variables

```bash
# .env fájl használata
RABBITMQ_USER=production_user
RABBITMQ_PASS=secure_password
BACKEND_URL=https://api.yourdomain.com
```

### 2. Security

- ✅ TLS/SSL használata (HTTPS)
- ✅ Secrets management (Docker secrets, Vault)
- ✅ Rate limiting (FastAPI middleware)
- ✅ Authentication (JWT tokens)

### 3. Monitoring

- ✅ Health checks minden service-nél
- ✅ Logging (ELK stack)
- ✅ Metrics (Prometheus + Grafana)
- ✅ Tracing (Jaeger, OpenTelemetry)

### 4. Scaling

```yaml
# docker-compose.yml
services:
  backend:
    deploy:
      replicas: 3  # Több instance
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

---

## 🤝 Közreműködés

Ez egy oktatási projekt. Javítások és fejlesztések szívesen fogadottak!

### Development Workflow

1. Fork a repository
2. Új branch létrehozása (`git checkout -b feature/amazing-feature`)
3. Commit változtatások (`git commit -m 'Add amazing feature'`)
4. Push branch (`git push origin feature/amazing-feature`)
5. Pull Request nyitása

---

## 📝 Licence

Ez a projekt oktatási célokat szolgál.

---

## 👨‍🏫 Tanulási Célok

Ezen a projekten keresztül a hallgatók:

✅ **SOLID Elvek** - Gyakorlati alkalmazás valós projektben  
✅ **Microservices** - Service-oriented architektúra  
✅ **Message Queues** - Aszinkron kommunikáció RabbitMQ-val  
✅ **REST API** - FastAPI best practices  
✅ **Docker** - Konténerizáció és orchestration  
✅ **ML Deployment** - Production-ready ML rendszer  
✅ **Clean Code** - Karbantartható, bővíthető kód  

---

## 🎉 Következő Lépések

1. ✅ Végezd el az 5 tutorialt sorrendben
2. ✅ Build-eld le a Docker környezetet
3. ✅ Próbálj ki saját adatokat
4. ✅ Bővítsd új feature-ökkel (SOLID elvekkel!)
5. ✅ Készíts teszteket
6. ✅ Deploy production környezetbe

---

**Jó tanulást! 🚀**
