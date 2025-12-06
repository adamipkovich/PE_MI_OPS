# 05 - Docker Setup és Orchestration

## Mi a Docker és miért van rá szükség?

### A Probléma

Amikor egy komplex rendszert építünk több komponenssel:

1. **Environment különbségek** - "Nálam működik" szindróma
2. **Dependency konfliktusok** - Különböző Python verziók, package-ek
3. **Deployment bonyolultsága** - Minden service manuális telepítése
4. **Skálázhatóság** - Nehéz több instance-t indítani
5. **Izolláció hiánya** - Service-ek interferálnak egymással

### A Megoldás: Docker Konténerek

**Docker** konténerek könnyűsúlyú, hordozható, izolált környezetek alkalmazások futtatására.

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Backend   │  │  Frontend   │  │  RabbitMQ   │  │   Volumes   │
│  Container  │  │  Container  │  │  Container  │  │   (Data)    │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
       │                │                │                │
       └────────────────┴────────────────┴────────────────┘
                        Docker Network
```

---

## Docker Alapfogalmak

### 1. **Image (Kép)**
- Egy alkalmazás "blueprint"-je
- Tartalmazza a kódot, függőségeket, környezeti változókat
- Immutable (változtathatatlan)

### 2. **Container (Konténer)**
- Egy image futó instance-a
- Izolált környezet (saját filesystem, network, process space)
- Lightweight (könnyűsúlyú)

### 3. **Dockerfile**
- Recept egy image elkészítéséhez
- Lépésről-lépésre definiálja az environment-et

### 4. **Docker Compose**
- Több konténer orchestrációja
- YAML konfigurációval
- Networks, volumes, dependencies kezelése

---

## Architektúra Docker-rel

### Service-ek

```yaml
services:
  rabbitmq:     # Message broker
  backend:      # FastAPI + ML services
  frontend:     # Streamlit UI
```

### Networks

```yaml
networks:
  ml-network:   # Belső kommunikáció
```

### Volumes

```yaml
volumes:
  models-data:  # Modellek perzisztens tárolása
  app-data:     # Alkalmazás adatok
```

---

## Dockerfile-ok Létrehozása

### Backend Dockerfile

**Fájl:** `docker/Dockerfile.backend`

```dockerfile
FROM python:3.11-slim

# Munkakönyvtár
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

# App kód másolása
COPY app/ /app/app/
COPY backend/ /app/backend/
COPY data/ /app/data/

# Port expose
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Entrypoint
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Miért így?**
- `python:3.11-slim` - Kis méretű base image
- Multi-stage nem kell (egyszerűség kedvéért)
- Health check - Docker tudja mikor ready
- `--no-cache-dir` - Kisebb image méret

---

### Frontend Dockerfile

**Fájl:** `docker/Dockerfile.frontend`

```dockerfile
FROM python:3.11-slim

# Munkakönyvtár
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements-frontend.txt .
RUN pip install --no-cache-dir -r requirements-frontend.txt

# App kód másolása
COPY app/ /app/app/
COPY frontend/ /app/frontend/
COPY data/ /app/data/

# Port expose
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Entrypoint
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Streamlit specifikus:**
- `/_stcore/health` - Streamlit health endpoint
- `--server.address=0.0.0.0` - Minden interface-en hallgat

---

## Docker Compose Konfiguráció

**Fájl:** `docker-compose.yml`

```yaml
version: '3.8'

services:
  # RabbitMQ Message Broker
  rabbitmq:
    image: rabbitmq:3-management
    container_name: mlops-rabbitmq
    ports:
      - "5672:5672"      # AMQP protocol
      - "15672:15672"    # Management UI
    environment:
      RABBITMQ_DEFAULT_USER: guest
      RABBITMQ_DEFAULT_PASS: guest
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "ping"]
      interval: 30s
      timeout: 10s
      retries: 5
    networks:
      - ml-network

  # Backend Service (FastAPI + ML)
  backend:
    build:
      context: .
      dockerfile: docker/Dockerfile.backend
    container_name: mlops-backend
    ports:
      - "8000:8000"
    environment:
      - RABBITMQ_HOST=rabbitmq
      - RABBITMQ_PORT=5672
    volumes:
      - ./backend/models:/app/backend/models  # Model persistence
      - ./data:/app/data                       # Data access
    depends_on:
      rabbitmq:
        condition: service_healthy
    networks:
      - ml-network
    restart: unless-stopped

  # Frontend Service (Streamlit)
  frontend:
    build:
      context: .
      dockerfile: docker/Dockerfile.frontend
    container_name: mlops-frontend
    ports:
      - "8501:8501"
    environment:
      - BACKEND_URL=http://backend:8000
      - RABBITMQ_HOST=rabbitmq
      - RABBITMQ_PORT=5672
    volumes:
      - ./data:/app/data  # Data access
    depends_on:
      - backend
      - rabbitmq
    networks:
      - ml-network
    restart: unless-stopped

networks:
  ml-network:
    driver: bridge

volumes:
  models-data:
  app-data:
```

**Kulcs elemek:**

1. **depends_on** - Indítási sorrend
2. **healthcheck** - Service readiness
3. **volumes** - Adat perzisztencia
4. **networks** - Service-ek kommunikációja
5. **restart** - Automatikus újraindítás hiba esetén

---

## Requirements Fájlok

### Backend Requirements

**Fájl:** `requirements-backend.txt`

```txt
fastapi==0.114.1
uvicorn==0.30.6
pydantic==2.9.2
pandas==2.2.2
numpy==2.1.1
scikit-learn==1.5.2
joblib==1.4.2
pika==1.3.2
requests==2.32.3
```

### Frontend Requirements

**Fájl:** `requirements-frontend.txt`

```txt
streamlit==1.38.0
pandas==2.2.2
matplotlib==3.9.2
scikit-learn==1.5.2
pika==1.3.2
requests==2.32.3
```

---

## Docker Parancsok

### Build

```bash
# Minden service build-elése
docker-compose build

# Csak egy service build-elése
docker-compose build backend
docker-compose build frontend
```

### Indítás

```bash
# Összes service indítása (detached mode)
docker-compose up -d

# Logok követése
docker-compose logs -f

# Csak egy service logjai
docker-compose logs -f backend
```

### Leállítás

```bash
# Service-ek leállítása (konténerek megmaradnak)
docker-compose stop

# Service-ek leállítása és törlése
docker-compose down

# Volumes is törlése
docker-compose down -v
```

### Debugging

```bash
# Service állapot ellenőrzése
docker-compose ps

# Belépés egy futó konténerbe
docker-compose exec backend bash
docker-compose exec frontend bash

# Konténer logok
docker-compose logs backend
docker-compose logs frontend
docker-compose logs rabbitmq
```

---

## .dockerignore Fájl

**Fájl:** `.dockerignore`

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Git
.git/
.gitignore

# Docker
Dockerfile*
docker-compose*.yml

# OS
.DS_Store
Thumbs.db

# Project specific
backend/models/saved/*.pkl
mlruns/
filled/
assets/
*.md
```

---

## Teljes Workflow

### 1. Development

```bash
# Build
docker-compose build

# Indítás
docker-compose up -d

# Logok nézése
docker-compose logs -f
```

**Elérések:**
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- Backend Docs: http://localhost:8000/docs
- RabbitMQ Management: http://localhost:15672

### 2. Model Training

1. Nyisd meg a frontend-et: http://localhost:8501
2. Menj a "Model Training" oldalra
3. Tölts fel CSV-t (`data/cars.csv`)
4. Válaszd ki a target column-t (pl. `Origin`)
5. Kattints "Train" gombra
6. Model mentésre kerül: `backend/models/saved/`

### 3. Prediction

1. Menj a "Prediction" oldalra
2. Válaszd ki a modelt
3. Töltsd be ("Load Model")
4. Tölts fel új adatokat
5. Kattints "Predict" gombra
6. Eredmények megjelenítése

### 4. Leállítás

```bash
docker-compose down
```

---

## Troubleshooting

### RabbitMQ nem érhető el

```bash
# RabbitMQ health check
docker-compose exec rabbitmq rabbitmq-diagnostics ping

# RabbitMQ logs
docker-compose logs rabbitmq
```

### Backend nem indul

```bash
# Backend logs
docker-compose logs backend

# Dependency ellenőrzés
docker-compose exec backend pip list
```

### Frontend nem kapcsolódik backend-hez

```bash
# Network ellenőrzés
docker network inspect mlops_ml-network

# Backend elérhetőség tesztelése frontend-ből
docker-compose exec frontend curl http://backend:8000/
```

---

## Production Considerations

### 1. Environment Variables

Használj `.env` fájlt:

```env
# .env
RABBITMQ_USER=admin
RABBITMQ_PASS=secure_password
BACKEND_PORT=8000
FRONTEND_PORT=8501
```

### 2. Secrets Management

Ne commitálj érzékeny adatokat:
- Docker secrets használata
- Vagy external secrets manager (Vault, AWS Secrets Manager)

### 3. Resource Limits

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### 4. Monitoring

- Health checks minden service-nél
- Logging aggregation (ELK stack)
- Metrics (Prometheus + Grafana)

---

## SOLID és Docker

Docker is támogatja a SOLID elveket:

- **SRP**: Minden konténer egy service
- **OCP**: Új service hozzáadása nem érint meglévőket
- **DIP**: Service-ek interfészeken (network) keresztül kommunikálnak

---

## Összefoglalás

✅ **Dockerfile.backend** - FastAPI + ML backend  
✅ **Dockerfile.frontend** - Streamlit UI  
✅ **docker-compose.yml** - Orchestration  
✅ **Requirements** - Dependency management  
✅ **.dockerignore** - Build optimization  

### Parancsok

```bash
# Build és indítás
docker-compose up -d --build

# Logok
docker-compose logs -f

# Leállítás
docker-compose down
```

---

## Következő Lépések

Most már a teljes rendszer Dockerizálva van! A hallgatók:
1. Megértik a Docker előnyeit
2. Tudják használni a docker-compose-t
3. Debugolni tudnak konténereket
4. Production-ready ismeretekkel rendelkeznek

🎉 **Gratulálok! A teljes ML platform készen áll!**
