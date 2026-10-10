# Image python
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers requirements.txt et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application
COPY . .

# Exposer le port sur lequel l'application s'exécute
EXPOSE 8000
EXPOSE 8501