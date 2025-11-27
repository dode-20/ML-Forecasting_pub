# Configuration Files

## data_paths.json
**Konfiguriert direkte CSV-Dateien oder Verzeichnisse für die Trainingsdaten**

### Direkte Datei-Pfade (training_data_files):
```json
{
  "silicon": "/vollständiger/pfad/zur/silicon_datei.csv",
  "perovskite": "/vollständiger/pfad/zur/perovskite_datei.csv"
}
```

### Fallback-Verzeichnisse (fallback_directories):
Falls die direkten Dateien nicht existieren, wird in diesen Verzeichnissen gesucht.

### Search Preferences:
- **resolution**: Bevorzugte Datenauflösung (5min, 10min, 1h) für Verzeichnissuche
- **weather_integrated**: Bevorzugung für weather-integrierte Dateien
- **prefer_latest**: Neueste Datei auswählen

## Anpassung der Pfade:

### Option 1: Direkte CSV-Datei angeben
```json
{
  "training_data_files": {
    "silicon": ".../ML-forecasting/results/training_data/Silicon/cleanData/20240901_20250725/20240901_20250725_test_lstm_model_clean-5min_weather-integrated.csv"
  }
}
```

### Option 2: Verzeichnis für automatische Suche
```json
{
  "fallback_directories": {
    "silicon": ["/ihr/verzeichnis/pfad"]
  }
}
```

## Priorität:
1. **Direkte Datei** aus `training_data_files` wird zuerst versucht
2. **Fallback-Suche** in `fallback_directories` falls direkte Datei nicht gefunden wird