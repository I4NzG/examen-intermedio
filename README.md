# Examen intermedio: Reconocimiento de Patrones

Soluciones teóricas y prácticas del examen intermedio de la materia
Reconocimiento de Patrones, Universidad de Sonora, semestre 2026-2.

## Datos del estudiante

- **Nombre:** Ian Alonso Zepeda Godoy
- **Expediente:** `222215025`
- **Parámetros personales:** `a=5`, `b=2`
- **Semilla aleatoria:** `222215025` (expediente completo)

## Requisitos

Se requiere Python 3.11 o posterior. Las versiones exactas de las
bibliotecas usadas están en [`requirements.txt`](requirements.txt).

Desde la raíz del repositorio, crea y activa un entorno virtual e instala
las dependencias:

```bash
python -m venv .venv
```

En Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

En macOS o Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Ejecutar los problemas

### Problema 1: anillos y clase seno

```bash
python problema-1/experimentos.py --seed 222215025
```

El programa genera las tablas y gráficas en `problema-1/resultados/`.
El detalle del experimento y la interpretación de las figuras están en
[`problema-1/README.md`](problema-1/README.md).

### Problema 2: regresión ordinal con abulón

```bash
python problema-2/run_analysis.py --seed 222215025 --metric qwk --invalid exclude --m5-conversion cuts
```

Este programa necesita conexión a internet para descargar el conjunto de
abulón desde UCI. No guarda los datos crudos en el repositorio. Las tablas
y la matriz de confusión se generan en `problema-2/resultados/`; las
decisiones y resultados se explican en
[`problema-2/README.md`](problema-2/README.md).

Para revisar primero el balance de niveles y las medidas imposibles, sin
ejecutar los modelos:

```bash
python problema-2/run_analysis.py --seed 222215025 --audit-only
```

## Estructura

```text
examen-intermedio/
├── README.md
├── requirements.txt
├── problema-1/
│   ├── README.md
│   ├── experimentos.py
│   └── resultados/          # se generan al ejecutar el problema 1
└── problema-2/
    ├── README.md
    ├── requirements.txt    # dependencias específicas del problema 2
    ├── ordinal.py
    ├── run_analysis.py
    └── resultados/          # se generan al ejecutar el problema 2
```

La semilla de ambos problemas es el expediente completo: `222215025`.
