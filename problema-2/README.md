# Problema 2: regresión ordinal

Este directorio contiene el análisis de abulón que corresponde a los parámetros personales (`a=5`, `b=2`, `a mod 3=2`). El programa descarga el conjunto directamente de UCI en cada ejecución; no guarda los datos crudos en el repositorio.

## Reproducir

Desde la raíz del repositorio:

```bash
python -m pip install -r requirements.txt
python problema-2/run_analysis.py --seed 222215025 --metric qwk --invalid exclude --m5-conversion cuts
```

El archivo `problema-2/requirements.txt` también permite instalar únicamente las dependencias de este problema; para correr ambos, usa el `requirements.txt` de la raíz.

Los argumentos que controlan el análisis son `--metric`, `--invalid` y `--m5-conversion`. Para `--metric` se puede elegir `accuracy` (exactitud), `mae` (error absoluto medio) o `qwk` (kappa cuadrático ponderado). En `--invalid`, `exclude` elimina las filas con medidas físicamente imposibles y `keep` las conserva. Para M5, `round` redondea la predicción y `cuts` ajusta los cortes con validación. La opción `--audit-only` permite revisar primero el balance de niveles y las mediciones sospechosas.

## Datos y preparación

Los niveles son `1-7`, `8-9`, `10-11`, `12-14` y `15+` anillos. Se conserva una partición estratificada 75/25 con la semilla del expediente. Los parámetros se eligen con validación cruzada de cinco pliegues sobre el conjunto de entrenamiento; las etiquetas de prueba se usan solo para la evaluación final.

UCI reporta 4,177 filas y no hay valores faltantes. El sexo solo toma `M`, `F` e `I`; la categoría `I` significa juvenil y es válida. La revisión física encontró dos filas con altura cero; las excluyo porque una altura de cero no es físicamente posible. Tras excluirlas quedan 4,175 registros, distribuidos por nivel como `[838, 1256, 1121, 596, 364]`. El script permite conservar las dos filas con `--invalid keep` para comparar la decisión.

Los siete atributos numéricos se estandarizan. La variable de sexo (`Sex`) se representa con dos indicadores y se usa `F` como categoría de referencia; también se estandarizan esos indicadores. De esta forma, los coeficientes se pueden comparar en escalas similares. La dimensión de entrada resultante es `d=9`.

## Modelos y decisiones

Se comparan seis alternativas: regresión lineal con predicciones redondeadas (M1), regresión logística multinomial (M2), modelo logístico ordinal propio con predicciones por moda y mediana (M3), clasificador de potenciación por gradiente (M4), regresor de potenciación por gradiente (M5) y un conjunto de clasificadores binarios, uno por umbral (M6). Se prueban distintos valores de regularización y de hojas por árbol mediante validación cruzada de cinco pliegues.

Para M5, `cuts` ajusta cuatro puntos de corte usando predicciones de validación fuera de pliegue del entrenamiento; `round` redondea al nivel más cercano. Para M6 se calcula por separado cada probabilidad `P(Y>k)`. Se cuenta en qué fracción de las filas estas probabilidades no disminuyen al subir el umbral y luego se aplica una proyección isotónica para imponer el orden esperado antes de calcular las probabilidades de cada nivel.

Los umbrales del modelo ordinal se mantienen ordenados parametrizando cada diferencia como un exponencial positivo: `theta_k = theta_1 + sum(exp(u_j))`. Así el optimizador nunca puede invertir el orden de las categorías.

La métrica principal elegida es el kappa cuadrático ponderado (QWK). Con esta medida, equivocarse por varios niveles pesa más que confundir niveles vecinos. El programa también informa la exactitud y el error absoluto medio (MAE), para mostrar si el modelo ganador cambia según la métrica.

La matriz de costos ordinales que corresponde a esa preferencia es proporcional al cuadrado de la distancia entre niveles. QWK la normaliza dividiendo los valores entre 16:

```text
0  1  4  9 16
1  0  1  4  9
4  1  0  1  4
9  4  1  0  1
16 9  4  1  0
```

## Resultados de esta ejecución

Con QWK como criterio de selección de hiperparámetros, los resultados en prueba son:

| Modelo | Exactitud | Error absoluto medio (MAE) | Kappa cuadrático (QWK) |
|---|---:|---:|---:|
| M1: lineal redondeada | 0.473 | 0.602 | 0.660 |
| M2: logística multinomial | **0.551** | **0.559** | 0.691 |
| M3: ordinal, moda | 0.522 | 0.581 | 0.683 |
| M3: ordinal, mediana | 0.504 | 0.587 | 0.680 |
| M4: clasificador de potenciación por gradiente | 0.544 | 0.580 | 0.678 |
| M5: regresor de potenciación por gradiente con cortes validados | 0.533 | 0.562 | **0.736** |
| M6: clasificadores de potenciación por umbrales | 0.273 | 1.325 | 0.489 |

El modelo ganador cambia según la métrica: M2 tiene la mayor exactitud y el menor MAE, mientras que M5 obtiene el mayor QWK. Los cortes que M5 eligió usando predicciones fuera de pliegue fueron aproximadamente `[0.732, 1.500, 2.243, 2.803]`. Como la métrica principal es QWK, recomiendo M5 para este conjunto. M3 no lo supera en esta evaluación, aunque ofrece una estructura ordinal interpretable. En M6, el 2.2 % de las filas de prueba tuvo probabilidades de umbral que no disminuían; la proyección isotónica las ordena antes de convertirlas en probabilidades de nivel. Su desempeño fue inferior en esta ejecución.

**Qué muestra la matriz de confusión.** El eje vertical indica el nivel real y el horizontal el nivel predicho; cada celda cuenta abulones de esa combinación. Los valores de la diagonal son aciertos: por ejemplo, M5 clasificó correctamente 171 de los 210 casos del nivel `1-7` y 49 de los 91 casos del nivel `15+`. La mayoría de los errores se concentra en niveles vecinos; por ejemplo, de los 314 casos reales `8-9`, 119 se predijeron como `10-11` y 39 como `1-7`. Los niveles extremos son más difíciles: el modelo tiende a predecir un nivel menor para algunos abulones `15+`. Esto ayuda a entender por qué QWK, que penaliza más los errores lejanos, es una medida relevante para esta tarea.

El gradiente analítico del modelo ordinal difiere del gradiente numérico en `3.5e-8`. Sus coeficientes prácticamente coinciden con los de `OrderedModel`, de la biblioteca `statsmodels`. Entre los atributos estandarizados, el peso total (`Whole_weight`) tiene un coeficiente positivo y el peso de la carne sin concha (`Shucked_weight`) uno negativo, manteniendo fijos los demás atributos. Ese signo puede deberse a que los distintos pesos físicos están correlacionados. La importancia por permutación del mejor modelo de potenciación por gradiente destaca el peso de la concha (`Shell_weight`), el peso de la carne (`Shucked_weight`), el peso total (`Whole_weight`) y la altura (`Height`). Los dos modelos coinciden en tres de sus atributos más importantes; el modelo ordinal, en cambio, destaca más el diámetro (`Diameter`).

## Archivos de salida

- `auditoria_datos.json`: resumen de los datos, medidas imposibles y partición de entrenamiento y prueba.
- `validacion_cruzada.json`: métricas promedio de las configuraciones probadas en los pliegues.
- `metricas_prueba.csv`: métricas finales de cada modelo.
- `matriz_confusion.csv` y `matriz_confusion.png`: tabla y gráfica de los niveles reales frente a los predichos por el modelo con mayor QWK.
- `coeficientes_ordinales.csv`: comparación entre los coeficientes del modelo propio y los de `OrderedModel`.
- `importancia_permutacion.csv`: cuánto aumenta el MAE al permutar cada atributo.
- `resultados_finales.json`: resumen de los resultados y de las decisiones usadas en la ejecución.

