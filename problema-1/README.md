# Problema 1: dimensión VC y generalización

En esta carpeta se encuentran la parte práctica del problema 1. Para el problema sobre anillos uso los parámetros personales `a=5` y `b=2`. El expediente completo fija la semilla para que los resultados se puedan reproducir. 

## Requisitos y reproducción

Para ejecutar el programa se necesita Python 3.11 o posterior. Desde la raíz del repositorio, instala el entorno común y ejecuta el experimento así:

```bash
python -m pip install -r requirements.txt
python problema-1/experimentos.py --seed 222215025
```

El programa guarda los resultados en `problema-1/resultados/`: los CSV contienen los valores numéricos, los PNG son las gráficas y `resumen.json` reúne los resultados principales.

## (a) Conteo de etiquetaciones

La función `realizable(puntos, etiquetas)` primero ordena los puntos por su distancia al origen. Un anillo puede etiquetar como positivos solo un bloque seguido de puntos en ese orden. Para cada `n` entre 1 y 12, reviso las `2^n` etiquetaciones en 100 conjuntos aleatorios y guardo el mayor conteo encontrado. La gráfica lo compara con todas las etiquetaciones posibles, `2^n`, y con la cota de Sauer, `(n+1)^2`, para una dimensión VC de 2.

**Qué muestra la gráfica.** El máximo empírico coincide con la fórmula exacta `1+n(n+1)/2` para los doce valores de `n`. Para `n=12`, por ejemplo, los anillos producen 79 etiquetaciones: muchas menos que las 4,096 posibles, aunque por debajo de la cota de Sauer de 169. La separación de `2^n` empieza en `n=3`, porque un anillo no puede etiquetar positivo-negativo-positivo al recorrer los radios en orden.

## (b) Curva de aprendizaje y ERM

Para ver cómo aprende el ERM, genero puntos uniformes en `[-1,1]^2` y uso como hipótesis objetivo el anillo `0.30 <= ||x|| <= 0.70`. Invierto al azar el 10 % de las etiquetas. Repito el experimento 30 veces para cada tamaño de muestra, desde 10 hasta 2,000 puntos, y estimo `Eout` con 10,000 puntos nuevos sujetos al mismo ruido.

El ERM busca el bloque contiguo de radios que comete menos errores. Lo hace de manera exacta: después de ordenar los radios, usa sumas prefijas para encontrar el mejor bloque en un solo recorrido. La gráfica compara la brecha promedio `Eout - Ein` con la cota de P1 y Sauer:

```text
sqrt((8/n) * ln(4 * (2n+1)^2 / 0.05))
```

**Qué muestra la gráfica.** La brecha observada disminuye conforme crece la muestra: pasa de 0.218 con `n=10` a 0.0166 con `n=200`, y queda cerca de cero con `n=2,000`. El último promedio es ligeramente negativo (−0.0020), algo posible al estimar los errores con muestras finitas independientes. La cota también baja al aumentar `n`, pero es mucho más alta que la brecha observada; por ejemplo, vale 0.290 en `n=2,000`. Esto ilustra que la cota ofrece una garantía general conservadora, no una predicción precisa del error en este experimento.

## (c)(i) Clase seno y precisión

Para cada `m` entre 1 y 25, genero 200 etiquetaciones y calculo `omega` con P3. Luego compruebo si cada etiquetación completa se reproduce tanto con `float64` como con `mpmath` a precisión arbitraria.

**Qué muestra la gráfica.** Con esta semilla, `float64` reproduce todas las etiquetaciones hasta `m=15`; a partir de `m=16` la proporción correcta cae, hasta llegar a cero en `m=25`. En cambio, `mpmath` reproduce todas las etiquetas en todos los tamaños. La diferencia apunta a un límite de precisión numérica de `float64`, no a una falla de la construcción teórica.

## (c)(ii) Búsqueda de omega

Aquí no hay una gráfica, sino una tabla con los resultados de la búsqueda. Uso como objetivo `f(x)=sign(sin(6 pi x))`, genero `x` uniformemente en `(0,1)` e invierto el 15 % de las etiquetas. Para cada tamaño `n` y límite `A`, elijo el `omega` de menor error de entrenamiento entre 60,000 valores igualmente espaciados en `(0,A]`. Después calculo `Eout` con 10,000 puntos nuevos y el mismo ruido.

**Qué indican los resultados.** Para `n=200`, al aumentar `A` de 30 a 30,000, `Ein` sube de 0.230 a 0.245 y `Eout` de 0.157 a 0.169. En esta muestra, permitir frecuencias más altas no mejora la predicción. Para `n=20`, `Ein` se mantiene en 0.05 y `Eout` cambia de 0.243 a 0.225. Los resultados dependen de la muestra y no tienen que cambiar de forma monótona. Limitar `A` restringe las frecuencias que el modelo puede elegir y funciona como regularización, aunque podría dejar fuera una frecuencia útil. Además, al mantener fija la cantidad de puntos de la rejilla, su espaciamiento crece con `A`; por eso, el mínimo encontrado es el mejor de los valores probados, no necesariamente el mínimo continuo.

## Decisiones y alcance

Elegí el mismo tipo de ruido y la misma distribución para los datos de entrenamiento y prueba, para que sus errores se puedan comparar. La semilla corresponde al expediente personal.
