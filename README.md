# counterfactual-explanations

[![Building Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Con el proposito de estudiar modelos de interpretabilidad en machine learning cree este repo para hacer un roadmap que me permita llegar entender el estado del arte de interpretabilidad de modelos de deep learning.

Para esto me crearé un apunte que secaré de distintas fuentes ver (última parte)

# Apunte de estudio

## Taxonomía de los métodos de interpretabilidad 

Los métodos para la interpretabilidad del machine learning se pueden clasificar según varios criterios.

¿Intrínseco o post hoc? (Intrinsic or post hoc? )
Este criterio distingue si la interpretabilidad se logra restringiendo la complejidad del modelo de aprendizaje automático (intrínseco) o aplicando métodos que analizan el modelo después del entrenamiento (post hoc). La interpretabilidad intrínseca se refiere a modelos de aprendizaje automático que se consideran interpretables debido a su estructura simple, como árboles de decisión cortos o modelos lineales dispersos. La interpretabilidad post hoc se refiere a la aplicación de métodos de interpretación después del entrenamiento del modelo. La importancia de la característica de permutación es, por ejemplo, un método de interpretación post hoc. Los métodos post hoc también se pueden aplicar a modelos intrínsecamente interpretables. Por ejemplo, la importancia de la característica de permutación se puede calcular para árboles de decisión. 

Resultado del método de interpretación Los distintos métodos de interpretación se pueden diferenciar a grandes rasgos según sus resultados.

- Feature summary statistic: Muchos métodos de interpretación proporcionan estadísticas de resumen para cada característica. Algunos métodos devuelven un solo número por característica, como la importancia de la característica, o un resultado más complejo, como las fortalezas de interacción de características por pares, que consisten en un número para cada par de características.

- Feature summary visualization: La mayoría de las estadísticas de resumen de funciones también se pueden visualizar. Algunos resúmenes de funciones solo son significativos si se visualizan y una tabla sería una elección incorrecta. La dependencia parcial de una característica es tal caso. Los gráficos de dependencia parcial son curvas que muestran una característica y el resultado promedio previsto. La mejor manera de presentar dependencias parciales es dibujar la curva en lugar de imprimir las coordenadas.

- Model internals (por ejemplo, pesos aprendidos): La interpretación de modelos intrínsecamente interpretables cae en esta categoría. Algunos ejemplos son los pesos en modelos lineales o la estructura de árbol aprendida (las características y umbrales utilizados para las divisiones) de árboles de decisión. Las líneas están borrosas entre los componentes internos del modelo y la estadística de resumen de características en, por ejemplo, modelos lineales, porque los pesos son tanto internos del modelo como estadísticas de resumen para las características al mismo tiempo. Otro método que genera los componentes internos del modelo es la visualización de detectores de características aprendidos en redes neuronales convolucionales. Los métodos de interpretabilidad que generan los componentes internos del modelo son, por definición, específicos del modelo (consulte el siguiente criterio).

- Data point: esta categoría incluye todos los métodos que devuelven puntos de datos (ya existentes o creados recientemente) para hacer que un modelo sea interpretable. Un método se llama counterfactual explanations. Para explicar la predicción de una instancia de datos, el método encuentra un punto de datos similar cambiando algunas de las características para las que el resultado predicho cambia de manera relevante (por ejemplo, un cambio en la clase predicha). Otro ejemplo es la identificación de prototipos de clases predichas. Para ser útiles, los métodos de interpretación que generan nuevos puntos de datos requieren que los mismos puntos de datos puedan interpretarse. Esto funciona bien para imágenes y textos, pero es menos útil para datos tabulares con cientos de funciones.

¿Modelo específico o agnóstico del modelo? Las herramientas de interpretación específicas del modelo están limitadas a clases de modelos específicas. La interpretación de las ponderaciones de regresión en un modelo lineal es una interpretación específica del modelo, ya que, por definición, la interpretación de los modelos intrínsecamente interpretables es siempre específica del modelo. Herramientas que solo funcionan para la interpretación de p. Ej. Las redes neuronales son específicas del modelo. Las herramientas agnósticas del modelo se pueden usar en cualquier modelo de aprendizaje automático y se aplican después de que el modelo haya sido entrenado (post hoc). Estos métodos agnósticos suelen funcionar analizando pares de entrada y salida de características. Por definición, estos métodos no pueden tener acceso a los componentes internos del modelo, como los pesos o la información estructural.


# Fuentes

- documents/ --> carpeta con papers y libros de interpretabilidad
- https://christophm.github.io/interpretable-ml-book/taxonomy-of-interpretability-methods.html
- https://docs.seldon.io/projects/alibi/en/stable/methods/CF.html
- https://github.com/amirhk/mace
