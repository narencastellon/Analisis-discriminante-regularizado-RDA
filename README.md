# Analisis-discriminante-regularizado-RDA

## **7. Análisis discriminante regularizado.**

RDA construye una regla de clasificación al regularizar las matrices de covarianza de grupo (Friedman 1989) permitiendo un modelo más robusto contra la multicolinealidad en los datos. Esto puede resultar muy útil para un gran conjunto de datos multivariados que contienen predictores altamente correlacionados.

El análisis discriminante regularizado es una especie de compensación entre LDA y QDA. Recuerde que en LDA asumimos la igualdad de la matriz de covarianza para todas las clases. QDA asume diferentes matrices de covarianza para todas las clases. El análisis discriminante regularizado es un intermedio entre LDA y QDA.

RDA reduce las covarianzas separadas de QDA hacia una covarianza común como en LDA. Esto mejora la estimación de las matrices de covarianza en situaciones en las que el número de predictores es mayor que el número de muestras en los datos de entrenamiento, lo que podría conducir a una mejora de la precisión del modelo.

## **Paso 1. Carga de paquetes R requeridos.**
Carga de paquetes R requeridos

`tidyverse` para una fácil visualización y manipulación de datos.
`caret` para un flujo de trabajo de aprendizaje automático (Machine Learning) sencillo.

```{r message=FALSE}
library(tidyverse)
library(caret)
library(klaR)
theme_set(theme_classic())
```


## **Paso 2. Preparando los datos.**

Usaremos el conjunto iris de datos, para predecir especies de iris basadas en las variables predictoras Sepal.Length, Sepal.Width, Petal.Length, Petal.Width.

El análisis discriminante puede verse afectado por la escala / unidad en la que se miden las variables predictoras. Generalmente se recomienda estandarizar / normalizar el predictor continuo antes del análisis.

**2.1. Divida los datos en entrenamiento y conjunto de prueba:**

```{r}
# Cargamos la data
data("iris")
# Dividimos la data para entrenamiento en un (80%) y para la prueba en un (20%)
set.seed(123)
training.samples <- iris$Species %>%
createDataPartition(p = 0.8, list = FALSE)
train.data <- iris[training.samples, ]
test.data <- iris[-training.samples, ]
```

**2. Normaliza los datos. Las variables categóricas se ignoran automáticamente.**

```{r}
# Estimar parámetros de preprocesamiento
preproc.param <- train.data %>% 
preProcess(method = c("center", "scale"))
# Transformar los datos usando los parámetros estimados
train.transformed <- preproc.param %>% predict(train.data)
test.transformed <- preproc.param %>% predict(test.data)
```

# **Paso 3. Creación del Modelo RDA**
RDA se puede calcular usando la función `rda()`[paquete MASS]

```{r warning=FALSE}
library(klaR)
# Creando el Modelo
modelrda <- rda(Species~., data = train.transformed)
modelrda

```

## **Paso 4. Gráficos de partición RDA**

El uso de la partimatfunción nuevamente proporciona una forma de graficar las funciones discriminantes cuadráticas. La única diferencia en el código de la sección LDA anterior es reemplazar `method="lda"`con `method="qda"`. Estos gráficos proporcionan una buena visualización de la diferencia entre las funciones lineales utilizadas en LDA y las funciones cuadráticas utilizadas en QDA. Nuevamente, las regiones coloreadas delinean cada área de clasificación. Se predice que cualquier observación que se encuentre dentro de una región sea de una clase específica. Cada gráfico también incluye la tasa de error aparente para esa vista de los datos.

```{r}
partimat(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data=train.transformed, method="rda")
```

## **Paso 5: use el modelo para hacer predicciones RDA**

Una vez que hemos ajustado el modelo utilizando nuestros datos de entrenamiento, podemos usarlo para hacer predicciones sobre nuestros datos de prueba:

```{r}
# Haciendo predicciones
predictions <- modelrda %>% predict(test.transformed)
```

## **Paso 6: evaluar el modelo RDA**

Podemos usar el siguiente código para ver para qué porcentaje de observaciones el modelo RDA predijo correctamente la Specie:

```{r}
# Precisión del Modelo
mean(predictions$class == test.transformed$Species)
```


Resulta que el modelo predijo correctamente las especies para el 96.67% de las observaciones en nuestro conjunto de datos de prueba.

## **8. Conclusión.**
Hemos descrito el análisis discriminante lineal (LDA) y las extensiones para predecir la clase de una observación basada en múltiples variables predictoras. El análisis discriminante es más adecuado para problemas de clasificación multiclase en comparación con la regresión logística.

LDA asume que las diferentes clases tienen la misma matriz de varianza o covarianza. Hemos descrito muchas extensiones de LDA en este capítulo. La extensión más popular de LDA es el análisis discriminante cuadrático (QDA), que es más flexible que LDA en el sentido de que no asume la igualdad de matrices de covarianza de grupo.

LDA tiende a ser mejor que QDA para conjuntos de datos pequeños. Se recomienda QDA para grandes conjuntos de datos de entrenamiento.

## **9. Referencias.**

1. Friedman, Jerome H. 1989. “Regularized Discriminant Analysis.” Journal of the American Statistical Association 84 (405). Taylor & Francis: 165–75. doi:10.1080/01621459.1989.10478752.

2. James, Gareth, Daniela Witten, Trevor Hastie, and Robert Tibshirani. 2014. An Introduction to Statistical Learning: With Applications in R. Springer Publishing Company, Incorporated.

