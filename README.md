# Playground

Plataforma de comparación de precios de videojuegos entre tiendas digitales.
Este repositorio incluye, además, el componente de minería de datos del proyecto (Minería de Datos I).

## Integrantes
- Yorllanis Santana González
- Marwin Bello Flames
- Jonathan Gaviria Lara
- Isaac González Martínez

## Problema que aborda
Conocer solo el precio actual de un videojuego no permite saber si conviene comprarlo ahora o esperar una oferta.
Se propone un modelo predictivo que estime si un videojuego tendrá al menos un descuento durante los próximos 30 días, a partir de su historial de precios.

## Tecnologías utilizadas
- MongoDB (base de datos de Playground)
- Python 3, pandas, numpy y scikit-learn (análisis y modelo)
- Google Colab (ejecución del modelo)
- MinedApp (guía del proceso de minería de datos)

## Fuente de los datos
Dataset público de Steam con historial diario de precios:

Wannigamage, D., Barlow, M., Lakshika, E. y Kasmarik, K. *Steam Games Dataset: Player count history, Price history and data about games*. Mendeley Data, V1, 2020. DOI: 10.17632/ycy3sy3vj2.1
https://data.mendeley.com/datasets/ycy3sy3vj2/1

El dataset no se incluye en este repositorio. Ver `data/README.md` para obtenerlo.

## Cómo ejecutar el análisis
1. Abrir https://colab.research.google.com y crear un cuaderno nuevo.
2. Subir `PriceHistory.zip` (descargado de Mendeley) y `mineria-datos/modelo_descuentos.py`.
3. Ejecutar en una celda:

```
!unzip -q PriceHistory.zip
!python modelo_descuentos.py PriceHistory
```

Para ejecutarlo en local: `pip install pandas numpy scikit-learn` y luego `python modelo_descuentos.py PriceHistory`.

## Resultados obtenidos
Modelo ejecutado como prueba de concepto sobre el dataset público (aún no con datos propios de Playground).
División temporal: entrenamiento con observaciones hasta el 01-12-2019 (44.407) y prueba desde el 01-01-2020 (40.680).

| Modelo | Precisión | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Referencia: siempre "sí" | 0,518 | 1,000 | 0,683 | - |
| Referencia: en oferta ahora | 0,736 | 0,223 | 0,342 | - |
| Referencia: descuento en últimos 90 días | 0,551 | 0,906 | 0,685 | - |
| Regresión logística | 0,534 | 0,987 | 0,693 | 0,664 |
| Random Forest | 0,545 | 0,984 | 0,702 | 0,665 |

El desempeño es modesto: el Random Forest mejora poco a la referencia "siempre sí" y tiende a predecir descuento casi siempre.
Los resultados completos están en `mineria-datos/resultados/resultados.json`.

## Limitaciones
Datos de 2019 a 2020, solo Steam y USD, muestra sesgada hacia juegos populares. El modelo debe reentrenarse con el historial propio de Playground.

## Documentos de la entrega
- Propuesta del modelo predictivo: `documentos/Propuesta_Modelo_Predictivo_Playground.pdf`
- Informe metodológico: `documentos/informe_metodologico.pdf` (pendiente)

# Datos

Los datos no se incluyen en este repositorio. Se documenta su procedencia para poder obtenerlos.

## Dataset usado en el modelo
- **Nombre:** Steam Games Dataset: Player count history, Price history and data about games
- **Autores:** Dulakshi Wannigamage, Michael Barlow, Erandi Lakshika, Kathryn Kasmarik
- **Publicación:** Mendeley Data, V1, 17 de agosto de 2020
- **DOI:** 10.17632/ycy3sy3vj2.1
- **Enlace:** https://data.mendeley.com/datasets/ycy3sy3vj2/1
- **Licencia:** consultar en la ficha del dataset.

## Qué archivo se utilizó
Solo `PriceHistory.zip`: un CSV por videojuego (nombre = appid de Steam) con las columnas
`Date`, `Initialprice`, `Finalprice` y `Discount`, en frecuencia diaria y en USD,
del 07-04-2019 al 12-08-2020 (1.512 juegos).

## Cómo obtenerlo
1. Entrar al enlace anterior.
2. En la sección *Files*, descargar `PriceHistory.zip`.
3. Descomprimirlo en una carpeta `PriceHistory/` junto al script.

## Datos propios de Playground
Captura de precios de Steam del 16-10-2026 (8 países, 244.414 filas). Es un único corte en el tiempo, por lo que
no permite construir la variable objetivo; se conserva como primer punto de un historial propio que se seguirá
acumulando.
