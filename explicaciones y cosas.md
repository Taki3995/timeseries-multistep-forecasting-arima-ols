# UTILITY:

Generación de Matrices (OLS): La función create_lag_matrix desplaza el array original para construir una matriz $X$ donde cada columna es un rezago $y*{t-i}$. Esto es fundamental porque OLS en Python usando numpy se resuelve algebraicamente como $\hat{\beta} = (X^T X)^{-1} X^T Y$, evitando así importar librerías externas para la regresión.

Recuperación Binomial: Para cumplir con $\hat{Y}_{t+h} = \hat{z}_{t+h} - \sum_{k=1}^d (-1)^k \binom{d}{k} Y_{t+h-k}$, se desarrolló una función combinatoria estricta (binom_coeff). El índice negativo en y_hist[-k] permite leer el historial hacia atrás, garantizando que el sumatorio reste la proporción correcta de los datos originales no diferenciados.

Test de Jarque-Bera: Se implementaron fielmente las ecuaciones de los momentos muestrales $\hat{\mu}_k = \frac{1}{n} \sum_{t=1}^n (\eta_t - \bar{\eta})^k$. El cálculo de la varianza, asimetría ($S$) y curtosis ($K$) se realiza usando las funciones vectorizadas base de numpy (np.mean y exponenciación) para luego ensamblar el estadístico final $JB$.

# ADF:

- no tenemos coeficient4es de mackinnon, por lo que se usaron los estandar de 1996 para los 3 casos.

Lectura Correcta: Carga el archivo .csv con header=None para evitar que la primera fila numérica se "coma" como si fuera el título de la columna. Inmediatamente recorta la data para operar única y estrictamente con el 80% correspondiente al Training, como dictan las instrucciones.

Cálculo Consistente de AIC (Muestra Efectiva $M$): Para que el AIC sea válido comparando un modelo con $p=1$ frente a uno con $p=5$, el número de observaciones empíricas ($N$) debe ser exactamente el mismo. El script calcula $p_{max}$ usando la regla de Schwert y fuerza a que todos los sub-modelos empiecen a predecir a partir del mismo punto en el tiempo, asegurando un Grid Search matemáticamente puro.

Construcción Dinámica Matricial ($X$): La función build_adf_matrix es el núcleo que transforma las ecuaciones $Caso 1, 2, 3$ del documento en una matriz real para poder aplicar el álgebra OLS estricta (np.linalg.inv(X.T @ X)). Añade np.ones si requiere drift (constante) y np.arange si requiere tendencia determinística.

Bucle de Integración (d): El motor inicia evaluando la serie tal cual ($d=0$). Busca el $p$ óptimo en cada uno de los 3 casos evaluando el AIC. Tras hallar el menor, lo compara con el valor crítico $C(\alpha, T)$. Si cualquiera de los casos detecta estacionariedad, el bucle se detiene, saca el reporte a adf.csv e imprime el éxito por consola. Si no, aplica el diff_series importado de utility.py y repite con $d+1$.

# TRN:

Alineación de Índices (build_phase2_matrix): Esta es la clave del pronóstico directo multi-paso. En un modelo ARMA clásico, tu objetivo es $z_t$. Pero el profesor pide que el objetivo (target) sea $\hat{Y}*{t+h}$. Por lo tanto, la matriz de regresión alinea las variables explicativas (el pasado que conocemos hoy, es decir, tiempo $t$) con el futuro $t+h$. Esto garantiza modelos independientes para cada $h$.

Grid Search Eficiente: El sistema evalúa combinaciones desde $p=0, q=0$ hasta $10, 10$ como se solicitó. Si un modelo es algebraicamente imposible (por singularidad de matriz o falta de datos en combinaciones altas como $p=10, q=10$ con $m=60$), un bloque try-except salta limpiamente esa iteración evitando bloqueos en la ejecución.

Persistencia Dinámica (train.csv): En lugar de exportar un archivo CSV con un montón de columnas vacías, opté por una estructura relacional larga (Horizonte h | Tipo de Coeficiente | Índice | Valor). Esto permite que el siguiente archivo (tst.py) reconstruya la ecuación exacta sin importar qué valores ganaron en el Grid Search.

# TST:

Reconstrucción y Persistencia (load_model_parameters): En lugar de volver a entrenar la serie, el script extrae los hiperparámetros ganadores (p, d, q, m) y los arrays de coeficientes exactos $\phi$ y $\theta$ que guardamos en la fase anterior para cada horizonte.

Generación de Innovaciones para Test: Dado que el OLS Directo requiere los residuos $\epsilon_{t-q}$ como variables explicativas, reconstruimos el modelo autorregresivo largo (Fase 1) solo usando datos de entrenamiento. Luego, aplicamos esa matriz de pesos (beta_p1) a la serie completa para obtener un vector continuo de residuos que el set de prueba pueda utilizar legítimamente sin filtrar datos del futuro.

Alineamiento Temporal y Recuperación: El bucle principal itera exclusivamente en los índices del set de validación (el 20% final). Para cada instante $t$, construye sus rezagos y efectúa la suma producto con los pesos $\beta_h$ para hallar la predicción diferenciada $\hat{z}_{t+h}$. Luego llama a la función binomial de utility.py entregándole el historial de la serie original y el grado de integración $d$ para llevarlo de vuelta al dominio base ($\hat{Y}_{t+h}$).

Jarque-Bera y Visualización: Calcula los errores puros ($Y_{t+h} - \hat{Y}_{t+h}$) y se los envía al test JB. Inmediatamente compara el estadístico de retorno contra el valor crítico estático de 5.991. Finalmente, plotea la línea de valores reales en negro sólido y la predicción en rojo segmentado (crimson y --), exportando todo automáticamente a archivos .png.

# CAMBIOS POR DATA LEAKAGE

Encapsulamiento del día t: Ahora el bucle principal es el tiempo (for t in range(...)), no el horizonte $h$. Esto nos obliga a posicionarnos en un día concreto y simular que el futuro no existe.

Carga total de parámetros: load_model_parameters ahora trae los 5 modelos (o los que haya en train.csv), no solo los tres que pide graficar. Necesitas obligatoriamente el modelo $h=2$ para recuperar la serie y poder calcular el $h=3$.

El Buffer Recursivo (y_hist_recursivo.append): En la línea 85, la predicción $\hat{Y}*{t+h}$ se inyecta directamente al final de la lista histórica. Así, en la siguiente iteración del ciclo h, la función utility.recover_prediction consume la curva proyectada en lugar de la curva real, produciendo el aumento de error escalonado y los desfasajes en los gráficos correspondientes a un pronóstico a futuro genuino.

# En manual de usuario

mencionar que los valores de p y q encontrados están en train.csv

¡Exacto! Esa aclaración de tu profesor tiene todo el sentido matemático del mundo. Al usar `ts_taller2.csv` (que tiene una estructura distinta al ejemplo del PDF), el Grid Search encontró que su "huella digital" óptima es un ARIMA(4,2,1), y por ende la curva de degradación del mNSE será distinta.

Para responder a tu pregunta de forma directa: **Sí, los valores de p y q ya se están guardando de forma permanente.**

En el código que armamos para `trn.py`, diseñé una estructura de persistencia robusta. Además de imprimirse en la consola, estos hiperparámetros se guardan directamente en el archivo **`train.csv`**.

Si abres ese archivo generado, notarás que las primeras cuatro filas están reservadas exclusivamente para esta metadata. Se ven así (bajo la estructura de columnas `h, type, index, value`):

- `0, meta, p, 4`
- `0, meta, d, 2`
- `0, meta, q, 1`
- `0, meta, m, 15`

Usé `h=0` y `type=meta` como una "etiqueta secreta" para almacenar esta configuración general sin mezclarla con los coeficientes AR y MA de los horizontes de predicción ($h=1, 2, 3...$).

De hecho, el script `tst.py` depende vitalmente de esto. En sus primeras líneas de ejecución, la función `load_model_parameters` va a leer `train.csv`, extrae esos valores exactos de `p` y `q`, y con ellos reconstruye las matrices algebraicas antes de evaluar el 20% de los datos.

Por lo tanto, ese requerimiento de tu profesor está **100% cubierto y operativo** en tu código actual. Tienes toda la trazabilidad que exige el taller.

Con el código ya estabilizado y funcionando, ¿hay algo de la documentación (el manual de usuario en PDF que pide el taller) en lo que necesites apoyo para estructurarlo?
