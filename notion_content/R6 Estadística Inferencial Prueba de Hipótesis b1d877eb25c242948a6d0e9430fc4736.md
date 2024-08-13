# R6 Estadística Inferencial: Prueba de Hipótesis

Prueba de Hipótesis

es como una Inferencia formal

dices un valor y tienes que plantear bien la hipótesis

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled.png)

Tipos de hipótesis

Una hipótesis estadística es una afirmación sobre los parámetros de una o más poblaciones

Afirmación sobre algo

Hipótesis nula

Ho

Con igualdad (=)

Cuando se contrasta usando la información muestral.

Lo que se quiere corroborar usando una muestra.

Hipótesis alterna

H1

hipótesis aceptada es la que debe ser aceptada si se rechaza la nula

El no rechazar la hipótesis nula no implica que sea cierta

Tipos de hipótesis

X es normal (media 5,desviación 10)

Y es binomial con pi 0.25

media de UL es diferente a 21

Fracción de unidades defectuosas producidas en cierto proceso es menor a 7%. pi menor a 0.07

Usando información muestral se quiere corroborar

3 tipos de planteamiento de hipótesis

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%201.png)

Unilateral 

teta mayor igual al valor observado

Común en los 3

que la igualdad siempre esta en la hipótesis nula

Hipótesis nula siempre =

Hay que plantear bien las hipótesis

**Tipos de Error en estadística**

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%202.png)

Comparación entre la

Situación Real

Decisión estadística

Si una verdad absoluta se rechaza con una muestra

Es un error de Tipo 1

Por qué algo verdadero sea 

Comprobadamente, por ejemplo la edad de alguien

Sale en la muestra que es falso

Diseño muestral pésimo

Inadecuado instrumento 

Inadecuado recojo de datos 

errores

**Error tipo 2**

Situación real

si es falso absolutamente comprobado

la hipótesis muestral dice que es verdadera

Entonces es Error tipo 2

Inadecuado recojo de datos

Los otros casos son decisiones correctas.

También tiene un punto de vista probabilístico

Tipo de Error

Error tipo 1

Rechaza algo verdadero

Error tipo 2

Acepta algo falso

Nivel de confianza (1-alfa)

Probabilidad de aceptar una hipótesis nula que es verdadera

Nivel de Significancia alfa

alfa es el nivel de significancia, es la probabilidad de cometer el error de tipo 1

Valor de fijado por la persona de investigación

5% es estándar, 1% precisión alta en medicina

Potencia de la prueba (1-beta)

lo usan bio estadístico

probabilidad de rechazar una hipótesis nula que es falsa

Probabilidad de cometer tipo 2

beta

Región crítica

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%203.png)

Antiguamente se usaban manuales

Si el estadístico de prueba caía en los límites Región de Aceptación

se acepta la hipótesis nula

Luego lo demás en Región crítica o región de rechazo

se rechazaba la hipótesis nula

Región rechazo conjunto todos los valores que conducen a rechazar H0.

La división de la forma de la hipótesis alternativa, del nivel de significación y de la distribución muestral del valor estadístico.

la hipótesis alternativa se corresponde a la región crítica

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%204.png)

pruebas unilaterales

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%205.png)

prueba bilateral es cuando la hipotesis alterna es con igualdad

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%206.png)

P valor

P value

Valor de probabilidad

Concordancia entre datos observados en la muestra y Ho.

Un valor de probabilidad bajo indica que lo más probable es que dicha hipótesis sea falsa.

También se conoce como p value o signficancia (SPSS) de la prueba

Regla de decisión: rechazar hipotesis nula si p menor que alfa.

Para decidir sobre una hipotesis se usa la regla

Ahora se usa el p valor con la significancia

ya no la región

Se usa para todo

—

Procedimiento general de la prueba de hipótesis

Teorema de Limite Central

formular la hipotesis correcta

elegir el nivel de signficación alfa

obtener el p valor

si p valor menor que alfa se rechaza Ho

si p valor es mayor igual que alfa no se rechaza Ho

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%207.png)

Conclusiones

Decidir el rechazo de la hipótesis nula si EP cae en la región crítica. En caso contrario no rechazar la hipotesis nula 

Prueba de Hipotesis para la media.

Prueba de hipotesis para mu

Caso 1: delta cuadrado conocida

Supuestos

población normal muestra al azar

Pruebas de hipotesis de 3 tipos

unilateral izq, bilateral y unilateral derecha

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%208.png)

Estadistico de prueba 

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%209.png)

Se agrega un factor de correccion para poblaciones finitas

- **zcal**, o valor Z calculado, es el valor que obtienes al realizar tu prueba de hipótesis. Es una medida de cuántas desviaciones estándar está tu muestra de la media de la población. Se calcula utilizando la fórmula:

zcal=(σ/n)(xˉ−μ)

donde:

- xˉ es la media de la muestra,
- μ es la media de la población,
- σ es la desviación estándar de la población, y
- n es el tamaño de la muestra.
- **zcrit**, o valor Z crítico, es el valor que determina los límites de la región de rechazo en una prueba de hipótesis. Este valor se obtiene de la tabla de distribución normal estándar, basándose en el nivel de significancia que hayas elegido para tu prueba.

es un insumo directo para calcular el p valor

antes era la región para calcular

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%2010.png)

Caso 1 delta 2 desconocida

cambia el estadístico de prueba

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%2011.png)

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%2012.png)

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%2013.png)

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%2014.png)

Cambió

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%2015.png)

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%2016.png)

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%209.png)

Hipótesis nula quiero Corroborar 

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%2017.png)

Es al revés

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%2018.png)

debe ser

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%2019.png)

a pesar de que la hipótesis es al revés se cumple con lo que se requiere corroborar

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%2020.png)

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%209.png)

Prueba de hipótesis en proporción

Variables categóricas

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%2021.png)

Valor de aceptación

Porcentaje de defectuosos

proporción de defectuosos es inferior al 3%

Control estadístico y calidad

Six Sigma

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%2022.png)

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%2023.png)

Dicotomico

Bueno o malo

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%2024.png)

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%2025.png)

Intensa campaña publicitaria

muestra n 450

p = 105/450

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%2026.png)

Campaña de marketing no obtuvo apoyo si la hipótesis nula es correcta

Datos <- rnorm(n=25, mean = 2.065, sd = 0.07)

t.test(Datos, alternative='two.sided',
conf.level=0.99, mu=2)

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%2027.png)

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%2028.png)

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%2029.png)

Greater

Alfa esta con el conf.level

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%2030.png)

Normalidad en base a R

![Untitled](R6%20Estadi%CC%81stica%20Inferencial%20Prueba%20de%20Hipo%CC%81tesis%20b1d877eb25c242948a6d0e9430fc4736/Untitled%2031.png)

buenas disculpa cuál es la hipótesis para la proporción en este caso:
#------------------------
prop.test(x=105, n=450, p=0.2, alternative='greater',
conf.level=0.975)

La función **`prop.test()`** en R se utiliza para realizar una prueba de hipótesis sobre proporciones. En este caso, la hipótesis nula (H0) y la hipótesis alternativa (H1) están relacionadas con la proporción poblacional 𝑝*p*.

Aquí están los detalles específicos de la prueba que proporcionaste:

```r
rCopiar código
prop.test(x=105, n=450, p=0.2, alternative='greater', conf.level=0.975)

```

### **Componentes de la Prueba:**

1. **x=105**: El número de éxitos observados (105).
2. **n=450**: El tamaño de la muestra (450).
3. **p=0.2**: La proporción hipotética bajo la hipótesis nula (0.2).
4. **alternative='greater'**: La hipótesis alternativa es que la proporción verdadera es mayor que 0.2.
5. **conf.level=0.975**: El nivel de confianza para el intervalo de confianza (97.5%).

### **Hipótesis de la Prueba:**

- **Hipótesis Nula (H0)**: La proporción verdadera 𝑝*p* es igual a 0.2.
    
    𝐻0:𝑝=0.2*H*0:*p*=0.2
    
- **Hipótesis Alternativa (H1)**: La proporción verdadera 𝑝*p* es mayor que 0.2.
    
    𝐻1:𝑝>0.2*H*1:*p*>0.2
    

En resumen, estás probando si la proporción observada de éxitos (105/450) es significativamente mayor que la proporción hipotética de 0.2, con un nivel de confianza del 97.5%. La prueba es de una cola porque la hipótesis alternativa es que la proporción es **mayor** que 0.2.