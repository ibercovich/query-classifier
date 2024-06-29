"""batch query classifier using GPT4"""

import time

import pandas as pd
from openai import OpenAI

# locations = [
#     "ar",
#     "bo",
#     "cl",
#     "co",
#     "cr",
#     "cu",
#     "do",
#     "ec",
#     "es",
#     "general",
#     "gt",
#     "hn",
#     "mx",
#     "ni",
#     "pa",
#     "pe",
#     "pr",
#     "py",
#     "sv",
#     "us",
#     "uy",
#     "ve",
# ]

categories = [
    "Historia",
    "Religión",
    "Deportes",
    "Geografía",
    "Ciencia",
    "Datos",
    "Transacciones",
    "Entretenimiento",
    "Locales",
    "Bienestar",
    "Noticias",
    "Educación",
    "Técnicas",
    "Estilo",
    "Negocios",
    "Empleo",
    "Misceláneas",
]


client = OpenAI(api_key="")


def gpt_classify(queries):
    """Classifies a list of queries based on the categorization in the prompt below"""
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """ 
    Dado un conjunto de consultas de búsqueda, clasifica cada consulta en una de las siguientes categorías:

    Historia: Información histórica, Biografías, Eventos, Cronologías.
    Religión: religiones y creencias, Textos sagrados y escrituras, Prácticas y rituales religiosos.
    Deportes: Resultados y estadísticas deportivas, Atletas, Reglas y regulaciones deportivas, Equipos y ligas.
    Geografía: Datos geográficos y mapas, países y ciudades, Climas y ecosistemas, Demografia.
    Ciencia: Información científica, Descubrimientos y avances, experimentos, Biología, química, física, etc.
    Datos: Estadísticas y datos numéricos, Cifras económicas y financieras, Datos de censos y encuestas.
    Transacciones: Búsqueda de productos, búsqueda de servicios, reservas y reservas (vuelos, hoteles), compras y comercio.
    Entretenimiento: Películas y programas de televisión, música, juegos, celebridades y chismes, eventos y conciertos.
    Locales: Negocios locales, restaurantes y cafeterías, servicios locales (fontaneros, electricistas), eventos locales.
    Bienestar: Síntomas y condiciones, medicamentos y tratamientos, dieta y nutrición, fitness y ejercicio, salud mental.
    Noticias: Noticias de última hora, política, economía y finanzas, deportes, noticias tecnológicas.
    Educación: Temas académicos, cursos y tutoriales, escuelas y universidades, artículos y trabajos de investigación.
    Técnicas: Software y aplicaciones, programación y codificación, gadgets y electrónica, solución de problemas.
    Estilo: Moda y belleza, hogar y jardín, viajes y turismo, relaciones y consejos.
    Negocios: Asesoría legal, planificación financiera, inversiones y mercado de valores, información tributaria.
    Empleo: Búsqueda de empleo, consejos para currículums, consejos para entrevistas, desarrollo profesional.
    Misceláneas: Clima, horóscopos, trivias, curiosidad general.

    Por favor, clasifica las siguientes consultas de búsqueda en una de las categorías anteriores.
    Devuelve un csv en el mismo formato incluyendo la columna "categoria" donde solo pones el nombre de la categoria como "Empleo" o "Noticias".

    por ejemplo

    query,  categoria

    cuando fue la ultima vez que olimpia salio campeon,Deportes
    qué visitar en quintero,Locales
    para que fue creado windows mobile,Técnicas
    para que es el gel polish,Estilo
    cuando se descubrio pompeya y herculano,Historia
    que invento zeppelin,Historia
    porque da la sensacion de orinar a cada rato,Bienestar
    qué le pasó a benjamín galindo,Entretenimiento
    porque es bueno obedecer a dios,Religión
    que lenguas se hablan en noruega,Geografía
    a dónde viven los camaleones,Ciencia
    que puede hacer una computadora,Técnicas
    en qué ríos desemboca el lago tanganyika,Geografía
    que conocer en los angeles california,Locales
    quienes eran los bolcheviques,Historia
    quienes forman parte del sistema solar,Ciencia
    cuanto late el corazon al hacer ejercicio,Bienestar
    cuantas veces fue el hombre a la luna,Historia
    que hizo fujimori en su gobierno,Historia
    como darse cuenta si estoy intoxicada,Bienestar
    que manifestacion hay el domingo en valencia,Noticias
    cuándo hace efecto la creatina,Ciencia
    ...

    No uses herramientas como Python. Simplemente lee el csv abajo y produce uno nuevo como requerido. No des ninguna explicacion-- solo responde con el csv.
    """,
            },
            {
                "role": "user",
                "content": "```csv\n" + "\n".join(queries) + "```",
            },
        ],
    )

    return completion.choices[0].message.content


# Login using e.g. `huggingface-cli login` to access this dataset
splits = {
    "train": "general/train-00000-of-00001.parquet",
    "test": "general/test-00000-of-00001.parquet",
}
df = pd.read_parquet("hf://datasets/spanish-ir/google_qrels/" + splits["train"])

# Shuffle the DataFrame
df = df.sample(frac=1).reset_index(drop=True)

# Initialize a dictionary to store counts
category_counts = {category: 0 for category in categories}

# Loop to create 100 CSV files, each with 100 items
for i in range(10):
    # Select 100 values from the specified column
    start_idx = i * 100
    end_idx = start_idx + 100
    selected_values = df["query"].iloc[start_idx:end_idx]
    completions = gpt_classify(selected_values.tolist())
    print(f"Batch {i} of 100")
    print(completions)
    # Count appearances of each category in the long string
    for category in categories:
        category_counts[category] += completions.count(category)

    # Display the counts
    for category, count in category_counts.items():
        print(f"{category}: {count}")

    time.sleep(2)


# Example Output

# Historia: 97
# Religión: 50
# Deportes: 25
# Geografía: 85
# Ciencia: 190
# Datos: 16
# Transacciones: 15
# Entretenimiento: 59
# Locales: 10
# Bienestar: 256
# Noticias: 9
# Educación: 70
# Técnicas: 33
# Estilo: 37
# Negocios: 24
# Empleo: 4
# Misceláneas: 19
