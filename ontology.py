import pandas as pd
from rdflib import Graph, Namespace, RDF, RDFS, Literal, URIRef
from rdflib.plugins.sparql import prepareQuery, sparql

# Definisci il namespace
ontologia = Namespace("http://example.org/ontologia/canzoni#")

# Crea un grafo RDF
grafo = Graph()

# Carica il dataset in un dataframe
dataset = pd.read_csv("SpotifyFeatures.csv")

# Rinomina le colonne
dataset.rename(columns={"Track.Name": "Track_Name"}, inplace=True)
dataset.rename(columns={"Artist.Name": "Artist_Name"}, inplace=True)
dataset.rename(columns={"Beats.Per.Minute": "Beats_Per_Minute"}, inplace=True)
dataset.rename(columns={"Loudness.dB": "Loudness_db"}, inplace=True)

# Rimuovi gli spazi dai valori delle colonne "Track Name", "Artist Name" e "Genre"
dataset["Track_Name"] = dataset["Track_Name"].str.replace(' ', '_')
dataset["Artist_Name"] = dataset["Artist_Name"].str.replace(' ', '_')
dataset["Genre"] = dataset["Genre"].str.replace(' ', '_')

# Definisci le proprietà
track_name = ontologia.track_name
artist_name = ontologia.artist_name
genre = ontologia.genre
year = ontologia.year
beats_per_minute = ontologia.Beats_Per_Minute
energy = ontologia.energy
danceability = ontologia.danceability
loudness_db = ontologia.Loudness_db
liveness = ontologia.liveness
valence = ontologia.valence
length = ontologia.length
acousticness = ontologia.acousticness
speechiness = ontologia.speechiness
popularity = ontologia.popularity

# Definisci la risorsa per la classe "Song"
song_class_uri = URIRef(ontologia["Song"])

# Collega le proprietà alla classe "Song"
grafo.add((song_class_uri, RDF.type, RDFS.Class))
grafo.add((song_class_uri, track_name, RDF.Property))
grafo.add((song_class_uri, artist_name, RDF.Property))
grafo.add((song_class_uri, genre, RDF.Property))
grafo.add((song_class_uri, year, RDF.Property))
grafo.add((song_class_uri, beats_per_minute, RDF.Property))
grafo.add((song_class_uri, energy, RDF.Property))
grafo.add((song_class_uri, danceability, RDF.Property))
grafo.add((song_class_uri, loudness_db, RDF.Property))
grafo.add((song_class_uri, liveness, RDF.Property))
grafo.add((song_class_uri, valence, RDF.Property))
grafo.add((song_class_uri, length, RDF.Property))
grafo.add((song_class_uri, acousticness, RDF.Property))
grafo.add((song_class_uri, speechiness, RDF.Property))
grafo.add((song_class_uri, popularity, RDF.Property))

# Crea istanze della classe "Song" per le canzoni nel dataset
for index, row in dataset.iterrows():
    song_uri = URIRef(ontologia[f"Song{index + 1}"])
    grafo.add((song_uri, RDF.type, song_class_uri))
    grafo.add((song_uri, track_name, Literal(row["Track_Name"])))
    grafo.add((song_uri, artist_name, Literal(row["Artist_Name"])))
    grafo.add((song_uri, genre, Literal(row["Genre"])))
    grafo.add((song_uri, year, Literal(row["year"])))
    grafo.add((song_uri, beats_per_minute, Literal(row["Beats_Per_Minute"])))
    grafo.add((song_uri, energy, Literal(row["Energy"])))
    grafo.add((song_uri, danceability, Literal(row["Danceability"])))
    grafo.add((song_uri, loudness_db, Literal(row["Loudness_db"])))
    grafo.add((song_uri, liveness, Literal(row["Liveness"])))
    grafo.add((song_uri, valence, Literal(row["Valence"])))
    grafo.add((song_uri, length, Literal(row["Length"])))
    grafo.add((song_uri, acousticness, Literal(row["Acousticness"])))
    grafo.add((song_uri, speechiness, Literal(row["Speechiness"])))
    grafo.add((song_uri, popularity, Literal(row["Popularity"])))

# Definisci le proprietà RDF per i collegamenti tra le caratteristiche
has_name = ontologia.has_name
has_artist = ontologia.has_artist
has_genre = ontologia.has_genre
has_year = ontologia.has_year
has_beats_per_minute = ontologia.has_beats_per_minute
has_energy = ontologia.has_energy
has_danceability = ontologia.has_danceability
has_loudness_db = ontologia.has_loudness_db
has_liveness = ontologia.has_liveness
has_valence = ontologia.has_valence
has_length = ontologia.has_length
has_acousticness = ontologia.has_acousticness
has_speechiness = ontologia.has_speechiness
has_popularity = ontologia.has_popularity

# Collega le caratteristiche alle istanze delle canzoni
for index, row in dataset.iterrows():
    song_uri = URIRef(ontologia[f"Song{index + 1}"])

    # Collegamento con il nome della canzone
    name_uri = URIRef(ontologia["Name_" + row["Track_Name"]])
    grafo.add((song_uri, has_name, name_uri))

    # Collegamento con l'artista
    artist_uri = URIRef(ontologia["Artist_" + row["Artist_Name"]])
    grafo.add((song_uri, has_artist, artist_uri))

    # Collegamento con il genere
    genre_uri = URIRef(ontologia["Genre_" + row["Genre"]])
    grafo.add((song_uri, has_genre, genre_uri))

    # Collegamento con l'anno di pubblicazione
    year_uri = URIRef(ontologia["Year_" + str(row["year"])])
    grafo.add((song_uri, has_year, year_uri))

    # Collegamenti con altre caratteristiche
    beats_per_minute_uri = URIRef(ontologia["Beats_Per_Minute_" + str(row["Beats_Per_Minute"])])
    grafo.add((song_uri, has_beats_per_minute, beats_per_minute_uri))

    energy_uri = URIRef(ontologia["Energy_" + str(row["Energy"])])
    grafo.add((song_uri, has_energy, energy_uri))

    danceability_uri = URIRef(ontologia["Danceability_" + str(row["Danceability"])])
    grafo.add((song_uri, has_danceability, danceability_uri))

    loudness_db_uri = URIRef(ontologia["Loudness_db_" + str(row["Loudness_db"])])
    grafo.add((song_uri, has_loudness_db, loudness_db_uri))

    liveness_uri = URIRef(ontologia["Liveness_" + str(row["Liveness"])])
    grafo.add((song_uri, has_liveness, liveness_uri))

    valence_uri = URIRef(ontologia["Valence_" + str(row["Valence"])])
    grafo.add((song_uri, has_valence, valence_uri))

    length_uri = URIRef(ontologia["Length_" + str(row["Length"])])
    grafo.add((song_uri, has_length, length_uri))

    acousticness_uri = URIRef(ontologia["Acousticness_" + str(row["Acousticness"])])
    grafo.add((song_uri, has_acousticness, acousticness_uri))

    speechiness_uri = URIRef(ontologia["Speechiness_" + str(row["Speechiness"])])
    grafo.add((song_uri, has_speechiness, speechiness_uri))

    popularity_uri = URIRef(ontologia["Popularity_" + str(row["Popularity"])])
    grafo.add((song_uri, has_popularity, popularity_uri))

# Salva il grafo in un file
grafo.serialize("ontologia_canzoni.rdf", format="xml")

# Specifica il nome del file in cui desideri salvare il grafo
nome_file = "grafo_export.ttl"  # Sostituisci con l'estensione del formato che desideri, ad esempio .xml, .jsonld, etc.

# Salva il grafo in un file con il formato specificato
grafo.serialize(destination=nome_file, format="turtle")  # Puoi sostituire "turtle" con il formato desiderato

# Query SPARQL 1
query_string_1 = '''
    PREFIX ontologia: <http://example.org/ontologia/canzoni#>
    SELECT ?anno
    WHERE {
        ?song ontologia:has_artist ?artist .
        ?song ontologia:popularity ?popularity .
        ?song ontologia:year ?anno .
        FILTER (?popularity > 85)
    }
    ORDER BY DESC(?popularity)
    LIMIT 1
'''

# Prepara la query
query_1 = prepareQuery(query_string_1, initNs={"ontologia": ontologia})

# Esegui la query per ottenere l'anno della canzone più popolare
risultati = grafo.query(query_1)

# Stampa l'anno della canzone più popolare
for row in risultati:
    anno = row.anno
    print(f"Anno della canzone più popolare: {anno}")