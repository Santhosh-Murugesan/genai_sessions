from sentence_transformers import SentenceTransformer
import json
import faiss

# Download from the 🤗 Hub
model = SentenceTransformer("google/embeddinggemma-300m")

documents = [
    "Mercury is the closest planet to the Sun and has a very short orbital period.",
    "Venus is often called Earth's twin because it is similar in size and structure to Earth.",
    "Mars is known as the Red Planet because iron minerals in its soil and rocks give its surface a reddish appearance.",
    "Jupiter is the largest planet in the solar system and is famous for its Great Red Spot.",
    "Saturn is a gas giant best known for its spectacular system of rings made primarily of ice and rock.",
    "Uranus rotates on its side compared with most other planets and has a blue-green appearance.",
    "Neptune is the eighth planet from the Sun and is known for its powerful winds and deep blue color.",
    "Mars has two small natural satellites called Phobos and Deimos.",
    "The atmosphere of Mars is very thin and consists mostly of carbon dioxide, with smaller amounts of nitrogen and argon.",
    "Earth is the third planet from the Sun and is currently the only known planet to support life.",
    "The Moon is Earth's natural satellite and takes approximately 27.3 days to orbit Earth.",
    "Jupiter has dozens of known moons, including the large Galilean moons Io, Europa, Ganymede, and Callisto.",
    "Mars has a giant volcano called Olympus Mons, which is the largest known volcano in the solar system.",
    "Venus has a very thick atmosphere dominated by carbon dioxide and has extremely high surface temperatures.",
    "The asteroid belt is located between the orbits of Mars and Jupiter and contains millions of rocky objects.",
    "Mars experiences seasons because its rotational axis is tilted, similar to Earth's axial tilt.",
    "Saturn has a lower average density than water, making it the least dense planet in the solar system.",
    "Mars appears reddish when observed from Earth, which is why ancient civilizations associated the planet with its distinctive red color.",
    "The Sun is a star located at the center of our solar system and contains most of the solar system's mass.",
    "Pluto was classified as a dwarf planet in 2006 and is located in the distant Kuiper Belt."
]

document_embeddings = model.encode_document(documents)

# Create a FAISS index
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the index
index.add(document_embeddings)

# Save the FAISS index
faiss.write_index(index, "document_embeddings.faiss")

# Save the metadata to a JSON file
id = 0
meta_data = {}
for chunk in documents:
    meta_data[id] = chunk
    id += 1

with open("document.json", "w") as f:
    json.dump(meta_data, f)
