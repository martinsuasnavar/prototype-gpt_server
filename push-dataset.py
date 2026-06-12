import json
import os

JSON_PATH = "intent.json"
TXT_OUTPUT_PATH = "dataset.txt"

def build_text_dataset():
    if not os.path.exists(JSON_PATH):
        print(f"Error: No se encontró el archivo {JSON_PATH}")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_lines = []

    for item in data:
        # Reading the updated English keys from your JSON
        user_text = item.get("user", "").strip().lower()
        agent_text = item.get("agent", "").strip().lower()
        
        if user_text and agent_text:
            # Structuring the raw text dataset with english prefixes
            block = f"user: {user_text}\nagent: {agent_text}\n###\n"
            dataset_lines.append(block)

    final_text = "".join(dataset_lines)

    with open(TXT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"¡Éxito! Dataset creado en '{TXT_OUTPUT_PATH}'")
    print(f"Total de caracteres: {len(final_text)}")
    
    unique_chars = sorted(list(set(final_text)))
    print(f"Tamaño del vocabulario detectado: {len(unique_chars)} caracteres.")

if __name__ == "__main__":
    build_text_dataset()