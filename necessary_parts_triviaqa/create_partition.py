#HAS TO BE RUN FROM INSIDE THE SAME FOLDER TO AVOID ISSUES WITH FINDING THE JSONS

import json

# --- Input and output file paths ---
INPUT_FILE = "wikipedia-train.json"
OUTPUT_FILE_1 = "wikipedia-development_new.json"   # first 7900 entries
OUTPUT_FILE_2 = "wikipedia-train_new.json"   # remaining entries

PARTITION_SIZE = 7900

def main():
    # Load original JSON
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract the list under "Data"
    entries = data.get("Data", [])
    
    # Partition the data
    part1 = entries[:PARTITION_SIZE]
    part2 = entries[PARTITION_SIZE:]

    # Save first partition
    with open(OUTPUT_FILE_1, "w", encoding="utf-8") as f:
        json.dump({"Data": part1}, f, indent=4, ensure_ascii=False)

    # Save second partition
    with open(OUTPUT_FILE_2, "w", encoding="utf-8") as f:
        json.dump({"Data": part2}, f, indent=4, ensure_ascii=False)

    print(f"Done! First file has {len(part1)} items, second has {len(part2)} items.")

if __name__ == "__main__":
    main()
