import embedder
from test_with_triviaqa import test_with_triviaqa
from pathlib import Path
import subprocess

ROOT_DIR    = Path(__file__).resolve().parent.parent
TRIVIA_DIR  = ROOT_DIR / "necessary_parts_triviaqa"

def run_create_partition():
    print("Running create_partition.py inside:", TRIVIA_DIR)
    # Run the script exactly as if user executed: python create_partition.py
    subprocess.run(
        ["python", "create_partition.py"],
        cwd=str(TRIVIA_DIR),
        check=True
    )


if __name__ == "__main__":
    if not Path("./necessary_parts_triviaqa/wikipedia-train_new.json").exists():
        run_create_partition()

    testfile = "./necessary_parts_triviaqa/wikipedia-development_new.json"
    prefix_path_to_db = "./databases/FAISS-DB_embeddingModel~paraphrase-MiniLM-L3-v2_chunkSize~"
    suffix_path_to_db = "_chunkOverlap~30_WITH_METADATA"

    chunks = [256, 512, 1024, 2048, 4096, 8192]
    for chunk in chunks:
        if not Path(prefix_path_to_db + str(chunk) + suffix_path_to_db).exists():
            print("Starting embedding chunk:",chunk)
            embedder.embedder(chunk,30, False)


    for chunk,n in [(256, 64), (512, 32), (1024, 16), (2048, 8), (4096, 4), (8192, 2)]:
        res= test_with_triviaqa(testfile, prefix_path_to_db + str(chunk) + suffix_path_to_db, n, n//2, True, debug=False) + " \n"
        with open("/content/d/MyDrive/Colab Notebooks/nlp_project_files/results_colab_round1.txt", "a") as t:
            t.writelines(res)

