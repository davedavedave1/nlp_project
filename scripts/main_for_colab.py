import embedder
import test_with_triviaqa
from pathlib import Path

from necessary_parts_triviaqa import create_partition

if __name__ == "__main__":
    if not Path("necessary_evidence_triviaqa/wikipedia-train_new.json").exists():
        create_partition.main()

    testfile = "./necessary_parts_triviaqa/wikipedia-development_new.json"
    prefix_path_to_db = "./databases/FAISS-DB_embeddingModel~paraphrase-MiniLM-L3-v2_chunkSize~"
    suffix_path_to_db = "_chunkOverlap~30"

    chunks = [256, 512, 1024, 2048, 4096, 8192]

    for chunk in chunks:
        if not Path(prefix_path_to_db + chunk + suffix_path_to_db).exists():
            embedder(chunk,30, False)

    res = []
    res.append(
        test_with_triviaqa(testfile, prefix_path_to_db + "256" + suffix_path_to_db, 64, 32, True, debug=False) + " \n")
    with open("resultswithrerankerbackup1.txt", "w") as t:
        t.writelines(res)
    res.append(
        test_with_triviaqa(testfile, prefix_path_to_db + "512" + suffix_path_to_db, 32, 16, True, debug=False) + " \n")
    with open("resultswithrerankerbackup2.txt", "w") as t:
        t.writelines(res)
    res.append(
        test_with_triviaqa(testfile, prefix_path_to_db + "1024" + suffix_path_to_db, 16, 8, True, debug=False) + " \n")
    with open("resultswithrerankerbackup3.txt", "w") as t:
        t.writelines(res)

    res.append(
        test_with_triviaqa(testfile, prefix_path_to_db + "2048" + suffix_path_to_db, 8, 4, True, debug=False) + " \n")
    with open("resultswithrerankerbackup4.txt", "w") as t:
        t.writelines(res)
    res.append(
        test_with_triviaqa(testfile, prefix_path_to_db + "4096" + suffix_path_to_db, 4, 2, True, debug=False) + " \n")
    with open("resultswithrerankerbackup5.txt", "w") as t:
        t.writelines(res)
    res.append(
        test_with_triviaqa(testfile, prefix_path_to_db + "8192" + suffix_path_to_db, 2, 1, True, debug=False) + " \n")
    with open("resultswithrerankerbackup6.txt", "w") as t:
        t.writelines(res)
