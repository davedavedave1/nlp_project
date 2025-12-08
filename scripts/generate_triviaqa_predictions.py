import json
import os
from tqdm import tqdm
from rag_system import Rag_system

# ----------------------------
# USER CONFIGURATION
# ----------------------------
TRIVIAQA_DATASET_PATH = "./necessary_parts_triviaqa/verified-wikipedia-dev.json"
OUTPUT_PREDICTIONS_PATH = "./predictions/first_prediction_give_meaningful_name.json"



# ----------------------------
# MAIN SCRIPT
# ----------------------------
def main():

    #load rag systen
    rag= Rag_system()


    # Load TriviaQA dataset
    with open(TRIVIAQA_DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    
    questions = []
    if 'Data' in data:
        for entry in data['Data']:
            if 'QuestionId' in entry and 'Question' in entry:
                questions.append({
                    "id": entry['QuestionId'],
                    "question": entry['Question']
                })
            elif 'Question' in entry and 'QuestionId' not in entry:
                for q in entry['Question']:
                    questions.append({
                        "id": q['QuestionId'],
                        "question": q['QuestionText']
                    })
    else:
        print("Warning: Dataset structure not recognized. Trying top-level list...")
        for entry in data:
            questions.append({
                "id": entry['QuestionId'],
                "question": entry['Question']
            })

    print(f"Total questions found: {len(questions)}")

    # Generate predictions
    predictions = {}
    for q in tqdm(questions, desc="Generating answers"):
        question_id = q['id']
        question_text = q['question']
        answer = rag.run(question_text)
        predictions[question_id] = answer

    # Save predictions
    os.makedirs(os.path.dirname(OUTPUT_PREDICTIONS_PATH), exist_ok=True)
    with open(OUTPUT_PREDICTIONS_PATH, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"Predictions saved to {OUTPUT_PREDICTIONS_PATH}")

if __name__ == "__main__":
    main()
