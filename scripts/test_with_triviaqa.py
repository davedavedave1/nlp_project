import json
from rag_system import rag_system

def test_with_triviaqa(file_path):
    # Open and load the JSON file
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questionsAsked=0
    questionsRight=0
    questionsWrong=0

    
    for item in data["Data"]:
        print(item["Question"])
        print(item["Answer"]["Aliases"]+item["Answer"]["NormalizedAliases"])#Maybe we dont have to take both => Take a look at the paper/docs of triviaqa
        #das RAG System mit der Frage testen und dann die Parameter upgraden
        if rag_system(item["Question"]) in (item["Answer"]["Aliases"]+item["Answer"]["NormalizedAliases"]):
            questionsRight= questionsRight+1
        else:
            questionsWrong= questionsWrong+1
        questionsAsked=questionsAsked+1

        print("questions asked: "+str(questionsAsked)+" Right Answers: "+str(questionsRight))
    

    

    


if __name__ == "__main__":
    test_with_triviaqa("./necessary_parts_triviaqa/wikipedia-train_new.json")