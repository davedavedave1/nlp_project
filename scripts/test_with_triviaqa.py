import json
from rag_system import rag_system



import string
import re


def normalize_answer(s):#taken from https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def test_with_triviaqa(file_path):
    # Open and load the JSON file
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questionsAsked=0
    questionsRight=0
    questionsWrong=0

    
    for item in data["Data"]:
        print(item["Question"])
        print(item["Answer"]["NormalizedAliases"])#Maybe we dont have to take both => Take a look at the paper/docs of triviaqa

        solution=item["Answer"]
        answer = normalize_answer(rag_system(item["Question"])) #put in lowercase and remove spaces in front and end

        print("Our answer is: "+answer)
        
        if answer in (solution["NormalizedAliases"]):
            questionsRight= questionsRight+1
        else:
            questionsWrong= questionsWrong+1
        
        questionsAsked=questionsAsked+1

        print("questions asked: "+str(questionsAsked)+" Right Answers: "+str(questionsRight))
    

    
    


if __name__ == "__main__":
    test_with_triviaqa("./necessary_parts_triviaqa/wikipedia-train_new.json")#53988 questions