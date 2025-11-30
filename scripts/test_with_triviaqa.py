import json
from rag_system import Rag_system



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


def test_with_triviaqa(file_path, path_to_db, how_many_docs):
    # Open and load the JSON file
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questionsAsked=0
    questionsRight=0
    questionsWrong=0

    rag = Rag_system(path_to_db,how_many_docs)
    

    
    for item in (data["Data"][:1000]):
        #print(item["Question"])
        #print(item["Answer"]["NormalizedAliases"])#Maybe we dont have to take both => Take a look at the paper/docs of triviaqa

        solution=item["Answer"]
        answer = normalize_answer(rag.run(item["Question"])) #put in lowercase and remove spaces in front and end

        print("Our answer is: "+answer)
        
        if answer in (solution["NormalizedAliases"]):
            questionsRight= questionsRight+1
        else:
            questionsWrong= questionsWrong+1
        
        questionsAsked=questionsAsked+1

        print("questions asked: "+str(questionsAsked)+" Right Answers: "+str(questionsRight))

    return "Benutzte db: "+str(path_to_db)+" Wieviele Documents werden retrieved: "+str(how_many_docs)+" questions asked: "+str(questionsAsked)+" Right Answers: "+str(questionsRight)
    

    
    


if __name__ == "__main__":
    #"./databases/FAISS-DB_embeddingModel~paraphrase-MiniLM-L3-v2_chunkSize~1024_chunkOverlap~30"
    testfile="./necessary_parts_triviaqa/wikipedia-development_new.json"
    prefix_path_to_db="./databases/FAISS-DB_embeddingModel~paraphrase-MiniLM-L3-v2_chunkSize~"
    suffix_path_to_db="_chunkOverlap~30"


    res=[]

    #res.append(test_with_triviaqa(testfile,+prefix_path_to_db+"1024"+suffix_path_to_db,16)+" \n")#53988 questions
    #res.append(test_with_triviaqa(testfile,prefix_path_to_db+"1024"+suffix_path_to_db,16)+" \n")
    
    #res.append(test_with_triviaqa(testfile,prefix_path_to_db+"512"+suffix_path_to_db,16)+" \n")
    

    #test_with_triviaqa(testfile,prefix_path_to_db+"512"+suffix_path_to_db,4)

    comment="""res.append(test_with_triviaqa(testfile,prefix_path_to_db+"512"+suffix_path_to_db,2)+" \n")
    with open("testresultsBACKUP1.txt", "w") as t:
        t.writelines(res)
    res.append(test_with_triviaqa(testfile,prefix_path_to_db+"512"+suffix_path_to_db,4)+" \n")
    res.append(test_with_triviaqa(testfile,prefix_path_to_db+"512"+suffix_path_to_db,8)+" \n")
    with open("testresults.txtBACKUP2", "w") as t:
        t.writelines(res)
    res.append(test_with_triviaqa(testfile,prefix_path_to_db+"512"+suffix_path_to_db,1)+" \n")
    #res.append(test_with_triviaqa(testfile,prefix_path_to_db+"512"+suffix_path_to_db,32)+" \n")
    res.append(test_with_triviaqa(testfile,prefix_path_to_db+"256"+suffix_path_to_db,1)+" \n")
    res.append(test_with_triviaqa(testfile,prefix_path_to_db+"256"+suffix_path_to_db,2)+" \n")
    with open("testresults.txtBACKUP3", "w") as t:
        t.writelines(res)
    res.append(test_with_triviaqa(testfile,prefix_path_to_db+"256"+suffix_path_to_db,4)+" \n")
    res.append(test_with_triviaqa(testfile,prefix_path_to_db+"256"+suffix_path_to_db,8)+" \n")
    res.append(test_with_triviaqa(testfile,prefix_path_to_db+"256"+suffix_path_to_db,16)+" \n")
    #res.append(test_with_triviaqa(testfile,prefix_path_to_db+"256"+suffix_path_to_db,32)+" \n")
    with open("testresults.txtBACKUP4", "w") as t:
        t.writelines(res)
    #res.append(test_with_triviaqa(testfile,prefix_path_to_db+"256"+suffix_path_to_db,64)+" \n")
    res.append(test_with_triviaqa(testfile,prefix_path_to_db+"2048"+suffix_path_to_db,1)+" \n")
    res.append(test_with_triviaqa(testfile,prefix_path_to_db+"2048"+suffix_path_to_db,2)+" \n")
    res.append(test_with_triviaqa(testfile,prefix_path_to_db+"2048"+suffix_path_to_db,4)+" \n")
    with open("testresults.txtBACKUP5", "w") as t:
        t.writelines(res)
    #res.append(test_with_triviaqa(testfile,prefix_path_to_db+"2048"+suffix_path_to_db,8)+" \n")
    res.append(test_with_triviaqa(testfile,prefix_path_to_db+"4096"+suffix_path_to_db,1)+" \n")
    res.append(test_with_triviaqa(testfile,prefix_path_to_db+"4096"+suffix_path_to_db,2)+" \n")
    #res.append(test_with_triviaqa(testfile,prefix_path_to_db+"4096"+suffix_path_to_db,4)+" \n")
    with open("testresults.txtBACKUP6", "w") as t:
        t.writelines(res)
    res.append(test_with_triviaqa(testfile,prefix_path_to_db+"8192"+suffix_path_to_db,1)+" \n")
    #res.append(test_with_triviaqa(testfile,prefix_path_to_db+"8192"+suffix_path_to_db,2)+" \n")"""
    res.append(test_with_triviaqa(testfile,prefix_path_to_db+"1024"+suffix_path_to_db,8)+" \n")
    with open("testresults_2.txt", "w") as t:
        t.writelines(res)
    
    

