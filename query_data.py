import argparse 
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Awser the question based only on following context:

{context}

---

Answer the question based on the above context: {question}

"""

def main():
    load_dotenv('.env')

    #Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    #Prepare the DB
    embeddings_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings_function)

    #search the DB
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return 
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    promt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    promt = promt_template.format(context=context_text, question=query_text)
    print(promt)

    model = ChatOpenAI()
    response_text = model.predict(promt)    

    sources = [doc.metadata.get("sources", None) for doc, _score in results]
    formatted_reponse = f"Response: {response_text}\nSources:{sources}"
    print(formatted_reponse)


if __name__ == "__main__":
    main()