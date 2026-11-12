from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
load_dotenv()

loader = PyPDFLoader("../data/evs_oil_price_shock.pdf")
raw_docs=loader.load()
#print(raw_docs[0].page_content)
print(f"Number of pages: {len(raw_docs)}")


from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(raw_docs)
print(f"Number of documents after splitting: {len(docs)}")
#print(docs[0].page_content)

from langchain_openai import ChatOpenAI,OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

from langchain_chroma import Chroma
vector_store=Chroma(embedding_function=embeddings,collection_name="rag_basic",persist_directory="./chroma_db")
vector_store.add_documents(documents=docs)
retriever=vector_store.as_retriever(search_kwargs={"k":3})
print("Vector store created and documents added successfully.")


# # direct search from vector store
# query = "How do electric vehicles impact oil prices?"
# results = retriever.invoke(query)
# print(f"Retrieved documents: {len(results)}")
# for i, doc in enumerate(results):
#     print(f"\n--- Result {i+1} ---")
#     print(doc.page_content)
#     print("Metadata:", doc.metadata)

from langgraph.graph import StateGraph,START,END,MessagesState
from langchain_core.documents import Document

#Graph State
class AgenticRagState(MessagesState):
    query :str
    retrieved_docs : list[Document]
    context :str
    generation :str

llm=ChatOpenAI(model="gpt-4o",temperature=0.0) 


# retrives the relavant documents from the vector store
def retrive_doc(state:AgenticRagState):
    docs=retriever.invoke(state["query"])
    context="\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_docs":docs,"context":context}

# generate a response using the retrived documents
from langchain_core.prompts import ChatPromptTemplate
def generate(state:AgenticRagState):
    query=state["query"]
    context=state.get("context") or ""
    prompt_template=ChatPromptTemplate.from_messages([
        ("system","You are a helpful assistant that answers questions based on the provided context.\n context:\n{context}"),
        ("human","Answer the following question based on the context:\n{query}")
    ])
    response=(prompt_template|llm).invoke({"context":context,"query":query})
    return {"generation":response.content}

graph_builder=StateGraph(AgenticRagState)

graph_builder.add_node("retrieve",retrive_doc)
graph_builder.add_node("generate",generate)

graph_builder.add_edge(START,"retrieve")
graph_builder.add_edge("retrieve","generate")
graph_builder.add_edge("generate",END)

graph=graph_builder.compile()

result=graph.invoke({"query":"How do electric vehicles impact oil prices?","messages":[]})

print("FINAL RESULT:")
print(result["generation"])






    






