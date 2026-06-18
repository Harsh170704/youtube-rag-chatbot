from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
load_dotenv()
from langchain_core.runnables import RunnableParallel, RunnablePassthrough,RunnableSequence, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

llm= HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", 
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.2,
)

model= ChatHuggingFace(llm=llm)

#building a chain(first part of parallel chain) helper function

def format_doc(retrieved_docs):
    context_text="\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

#step 1: get the transcript of the video
# video_id="fNk_zzaMoSs"
def create_chain(video_id):

 try:
    #don't care about the language of the transcript
    transcript_list=YouTubeTranscriptApi().fetch(video_id,languages=['en','en-US','hi',])

    #flattern it to plane text
    transcript=" ".join(chunk.text for chunk in transcript_list)
    # print(transcript)
 except TranscriptsDisabled:
    print("transcript is disabled for this video")

#step 1b: split the transcript into smaller chunks
 splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
 chunks=splitter.create_documents([transcript])
# print(len(chunks))

#step1 b&c= chunks embedding and store in vector database
 embedding= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
 vector_store=Chroma.from_documents(chunks, embedding)

#now we will form retriever
#step 2 :Retriever
 retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})
# print(retriever.invoke("what are vectors?"))

#step 3: now to create prompt for augmentation part
 prompt=PromptTemplate(
    template="""
    You are a helpful assistant 
    Answer only from the provided transcript context
    In case context is insufficient, just say you don't know
    {context}
    Question:{question} 
    """, 
    input_variables=['context','question']
 )

# question="is the topic alpha zero discussed in the video, if yes then what is it?"
# retrieved_docs=retriever.invoke(question)

# #concatinate these 4 retrieved docs in one string to form the context
# context_text="\n\n".join(doc.page_content for doc in retrieved_docs)
# final_prompt=prompt.invoke({'context':context_text,'question':question})
# #print(final_prompt)

# #step 4:generation
# answer = model.invoke(final_prompt)
# # print(answer)



 parallel_chain= RunnableParallel({
    'context': retriever| RunnableLambda(format_doc),
    'question': RunnablePassthrough()
 })

# print(parallel_chain.invoke("what is scaling?"))

# second of chains
 parser=StrOutputParser()
 mainchain= parallel_chain|prompt|model|parser
 return mainchain
# print(mainchain.invoke("what is scaling?"))
# testing

chain = create_chain(
    "fNk_zzaMoSs"
)


print(chain.invoke("what is scaling?")

)