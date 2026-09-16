from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFacePipeline
from langchain_deepseek import ChatDeepSeek 
from dotenv import load_dotenv
load_dotenv()

import os


def get_llm():
    llm = ChatDeepSeek(
        model="deepseek-llm-7b-chat",
        temperature=0.1,
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY")
    )
    return llm


def split_transcribe(transcript: str) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_text(transcript)
    return chunks


def summarize_transcript(transcript: str) -> str:
    llm = get_llm()

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            (
                "human",
                "Summarize the following transcript:\n\n{transcript}"
            )
        ]
    )

    map_chain = prompt_template | llm | StrOutputParser()

    chunks = split_transcribe(transcript)

    summaries = []

    for chunk in chunks:
        summary = map_chain.invoke({
            "transcript": chunk
        })
        summaries.append(summary)

    combined = "\n\n".join(summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            (
                "human",
                "Combine the following summaries into a single "
                "coherent summary:\n\n{combined_summaries}"
            )
        ]
    )

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(
            lambda x: {"combined_summaries": x}
        )
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    return combined_chain.invoke(combined)


def generate_title(transcript: str) -> str:
    llm = get_llm()

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            (
                "human",
                "Generate a concise and descriptive title "
                "for the following transcript:\n\n{transcript}"
            )
        ]
    )

    title_chain = prompt_template | llm | StrOutputParser()

    return title_chain.invoke({
        "transcript": transcript[:2000]
    })


# Keep this alias because main.py imports `summarize`
def summarize(transcript: str) -> str:
    return summarize_transcript(transcript)