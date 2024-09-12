import os
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from dotenv import load_dotenv
load_dotenv()
os.environ['GOOGLE_API_KEY']=os.getenv("GOOGLE_API_KEY")

#for monitoring and validation 
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
def load_chain(api_key):
    os.environ['GOOGLE_API_KEY']=api_key
    llm=ChatGoogleGenerativeAI(model='gemini-pro',temperature=0.1)
    
    # Existing chains
    f1=PromptTemplate(
        input_variables=['user_in'],
        template="""Write the outline of the coding steps to develop the program {user_in} in five steps.Use python3 and be concise.\n\n"""

    )
    chain1=LLMChain(llm=llm,prompt=f1)
    
    f2=PromptTemplate(
        input_variables=['program'],
        template="""Write the python3 code for each steps of {program} that is described. Use python3 style.Be concise in the code and opinionated about framework choice"""
        

    )
    chain2=LLMChain(llm=llm,prompt=f2)
    
    # New chain for generating unit test cases
    f3=PromptTemplate(
        input_variables=['code'],
        template="""Write unit test cases for the following code:\n{code}\nUse Python's unittest framework and be concise."""
    )
    chain3=LLMChain(llm=llm,prompt=f3)
    
    over_all_chain=SimpleSequentialChain(chains=[chain1, chain2, chain3],verbose=True)
    return over_all_chain


def answer_question(api_key,question):
    chain=load_chain(api_key)
    output=chain.run(input=question)
    return output



import gradio as gr

ifaces = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(
            label="Your Google GEMINI key",
            placeholder="e.g. AIzaSyCuH3I4vmvM3VItx8cLlci408c"
        ),
        gr.Textbox(
            label="Write a python script to:",
            placeholder="e.g. Find the 10th number of the Fibonacci sequence"
        )
    ],
    outputs=gr.Textbox(label="User guide"),
    title="Your Code Generator",
    description="Enter your GenAI API key below and a description of your desired python project in 1-2 sentences"
)

ifaces.launch()
