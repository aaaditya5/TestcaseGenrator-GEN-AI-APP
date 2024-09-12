import os
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI
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
    
    # New chain for generating unit test cases
    f3=PromptTemplate(
        input_variables=['code'],
        template="""Write unit test cases for the following code using Python's unittest framework:{code}

Here are a few examples of unit test cases:
def test_add_negative_numbers(self):
    self.assertEqual(add(-2, 3), 1)

def test_add_zero(self):
    self.assertEqual(add(0, 0), 0)


Your turn! Write unit test cases for the provided code:
"""
    )
    chain=LLMChain(llm=llm,prompt=f3)
    
    over_all_chain=SimpleSequentialChain(chains=[chain],verbose=True)
    return over_all_chain


def answer_question(api_key,code):
    chain=load_chain(api_key)
    output=chain.run(input=code)
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
            label="Enter your Python code:",
            placeholder="e.g. def add(a, b):\n    return a + b"
        )
    ],
    outputs=gr.Textbox(label="Unit Test Cases"),
    title="Unit Test Case Generator",
    description="Enter your GenAI API key below and your Python code to generate unit test cases"
)

ifaces.launch()