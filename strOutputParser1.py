# same code now with the help of strOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct", #this model cant give structire output, so we will use the text generation task to get the output in text format and then we will parse it to get the structured output
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

#1 Prompt1 : detailed report
template1 = PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables=['topic']
)

#2 Prompt2 : summary
template2 = PromptTemplate(
    template='write a 5 line summary of the following report: {report}',
    input_variables=['report']
)

parser = StrOutputParser()
chain = template1 | model | parser | template2 | model

result = chain.invoke({'topic': 'black holes'})
print(result.content)