# promp ->LLM -> text output(detailed) -> LLM -> 5 Line sumary
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

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

prompt1 = template1.invoke({'topic': 'black holes'})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'report': result1})
result2 = model.invoke(prompt2)

print(result1.content)
print(result2.content)