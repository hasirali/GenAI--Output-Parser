from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct", #this model cant give structire output, so we will use the text generation task to get the output in text format and then we will parse it to get the structured output
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)
parser = JsonOutputParser()

template = PromptTemplate(
    template = "Give me the name, age and city of the fictional person \n {format_instructions}",
    input_variables = [],
    partial_variables = {'format_instructions': parser.get_format_instructions()}
)

# prompt = template.format()
# # print(prompt)
# # Give me the name, age and city of the fictional person 
# #  Return a JSON object.

# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

chain = template | model | parser
result = chain.invoke({})
print(result)
# 31:56


# prompt = template.format()
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# Instead of this we can use chain