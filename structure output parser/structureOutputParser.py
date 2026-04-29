from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers.structured import StructuredOutputParser, ResponseSchema


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct", 
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact_1", description="fact2 about the topic "),
    ResponseSchema(name="fact_2", description="fact3 about the topic "),
    ResponseSchema(name="fact_3", description="fact1 about the topic "),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give me 3 facts about {topic} \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
chain = template | model | parser
result = chain.invoke({"topic": "black holes"})