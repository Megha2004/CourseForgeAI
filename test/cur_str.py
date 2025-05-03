from phi.agent import Agent
from phi.model.google import Gemini
from phi.embedder.google import GeminiEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.lancedb import LanceDb,SearchType

knowledge_base = PDFUrlKnowledgeBase(
            path="C:/Users/ICANIO10090/Documents/PhiData/embedd/TSA/Daniel Jurafsky, James H. Martin - Speech and Language Processing-Prentice Hall (2008).pdf",
        vector_db=LanceDb(
        table_name="FDSADB",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=GeminiEmbedder(),
        ),
)

knowledge_base.load()

retrieval_agent = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    description="You help the user by answering questions or generating structured content based on the syllabus.",
    instructions=(
        "ONLY generate content for **Lesson 1** from the syllabus./n"
        "Break Lesson 1 into **Modules**, and each module into **numbered Topics**./n"
        "Each topic must have a detailed 4–5 line explanation. No headings without explanations./n"
        "- DO NOT include Lessons 2, 3, etc./n"
        "- Use markdown: ### for Modules, bullet points for Topics./n"
        "- Keep it helpful and clear for students."
    ),
    knowledge=knowledge_base,
    search_knowledge=True,
    show_tool_calls=True,
    markdown=True,
)

retrieval_agent.knowledge.load(recreate=False)
curriculum_prompt = """
You are an expert academic planner. Based on the syllabus provided below and using the textbook materials embedded in your knowledge base, generate a well-structured curriculum. Divide the content into Modules, each with relevant Lessons and Topics. Ensure the content coverage aligns with each unit in the syllabus and keeps learning progression clear and logical.Generate only for 3rd lesson(QUESTION ANSWERING AND DIALOGUE SYSTEMS).Each topic should be of atleast 15-20 lines.Atleast have 3 modules for a lesson based on the relevancy of content.Strictly dont leave any topics.

Syllabus:

UNIT I NATURAL LANGUAGE BASICS 6
Foundations of natural language processing – Language Syntax and Structure- Text Preprocessing
and Wrangling – Text tokenization – Stemming – Lemmatization – Removing stop-words – Feature
Engineering for Text representation – Bag of Words model- Bag of N-Grams model – TF-IDF model
UNIT II TEXT CLASSIFICATION 6
Vector Semantics and Embeddings -Word Embeddings - Word2Vec model – Glove model –
FastText model – Overview of Deep Learning models – RNN – Transformers – Overview of Text
summarization and Topic Models
UNIT III QUESTION ANSWERING AND DIALOGUE SYSTEMS 9
Information retrieval – IR-based question answering – knowledge-based question answering –
language models for QA – classic QA models – chatbots – Design of dialogue systems -–
evaluating dialogue systems
UNIT IV TEXT-TO-SPEECH SYNTHESIS 6
Overview. Text normalization. Letter-to-sound. Prosody, Evaluation. Signal processing -
Concatenative and parametric approaches, WaveNet and other deep learning-based TTS
systems
UNIT V AUTOMATIC SPEECH RECOGNITION 6
Speech recognition: Acoustic modelling – Feature Extraction - HMM, HMM-DNN systems

"""

retrieval_agent.print_response(curriculum_prompt, stream=True)

