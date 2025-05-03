from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
import os
from dotenv import load_dotenv
import google.generativeai as genai
import PyPDF2
import re
import json
import random
import subprocess
import tempfile
import sys
from phi.agent import Agent
from phi.model.google import Gemini
from phi.embedder.google import GeminiEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import requests

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configure Google API
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")
genai.configure(api_key=api_key)

# Define upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define notes folder
NOTES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'notes')
if not os.path.exists(NOTES_FOLDER):
    os.makedirs(NOTES_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['NOTES_FOLDER'] = NOTES_FOLDER

# Load knowledge bases once at startup
knowledge_bases = {
    "bigdata": PDFUrlKnowledgeBase(
        path=[
            "C:/Users/ICANIO10090/Documents/PhiData/embedd/bda/[Wiley CIO] Michael Minelli, Michele Chambers, Ambiga Dhiraj - Big Data, Big Analytics_ Emerging Business Intelligence and Analytic Trends for Today's Businesses (2013, Wiley) - libgen.li.pdf",
            "C:/Users/ICANIO10090/Documents/PhiData/embedd/bda/Pramod J. Sadalage, Martin Fowler - NoSQL Distilled_ A Brief Guide to the Emerging World of Polyglot Persistence (2012, Addison-Wesley Professional) - libgen.li.pdf"
        ],
        vector_db=LanceDb(
            table_name="BigDataDB",
            uri="tmp/lancedb",
            search_type=SearchType.vector,
            embedder=GeminiEmbedder()
        )
    ),
    "deeplearning": PDFUrlKnowledgeBase(
        path=[
            "C:/Users/ICANIO10090/Documents/PhiData/embedd/dl/Andrew Glassner - Deep Learning_ A Visual Approach (2021)-Textbook2.pdf",
            "C:/Users/ICANIO10090/Documents/PhiData/embedd/dl/DeepLearning Book(Text book1).pdf"
        ],
        vector_db=LanceDb(
            table_name="DeepLearningDB",
            uri="tmp/lancedb",
            search_type=SearchType.vector,
            embedder=GeminiEmbedder()
        )
    ),
    "machinelearning": PDFUrlKnowledgeBase(
        path="C:/Users/ICANIO10090/Documents/PhiData/embedd/Machine_Learning_An_Algorithmic_Perspective_(2nd_ed).pdf.crdownload",
        vector_db=LanceDb(
            table_name="MachineLearningDB",
            uri="tmp/lancedb",
            search_type=SearchType.vector,
            embedder=GeminiEmbedder(),
        ),
    ),
    "TSA": PDFUrlKnowledgeBase(
        path="C:/Users/ICANIO10090/Documents/PhiData/embedd/TSA/Daniel Jurafsky, James H. Martin - Speech and Language Processing-Prentice Hall (2008).pdf",
        vector_db=LanceDb(
            table_name="TSADB",
            uri="tmp/lancedb",
            search_type=SearchType.vector,
            embedder=GeminiEmbedder(),
    )
    ),
    "FDSA": PDFUrlKnowledgeBase(
        path=["C:/Users/ICANIO10090/Documents/PhiData/embedd/fdsa/Robert S. Witte, John S. Witte - Statistics-Wiley (2016).pdf",
              "C:/Users/ICANIO10090/Documents/PhiData/embedd/fdsa/text book.pdf"],
        vector_db=LanceDb(
            table_name="FDSADB",
            uri="tmp/lancedb",
            search_type=SearchType.vector,
            embedder=GeminiEmbedder(),
    )
    ),
    "MDA": PDFUrlKnowledgeBase(
        path=[
            "C:/Users/ICANIO10090/Documents/PhiData/embedd/mda/Applied Multivariate Statistical Analysis by Johnson and Wichern.pdf",
            "C:/Users/ICANIO10090/Documents/PhiData/embedd/mda/Jr., William C. Black, Barry J. Ba Joseph F. Hair - Multivariate Data Analysis-Pearson Education Limited (2013).pdf",
            "C:/Users/ICANIO10090/Documents/PhiData/embedd/mda/Using Multivariate Statistics (Tabachnick and Fidell).pdf",
        ],
        vector_db=LanceDb(
            table_name="MDADB",
            uri="tmp/lancedb",
            search_type=SearchType.vector,
            embedder=GeminiEmbedder(),
        ),
    )
}

# Load knowledge bases
for kb in knowledge_bases.values():
    kb.load()

# Initialize the problem statement generation agent
PS_agent = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    description="You are a helpful assistant that creates a Python problem statement based on a user-given topic.",
    instructions=(
        "Generate a well-structured Python programming problem on the given topic.\n\n"
        "The output should include:\n"
        "1. **Problem Statement** - Clearly describe the task.\n"
        "2. **Input Format** - Describe expected input(s).\n"
        "3. **Output Format** - Describe the expected output(s).\n"
        "4. **Constraints** - Mention any constraints on the input or computation.\n"
        "5. **Sample Input & Output** - Give at least one sample input-output pair.\n\n"
        "Keep the language clear and concise. Ensure it is suitable for coding practice."
    ),
    knowledge=None,
    search_knowledge=False,
    show_tool_calls=False,
    markdown=True,
)

# Initialize the code evaluation agent
EVAL_agent = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    description="You are a helpful assistant that evaluates Python code against a problem statement.",
    instructions=(
        "Evaluate the provided Python code against the problem statement.\n\n"
        "Your evaluation should include:\n"
        "1. Output of the code\n"
        "2. If the code is relevant to the problem statement and correct then give as Test result-Passed else Failed\n"
        "3. DO NOT give correct answer, give answer only after 3 incorrect attempts\n"
        "Be thorough but encouraging in your evaluation."
    ),
    knowledge=None,
    search_knowledge=False,
    show_tool_calls=False,
    markdown=True,
)

# Initialize the tutor agent
tutor_agent = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    description="You are an expert tutor who generates comprehensive, well-structured notes on academic and computer science topics.",
    instructions=(
        "Generate detailed notes on the given topic with the following structure:\n\n"
        "1. Start with a clear title using 'Page X: Topic Name' format\n"
        "2. Break down the content into logical sections with headings\n"
        "3. For each section:\n"
        "   - Use clear, concise language\n"
        "   - Include relevant examples and explanations\n"
        "   - Use bullet points for lists and key points\n"
        "   - Maintain proper spacing between paragraphs\n"
        "4. Formatting rules:\n"
        "   - Use 'Page X:' for main headings\n"
        "   - Use '##' for section headings\n"
        "   - Use bullet points (*) for lists\n"
        "   - Keep paragraphs to 3-4 sentences\n"
        "   - Add proper spacing between sections\n"
        "5. Content guidelines:\n"
        "   - Be comprehensive but concise\n"
        "   - Include practical examples\n"
        "   - Explain complex concepts clearly\n"
        "   - Use analogies when helpful\n"
        "   - Maintain academic rigor\n"
        "6. IMPORTANT: If the topic is not related to academics or computer science, respond with:\n"
        "   'I'm sorry, but I can only generate notes on academic and computer science topics. Please ask about a subject in these areas.'\n\n"
        "Generate at least 5 pages of content."
    ),
    knowledge=None,
    search_knowledge=True,
    show_tool_calls=True,
    markdown=True,
)

# Initialize the knowledge assistant agent
KNOWLEDGE_agent = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    description="You are a helpful assistant for academic subjects.",
    instructions=(
        "Answer questions about academic subjects in detail.\n\n"
        "Your responses should:\n"
        "1. Be accurate and well-researched\n"
        "2. Include relevant examples when possible\n"
        "3. Use clear and concise language\n"
        "4. Break down complex concepts into understandable parts\n"
        "5. Provide additional resources or references when appropriate"
    ),
    knowledge=knowledge_bases,
    search_knowledge=True,
    show_tool_calls=True,
    markdown=True,
)

# Initialize the Learn with Me agent
learn_agent = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    description="I am a direct and efficient learning assistant that provides clear, concise answers about academic and computer science topics.",
    instructions="""
    1. Provide direct answers to user questions without unnecessary greetings or follow-up questions
    2. Keep responses focused and to the point
    3. Structure information clearly:
       - Start with the main answer
       - Use bullet points for key points
       - Include relevant examples when helpful
    4. Format responses for clarity:
       - Use **bold** for important terms
       - Use *italics* for emphasis
       - Use bullet points for lists
    5. If a concept is complex:
       - Break it down into simpler parts
       - Use analogies when helpful
       - Provide practical examples
    6. Maintain a professional and helpful tone
    7. Avoid asking questions unless absolutely necessary
    8. Keep the conversation flowing naturally
    9. IMPORTANT: If the question is not related to academics or computer science, respond with:
       "I'm sorry, but I can only answer questions about academic subjects and computer science topics. Please ask a question in these areas."
    """,
    knowledge=knowledge_bases,
    search_knowledge=True,
    show_tool_calls=True,
    markdown=True,
)

# Store the last generated problem statement and attempts
last_problem = None
attempt_count = 0

# Read PDF
def read_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Create flashcards from PDF
def create_flashcards(pdf_text, num_flashcards=10):
    # Divide text into chunks
    chunks = [pdf_text[i:i+1000] for i in range(0, len(pdf_text), 1000)]
    
    # Use Gemini to generate flashcards
    model = genai.GenerativeModel('gemini-1.5-flash')
    flashcards = []
    
    for chunk in chunks[:2]:  # Process only first 2 chunks to avoid rate limits
        prompt = f"""
        Create {num_flashcards} flashcards from this text that focus ONLY on core concepts and key ideas.
        DO NOT include metadata, publication details, or peripheral information.
        Each flashcard should have a question about a fundamental concept and a clear, concise answer.
        Format each flashcard as a JSON object with 'question' and 'answer' fields.
        Return an array of these objects.
        
        Text: {chunk}
        """
        
        try:
            response = model.generate_content(prompt)
            response_text = response.text
            
            # Try to extract JSON from the response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                try:
                    cards = json.loads(json_match.group(0))
                    if isinstance(cards, list):
                        flashcards.extend(cards)
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            print(f"Error generating flashcards: {str(e)}")
    
    # If no flashcards were generated, create some simple ones
    if not flashcards:
        sentences = re.split(r'(?<=[.!?])\s+', pdf_text)
        for i in range(0, min(len(sentences), int(num_flashcards)), 2):
            if i + 1 < len(sentences):
                flashcards.append({
                    "question": sentences[i],
                    "answer": sentences[i + 1]
                })
    
    return flashcards[:int(num_flashcards)]

# Get chatbot response
def get_chatbot_response(user_message):
    lower_message = user_message.lower()
    
    # Knowledge Assistant responses
    if any(word in lower_message for word in ['knowledge', 'subject', 'academic', 'study', 'learn']):
        return """The Knowledge Assistant helps you learn academic subjects. Here's what you can do:
1. Select from various subjects like Machine Learning, Deep Learning, Big Data Analytics, etc.
2. Ask specific questions about any topic in these subjects
3. Get detailed, well-researched answers with examples
4. Access structured learning materials and resources"""
    
    # Coding Assistant responses
    elif any(word in lower_message for word in ['code', 'programming', 'python', 'problem', 'solution']):
        return """The Coding Assistant helps you with programming practice:
1. Generate coding problems on any topic
2. Submit your solutions for evaluation
3. Get detailed feedback on your code
4. Access solutions after three attempts
5. Learn through practical programming exercises"""
    
    # Notes Assistant responses
    elif any(word in lower_message for word in ['notes', 'study material', 'content', 'lesson']):
        return """The Notes Assistant provides structured learning materials:
1. Access comprehensive notes for various subjects
2. Browse through organized lessons and modules
3. Study key concepts and topics
4. Get detailed explanations with examples
5. Track your learning progress"""
    
    # Flashcard Generator responses
    elif any(word in lower_message for word in ['flashcard', 'flash cards', 'memorize']):
        return """The Flashcard Generator helps you create study aids:
1. Upload PDF documents to generate flashcards
2. Customize the number of flashcards needed
3. Create effective study materials
4. Enhance your memorization and learning
5. Access your flashcards anytime"""
    
    # Matching Cards responses
    elif any(word in lower_message for word in ['matching cards', 'game', 'play', 'match']):
        return """Matching Cards is a fun memory game for learning:
1. Upload any PDF to create question-answer pairs
2. Play a memory game where you flip cards to find matching pairs
3. Test your memory and knowledge as you play
4. Try to complete the game with the fewest moves
5. Challenge yourself to beat your best time"""
    
    # Tutor Me responses
    elif any(word in lower_message for word in ['tutor', 'teach', 'explain', 'generate notes']):
        return """The Tutor Me feature helps you learn any topic:
1. Generate detailed notes on any subject
2. Get comprehensive explanations
3. Download notes as PDF files
4. Learn at your own pace
5. Access well-structured learning materials"""
    
    # General features overview
    elif any(word in lower_message for word in ['features', 'what can', 'help', 'assist']):
        return """Our application offers six powerful learning tools:
1. Knowledge Assistant - Get answers to academic questions
2. Coding Assistant - Practice programming problems
3. Notes Assistant - Access structured study materials
4. Flashcard Generator - Create study aids from PDFs
5. Matching Cards - Play a memory game with question-answer pairs
6. Tutor Me - Generate detailed notes on any topic
Each feature is designed to help you learn in different ways. What would you like to know more about?"""
    
    # Greeting responses
    elif any(word in lower_message for word in ['hello', 'hi', 'hey']):
        return """Hello! I'm your AI learning assistant. I can help you with:
1. Academic questions through the Knowledge Assistant
2. Programming practice with the Coding Assistant
3. Study materials via the Notes Assistant
4. Flashcards creation with the Flashcard Generator
5. Memory games with Matching Cards
6. Detailed notes generation with Tutor Me
What would you like to learn about today?"""
    
    # Default response
    else:
        return """I'm here to help you learn! You can ask about:
1. Knowledge Assistant - For academic questions
2. Coding Assistant - For programming practice
3. Notes Assistant - For study materials
4. Flashcard Generator - For creating study aids
5. Matching Cards - For playing memory games
6. Tutor Me - For generating detailed notes
Which feature would you like to know more about?"""

def get_subject_notes(subject_file):
    """Read and parse subject notes from a file."""
    try:
        with open(subject_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content into lessons
        lessons = content.split('LESSON')
        parsed_notes = []
        
        for lesson in lessons[1:]:  # Skip first empty split
            lesson_content = lesson.strip()
            # Split into modules
            modules = lesson_content.split('Module')
            lesson_num = modules[0].strip().split()[0]  # Get lesson number
            
            module_list = []
            for module in modules[1:]:  # Skip the lesson number part
                module_content = module.strip()
                module_lines = module_content.split('\n')
                module_title = module_lines[0].strip()
                
                # Extract topics (bullet points)
                topics = []
                current_topic = []
                
                for line in module_lines[1:]:
                    line = line.strip()
                    if line.startswith('•'):
                        # If we were building a previous topic, save it
                        if current_topic:
                            topics.append(' '.join(current_topic))
                            current_topic = []
                        # Start new topic
                        current_topic.append(line[1:].strip())  # Remove bullet point
                    elif line and current_topic:  # Continue previous topic
                        current_topic.append(line)
                
                # Add the last topic if exists
                if current_topic:
                    topics.append(' '.join(current_topic))
                
                module_list.append({
                    'title': module_title,
                    'topics': topics
                })
            
            parsed_notes.append({
                'lesson_num': lesson_num,
                'modules': module_list
            })
        
        return parsed_notes
    except Exception as e:
        print(f"Error parsing notes: {str(e)}")
        return None

@app.route('/')
def home():
    features = {
        'knowledge': {
            'title': 'Knowledge Assistant',
            'description': 'Ask questions about academic subjects',
            'icon': 'fas fa-graduation-cap',
            'route': 'knowledge'
        },
        'coding': {
            'title': 'Coding Assistant',
            'description': 'Practice coding with our interactive problem-solving environment',
            'icon': 'fas fa-code',
            'route': 'coding_instructions'
        },
        'flashcards': {
            'title': 'Flashcard Generator',
            'description': 'Create flashcards from your PDF documents',
            'icon': 'fas fa-clone',
            'route': 'flashcards'
        },
        'notes': {
            'title': 'Notes Assistant',
            'description': 'Access and study subject notes',
            'icon': 'fas fa-book-open',
            'route': 'notes'
        },
        'chatbot': {
            'title': 'Chatbot',
            'description': 'Chat with our AI assistant',
            'icon': 'fas fa-comments',
            'route': 'chatbot'
        },
        'matching_cards': {
            'title': 'Matching Cards',
            'description': 'Test your knowledge with a fun matching game',
            'icon': 'fas fa-puzzle-piece',
            'route': 'matching_cards'
        },
        'tutor_me': {
            'title': 'Tutor Me',
            'description': 'Generate detailed notes on any topic',
            'icon': 'fas fa-chalkboard-teacher',
            'route': 'tutor_me'
        },
        'learn_with_me': {
            'title': 'Learnora',
            'description': 'Interactive learning through conversation',
            'icon': 'fas fa-book-reader',
            'route': 'learn_with_me'
        }
    }
    return render_template('home.html', features=features)

@app.route('/flashcards', methods=['GET', 'POST'])
def flashcards():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('flashcards.html', error="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('flashcards.html', error="No file selected")
        
        if not file.filename.endswith('.pdf'):
            return render_template('flashcards.html', error="Only PDF files are supported")
        
        # Get number of flashcards from request
        num_flashcards = request.form.get('num_cards', '10')
        
        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        try:
            # Read the PDF
            pdf_text = read_pdf(file_path)
            
            # Create flashcards
            flashcards = create_flashcards(pdf_text, num_flashcards)
            
            # Clean up the uploaded file
            os.remove(file_path)
            
            return render_template('flashcards.html', flashcards=flashcards)
        except Exception as e:
            # Clean up the uploaded file in case of error
            if os.path.exists(file_path):
                os.remove(file_path)
            return render_template('flashcards.html', error=str(e))
    
    return render_template('flashcards.html')

@app.route('/notes', methods=['GET'])
def notes():
    # Get list of available subjects
    subject_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'subject context')
    subjects = {
        'MDA': {
            'name': 'Multivariate Data Analysis',
            'file': 'mda.txt',
            'description': 'Learn about multivariate analysis techniques and their applications.',
            'icon': 'fas fa-chart-bar'
        },
        'ML': {
            'name': 'Machine Learning',
            'file': 'ml.txt',
            'description': 'Explore machine learning algorithms, concepts, and implementations.',
            'icon': 'fas fa-brain'
        },
        'DL': {
            'name': 'Deep Learning',
            'file': 'dl.txt',
            'description': 'Study deep neural networks, architectures, and advanced concepts.',
            'icon': 'fas fa-network-wired'
        },
        'BDA': {
            'name': 'Big Data Analytics',
            'file': 'bda.txt',
            'description': 'Understand big data processing, analytics, and frameworks.',
            'icon': 'fas fa-database'
        },
        'TSA': {
            'name': 'Text and Speech Analysis',
            'file': 'tsa.txt',
            'description': 'Explore natural language processing and speech technologies.',
            'icon': 'fas fa-comments'
        },
        'FDSA': {
            'name': 'Fundamentals of Data Science and Analytics',
            'file': 'fdsa.txt',
            'description': 'Master the core concepts of data science and statistical analysis.',
            'icon': 'fas fa-chart-line'
        }
    }
    
    # Get selected subject from query parameter
    selected_subject = request.args.get('subject')
    notes_content = None
    
    if selected_subject and selected_subject in subjects:
        file_path = os.path.join(subject_dir, subjects[selected_subject]['file'])
        if os.path.exists(file_path):
            notes_content = get_subject_notes(file_path)
    
    return render_template('notes.html', 
                         subjects=subjects,
                         selected_subject=selected_subject,
                         notes=notes_content)

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_message = request.form.get('user_message', '')
        bot_response = get_chatbot_response(user_message)
        return jsonify({'response': bot_response})
    return render_template('chatbot.html')

@app.route('/knowledge')
def knowledge():
    return render_template('index.html')

@app.route('/coding_instructions')
def coding_instructions():
    return render_template('coding_instructions.html')

@app.route('/coding')
def coding():
    return render_template('coding.html')

@app.route('/ps_generator')
def ps_generator():
    return render_template('ps_generator.html')

@app.route('/generate_problem', methods=['POST'])
def generate_problem():
    try:
        data = request.json
        topic = data.get('topic')

        if not topic:
            return jsonify({'error': 'Please provide a topic'}), 400

        # Generate the problem statement using the PhiData agent
        response = PS_agent.run(f"Topic: {topic}", stream=False)
        
        # Extract the content from the response
        if hasattr(response, 'content'):
            problem = response.content
        else:
            problem = str(response)
        
        # Store the problem for evaluation and reset attempt count
        global last_problem, attempt_count
        last_problem = problem
        attempt_count = 0
            
        return jsonify({'problem': problem})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate_code', methods=['POST'])
def evaluate_code():
    try:
        data = request.json
        code = data.get('code')
        language = data.get('language')
        output = data.get('output')
        attempt = data.get('attempt', 1)

        if not code or not language:
            return jsonify({'error': 'Code and language are required'}), 400

        if not last_problem:
            return jsonify({'error': 'Please generate a problem statement first'}), 400

        # Create a temporary file with appropriate extension
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
            temp_file.write(code.encode('utf-8'))
            temp_file_path = temp_file.name

        try:
            # Run the code and capture output
            result = subprocess.run(
                ['python', temp_file_path],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Clean up temporary files
            os.unlink(temp_file_path)

            if result.returncode != 0:
                return jsonify({
                    'error': f'Runtime error:\n{result.stderr}'
                }), 400

            # Increment attempt count
            global attempt_count
            attempt_count += 1

            # Evaluate the code using the PhiData agent
            evaluation_prompt = f"""
            Problem Statement:
            {last_problem}

            Submitted Code Output:
            {result.stdout}

            Please evaluate this code output against the problem statement.
            Consider:
            1. Does the output match the expected format?
            2. Does it handle the sample input correctly?
            3. Are there any logical errors?
            4. Is the solution efficient?
            5. Does it follow good coding practices?

            IMPORTANT INSTRUCTIONS:
            1. DO NOT provide any code or solution in your response
            2. DO NOT show the correct implementation
            3. DO NOT include any code snippets
            4. Only explain what's wrong with the current solution
            5. Provide general guidance on how to improve
            6. Keep the feedback focused on the current attempt's issues

            Your response should only contain:
            - What the current code does wrong
            - What needs to be fixed
            - General suggestions for improvement
            - No code examples or solutions
            """
            
            response = EVAL_agent.run(evaluation_prompt, stream=False)
            
            # Extract the content from the response
            if hasattr(response, 'content'):
                evaluation = response.content
            else:
                evaluation = str(response)
                
            return jsonify({
                'evaluation': evaluation,
                'attempts': attempt_count,
                'show_solution': attempt_count >= 3
            })

        except subprocess.TimeoutExpired:
            return jsonify({'error': 'Code execution timed out'}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_solution', methods=['GET'])
def get_solution():
    try:
        if not last_problem:
            return jsonify({'error': 'No problem statement available'}), 400

        # Generate solution using the PhiData agent
        solution_prompt = f"""
        Problem Statement:
        {last_problem}

        Please provide a complete solution to this problem in Python.
        Include:
        1. A clear explanation of the approach
        2. The complete Python code
        3. Comments explaining key parts of the code
        4. Example usage with the sample input/output from the problem
        """
        
        response = PS_agent.run(solution_prompt, stream=False)
        
        # Extract the content from the response
        if hasattr(response, 'content'):
            solution = response.content
        else:
            solution = str(response)
            
        return jsonify({'solution': solution})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        question = data.get('question')
        subject = data.get('subject')

        if not question or not subject or subject not in knowledge_bases:
            return jsonify({"answer": "Invalid input. Please provide both a question and select a valid subject."}), 400

        # Replace "this subject" with the actual subject name in the question
        question = question.replace("this subject", subject).replace("this Subject", subject)

        # Create a temporary agent with only the selected subject's knowledge base
        subject_agent = Agent(
            model=Gemini(id="gemini-1.5-flash"),
            description=f"You are a helpful assistant for {subject}. You ONLY answer questions about {subject}.",
            instructions=(
                f"You are a {subject} expert. Your knowledge is strictly limited to {subject}.\n\n"
                "Your responses must:\n"
                "1. ONLY use information from the {subject} knowledge base\n"
                "2. If asked about any other subject, respond with: 'I'm sorry, but I can only answer questions about {subject}. Please ask a question related to this subject.'\n"
                "3. Be accurate and well-researched\n"
                "4. Include relevant examples when possible\n"
                "5. Use clear and concise language\n"
                "6. Break down complex concepts into understandable parts\n"
                "7. If you don't know the answer, say so\n"
                "8. Never make up information or use knowledge from other subjects\n"
                "9. When asked about {subject} (like 'What is {subject} about?'), provide:\n"
                "   - A clear definition of {subject}\n"
                "   - Main topics and concepts\n"
                "   - Key applications or uses\n"
                "   - Important principles or fundamentals\n"
                "   - A brief overview of the field"
            ),
            knowledge={subject: knowledge_bases[subject]},
            search_knowledge=True,
            show_tool_calls=True,
            markdown=True,
        )

        # Use the subject-specific agent
        response = subject_agent.run(question, stream=False)
        
        # Extract the content from the response
        if hasattr(response, 'content'):
            answer = response.content
        else:
            answer = str(response)
            
        # Clean up the response
        answer = answer.replace("content='", "").replace("' content_type='str'", "")
        
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"An error occurred: {str(e)}"}), 500

@app.route('/matching_cards', methods=['GET', 'POST'])
def matching_cards():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('matching_cards.html', error="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('matching_cards.html', error="No file selected")
        
        if not file.filename.endswith('.pdf'):
            return render_template('matching_cards.html', error="Only PDF files are supported")
        
        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        try:
            # Read the PDF
            pdf_text = read_pdf(file_path)
            
            # Generate QA pairs
            qa_pairs = generate_qa_pairs(pdf_text)
            
            # Clean up the uploaded file
            os.remove(file_path)
            
            return render_template('matching_cards.html', qa_pairs=qa_pairs)
        except Exception as e:
            # Clean up the uploaded file in case of error
            if os.path.exists(file_path):
                os.remove(file_path)
            return render_template('matching_cards.html', error=str(e))
    
    return render_template('matching_cards.html')

def generate_qa_pairs(pdf_text, num_pairs=5):
    # Divide text into chunks
    chunks = [pdf_text[i:i+1000] for i in range(0, len(pdf_text), 1000)]
    
    # Use Gemini to generate QA pairs
    model = genai.GenerativeModel('gemini-1.5-flash')
    qa_pairs = []
    
    for chunk in chunks[:2]:  # Process only first 2 chunks to avoid rate limits
        prompt = f"""
        Create {num_pairs} question-answer pairs from this text that focus ONLY on core subject concepts and key ideas.
        DO NOT include metadata, publication details, or peripheral information.
        Each pair should have a clear question about a fundamental concept and a concise answer.
        Format each pair as a JSON object with 'question' and 'answer' fields.
        Return an array of these objects.
        
        Text: {chunk}
        """
        
        try:
            response = model.generate_content(prompt)
            response_text = response.text
            
            # Try to extract JSON from the response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                try:
                    pairs = json.loads(json_match.group(0))
                    if isinstance(pairs, list):
                        qa_pairs.extend(pairs)
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            print(f"Error generating QA pairs: {str(e)}")
    
    # If no pairs were generated, create some simple ones
    if not qa_pairs:
        sentences = re.split(r'(?<=[.!?])\s+', pdf_text)
        for i in range(0, min(len(sentences), int(num_pairs)), 2):
            if i + 1 < len(sentences):
                qa_pairs.append({
                    "question": sentences[i],
                    "answer": sentences[i + 1]
                })
    
    return qa_pairs[:int(num_pairs)]

def save_notes(topic, notes):
    """Save generated notes to a PDF file."""
    try:
        # Create a filename from the topic (replace spaces and special characters)
        filename = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_').lower()
        filepath = os.path.join(app.config['NOTES_FOLDER'], f"{filename}.pdf")
        
        # Create PDF document with margins
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            leftMargin=50,
            rightMargin=50,
            topMargin=50,
            bottomMargin=50
        )
        styles = getSampleStyleSheet()
        
        # Custom style for main headings (Page X:)
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.purple,
            spaceAfter=30,
            alignment=1,  # Center alignment
            fontName='Helvetica-Bold'
        )
        
        # Custom style for subheadings (##)
        subheading_style = ParagraphStyle(
            'CustomSubheading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.purple,
            spaceAfter=20,
            fontName='Helvetica-Bold'
        )
        
        # Custom style for paragraphs
        para_style = ParagraphStyle(
            'CustomPara',
            parent=styles['Normal'],
            fontSize=12,
            leading=16,
            spaceAfter=15,
            fontName='Helvetica',
            alignment=0  # Left alignment
        )
        
        # Custom style for lists
        list_style = ParagraphStyle(
            'CustomList',
            parent=styles['Normal'],
            fontSize=12,
            leading=16,
            spaceAfter=8,
            leftIndent=20,
            fontName='Helvetica'
        )
        
        # Prepare content
        content = []
        
        # Split notes into sections
        sections = notes.split('\n\n')
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            # Handle main headings (Page X:)
            if section.startswith('Page'):
                content.append(Paragraph(section, heading_style))
                content.append(Spacer(1, 30))
                
            # Handle subheadings (##)
            elif section.startswith('##'):
                content.append(Paragraph(section[2:].strip(), subheading_style))
                content.append(Spacer(1, 20))
                
            # Handle lists
            elif section.startswith('*'):
                for line in section.split('\n'):
                    if line.strip().startswith('*'):
                        # Create bullet point with proper indentation
                        bullet_text = f"• {line[1:].strip()}"
                        content.append(Paragraph(bullet_text, list_style))
                content.append(Spacer(1, 15))
                
            # Handle regular paragraphs
            else:
                # Split long paragraphs into smaller ones if they contain multiple sentences
                sentences = section.split('. ')
                current_paragraph = []
                
                for sentence in sentences:
                    current_paragraph.append(sentence)
                    if len(current_paragraph) >= 3:  # Group 3 sentences per paragraph
                        content.append(Paragraph('. '.join(current_paragraph) + '.', para_style))
                        current_paragraph = []
                
                # Add any remaining sentences
                if current_paragraph:
                    content.append(Paragraph('. '.join(current_paragraph) + '.', para_style))
                content.append(Spacer(1, 15))
        
        # Build PDF
        doc.build(content)
        return filepath
    except Exception as e:
        print(f"Error saving notes: {str(e)}")
        return None

@app.route('/tutor_me', methods=['GET', 'POST'])
def tutor_me():
    if request.method == 'POST':
        topic = request.form.get('topic')
        if not topic:
            return render_template('tutor_me.html', error="Please provide a topic")
        
        try:
            # Generate notes using the tutor agent
            response = tutor_agent.run(topic, stream=False)
            
            # Extract the content from the response
            if hasattr(response, 'content'):
                notes = response.content
            else:
                notes = str(response)
            
            # Check if the response is about non-academic topic
            if notes.startswith("I'm sorry"):
                return render_template('tutor_me.html', message=notes)
            
            # Format the response with proper paragraphs and spacing
            formatted_notes = format_notes(notes)
            
            # Save the notes only for academic topics
            saved_file = save_notes(topic, formatted_notes)
            if saved_file:
                filename = os.path.basename(saved_file)
                message = f"Notes saved successfully as {filename}"
                return render_template('tutor_me.html', notes=formatted_notes, message=message, filename=filename)
            else:
                message = "Notes generated but could not be saved"
                return render_template('tutor_me.html', notes=formatted_notes, message=message)
        except Exception as e:
            return render_template('tutor_me.html', error=str(e))
    
    return render_template('tutor_me.html')

@app.route('/download_notes/<filename>')
def download_notes(filename):
    try:
        filepath = os.path.join(app.config['NOTES_FOLDER'], filename)
        if os.path.exists(filepath):
            return send_file(
                filepath,
                as_attachment=True,
                download_name=filename
            )
        return "File not found", 404
    except Exception as e:
        return str(e), 500

def format_notes(notes):
    """Format the notes with proper paragraphs and spacing."""
    # Split the content into sections
    sections = notes.split('\n\n')
    formatted_sections = []
    
    for section in sections:
        # Skip empty sections
        if not section.strip():
            continue
            
        # Handle main headings (Page X:)
        if section.strip().startswith('Page'):
            heading_text = section.strip()
            formatted_sections.append(f"\n\n## {heading_text}\n\n")
            continue
            
        # Handle subtopics (text ending with colon and not containing "Page")
        if section.strip().endswith(':') and 'Page' not in section:
            # Make the subtopic text bold
            subtopic_text = section.strip()
            formatted_sections.append(f"\n\n**{subtopic_text}**\n\n")
            continue
            
        # Handle lists
        if section.strip().startswith('*'):
            # Add spacing before list
            formatted_sections.append('\n')
            # Format each list item
            list_items = section.strip().split('\n')
            for item in list_items:
                if item.strip().startswith('*'):
                    formatted_sections.append(item.strip() + '\n')
            # Add spacing after list
            formatted_sections.append('\n')
            continue
            
        # Handle regular paragraphs
        # Split long paragraphs into smaller ones if they contain multiple sentences
        sentences = section.strip().split('. ')
        current_paragraph = []
        
        for sentence in sentences:
            current_paragraph.append(sentence)
            if len(current_paragraph) >= 3:  # Group 3 sentences per paragraph
                formatted_sections.append(' '.join(current_paragraph) + '.\n\n')
                current_paragraph = []
        
        # Add any remaining sentences
        if current_paragraph:
            formatted_sections.append(' '.join(current_paragraph) + '.\n\n')
    
    # Join sections with proper spacing
    formatted_content = ''.join(formatted_sections)
    
    # Clean up multiple newlines
    formatted_content = re.sub(r'\n{3,}', '\n\n', formatted_content)
    
    return formatted_content.strip()

@app.route('/learn_with_me', methods=['GET', 'POST'])
def learn_with_me():
    if request.method == 'POST':
        user_input = request.form.get('user_input', '').strip()
        if user_input.lower() in ['exit', 'quit', 'bye']:
            return jsonify({
                'response': "Thank you for learning with me! Have a great day! 👋",
                'end_session': True
            })
        
        try:
            # Use the learn agent without any subject-specific knowledge
            response = learn_agent.run(user_input, stream=False)
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
                
            return jsonify({
                'response': response_text,
                'end_session': False
            })
        except Exception as e:
            return jsonify({
                'response': f"An error occurred: {str(e)}",
                'end_session': False
            })
    
    return render_template('learn_with_me.html')

@app.route('/run_code', methods=['POST'])
def run_code():
    try:
        data = request.json
        code = data.get('code')
        language = data.get('language')

        if not code or not language:
            return jsonify({'error': 'Code and language are required'}), 400

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
            temp_file.write(code.encode('utf-8'))
            temp_file_path = temp_file.name

        try:
            # Run the Python code
            result = subprocess.run(
                ['python', temp_file_path],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Clean up temporary file
            os.unlink(temp_file_path)

            if result.returncode != 0:
                return jsonify({
                    'error': f'Runtime error:\n{result.stderr}'
                }), 400

            return jsonify({
                'output': result.stdout,
                'statusCode': result.returncode
            })

        except subprocess.TimeoutExpired:
            return jsonify({'error': 'Code execution timed out'}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

