import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.google import Gemini
from phi.embedder.google import GeminiEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType

# Load environment variables from .env file
load_dotenv()

def check_api_key():
    """Check if Google API key is properly configured"""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise EnvironmentError(
            "\nGoogle API Key not found! Please set up your API key by either:"
            "\n1. Creating a .env file with GOOGLE_API_KEY=your_api_key"
            "\n2. Or setting the environment variable: export GOOGLE_API_KEY=your_api_key"
            "\n\nTo get an API key:"
            "\n1. Go to https://makersuite.google.com/app/apikey"
            "\n2. Create or select a project"
            "\n3. Copy your API key"
        )
    return api_key

def create_ask_agent():
    # Verify API key before creating agent
    api_key = check_api_key()
    
    # Initialize the Ask agent with enhanced conversational capabilities
    Ask_agent = Agent(
        model=Gemini(api_key=api_key, id="gemini-1.5-flash"),
        description="I am a friendly learning assistant that helps you understand and review topics through interactive discussion.",
        instructions="""
        1. Start each conversation with a warm greeting and ask about the topic they want to discuss
        2. Be conversational and encouraging throughout the interaction
        3. Use these interaction strategies:
           - Ask open-ended questions to promote deeper thinking
           - Provide constructive feedback
           - Break down complex topics into simpler parts
           - Use examples and analogies when helpful
           - Encourage students to explain concepts in their own words
        4. After each topic discussion:
           - Summarize key points
           - Check for understanding
           - Suggest related topics for further exploration
        5. Maintain a supportive and patient tone
        6. If a student seems confused, try explaining the concept in a different way
        7. Celebrate their progress and understanding
        """,
        knowledge=knowledge_base,
        search_knowledge=True,
        show_tool_calls=True,
        markdown=True,
    )
    return Ask_agent

def interactive_session():
    try:
        ask_agent = create_ask_agent()
        print("👋 Hello! I'm your learning assistant. I'm here to help you understand and review topics through discussion.")
        print("What topic would you like to explore today?")
        
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for exit conditions
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nThank you for learning with me! Have a great day! 👋")
                break
            
            if user_input:
                # Get and print agent's response
                print("\nAssistant: ")
                ask_agent.print_response(user_input, stream=True)
            else:
                print("Please ask a question or type 'exit' to end our conversation.")
                
    except KeyboardInterrupt:
        print("\n\nConversation ended. Thank you for learning with me! 👋")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please try starting the conversation again.")

if __name__ == "__main__":
    try:
        knowledge_base = PDFUrlKnowledgeBase()
        knowledge_base.load()
        interactive_session()
    except EnvironmentError as e:
        print(e)
    except Exception as e:
        print(f"Error initializing the application: {str(e)}")

