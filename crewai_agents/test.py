import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import preprocess_text
from main import agentic_pipeline

if __name__ == "__main__":
    query = "I want a hotel with a quiet environment"
    response = preprocess_text(query)
    response = agentic_pipeline(response)
    print(response)