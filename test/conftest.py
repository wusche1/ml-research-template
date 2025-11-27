from dotenv import load_dotenv
import os

# Load .env file before running tests
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

