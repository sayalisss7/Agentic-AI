import os
from dotenv import load_dotenv
from transformers import pipeline

def main():
    print("=== Autonomous Task Planner ===")

    # Load environment variables from .env file
    load_dotenv()
    hf_token = os.getenv("HF_API_TOKEN")

    if not hf_token:
        print("API token not found. Please check your .env file.")
        return

# Initialize Hugging Face pipeline
    # generator = pipeline(
    #     'text-generation',
    #     model='distilgpt2',
    #     use_auth_token=hf_token
    # )    
    generator = pipeline('text-generation', model='distilgpt2')

    while True:
        task = input("\nEnter your task (or type 'exit' to quit): ")
        if task.lower() == 'exit':
            print("Goodbye!")
            break

        prompt = f"""You are a helpful assistant. Break the following task into actionable steps in order:

Task: {task}

Steps:"""

        result = generator(
    prompt,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    truncation=True,
    repetition_penalty=1.2
)
        steps = result[0]['generated_text'].split("Steps:")[-1].strip()

        print("\nHere is your task plan:\n")
        print(steps)

if __name__ == "__main__":
    main()
