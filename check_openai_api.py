import openai

# Replace with your OpenAI API key
API_KEY = ""

# Set up OpenAI API key
openai.api_key = API_KEY

def check_openai_api():
    try:
        # Sending a sample request to OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! How are you?"}
            ]
        )
        # Print the response
        print("OpenAI API is working!")
        print("Response:")
        print(response.choices[0].message["content"])
    except Exception as e:
        print("Error while connecting to OpenAI API:")
        print(e)

# Run the check
check_openai_api()
