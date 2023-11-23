from flask import Flask, request, jsonify
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI

app = Flask(__name__)

# Initialize QdrantClient
qdrant = QdrantClient(
    url="https://a8ddd992-7d4b-450a-a29e-0d2a3502f278.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="BbJFJznq0pXmuWGLlIlVfo8F0jmSb3gRRCi5-2uHQIwcUyrYqM54BQ",
)

# Initialize SentenceTransformer encoder
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize OpenAI client
openai_client = OpenAI(
    api_key='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',
)

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data['query']

        # Use SentenceTransformer to encode the query
        query_vector = encoder.encode([query])

        # Use Qdrant to search
        search_hits = qdrant.search(
            collection_name='items_log',
            query_vector=query_vector.tolist(),
            limit=6
        )

        context = str(search_hits)
        openai_prompt = f"""
        You are an AI assistant for an online store. A user is inquiring about products based on the given query. Your task is to provide detailed and helpful information about the products in the context. If you don't have information about a specific attribute, suggest another product. Do not add any information in your answer other than found products.

        **Context:**
        {context}

        **User Query:**
        {query}

        **Helpful Answer:**
        """
        openai_prompt = openai_prompt.format(context, query)

        # Use OpenAI to generate a response
        chat_completion = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": openai_prompt,
                }
            ],
            model="gpt-3.5-turbo",
            temperature=0.6,
        )

        result = chat_completion.choices[0].message.content
        return jsonify({'message': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
