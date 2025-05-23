from crewai import Agent
import openai

openai.api_key = "api_key"
class RetrieverAgent(Agent):
    def __init__(self, model, client, collection_name):
        super().__init__(
            role="Retriever Agent",
            goal="Search the knowledge base for relevant travel information",
            backstory="This agent is responsible for retrieving relevant hotel or travel reviews from the vector database."
        )
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_collection_name", collection_name)

    def run(self, query: str):
        query_vec = self._model.encode(query)
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vec,
            limit=5
        )
        texts = [res.payload['text'] for res in results.points if 'text' in res.payload]
        return texts

class SummarizerAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Summarizer Agent",
            goal="Summarize retrieved travel data and remove redundant information",
            backstory="This agent combines and summarizes the retrieved content to prepare it for the final response."
        )

    def run(self, texts: list[str]):
        combined_text = "\n\n".join(texts)
        prompt = f"""
        You are a travel assistant. Summarize the following hotel reviews into a concise summary (max 200 words), removing redundant information and focusing on key points like location, amenities, and service quality:
        {combined_text[:2000]}
        """
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful travel assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content

class ComposerAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Composer Agent",
            goal="Generate user-facing responses using LLMs",
            backstory="This agent crafts the final travel recommendation using GPT based on the summarized input."
        )

    def run(self, summary: str):
        prompt = f"""
        You are a helpful travel assistant.

        Here is a summary of reviews:
        {summary}

        Please write a hotel recommendation or short itinerary in a friendly, concise tone.
        """
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful travel assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content