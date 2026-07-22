from flask import Flask, render_template, request
from dotenv import load_dotenv
from openai import OpenAI
import os

# Your existing file name
from utils.retreiver import retrieve

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    answer = ""
    question = ""

    if request.method == "POST":

        question = request.form["question"].strip()

        # Greeting responses
        greetings = [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
            "who are you"
        ]

        thanks = [
            "thanks",
            "thank you",
            "ok",
            "okay"
        ]

        if question.lower() in greetings:

            answer = """
👋 Hello!

Welcome to Knock for Law.

I am your AI Legal Consultant trained on Indian legal documents.

You can ask questions like:

• My property was illegally occupied.
• Someone stole my bike.
• Police refused to register my FIR.
• I received a defective product.
• What are my Fundamental Rights?

How can I help you today?
"""

            return render_template(
                "index.html",
                answer=answer,
                question=question
            )

        if question.lower() in thanks:

            answer = """
😊 You're welcome!

If you have any questions related to Indian laws, feel free to ask.

Have a great day!
"""

            return render_template(
                "index.html",
                answer=answer,
                question=question
            )

        try:

            # Retrieve legal context
            retrieved_chunks = retrieve(question)
            context = "\n\n".join(retrieved_chunks)

            prompt = f"""
You are an AI Legal Consultant specializing in Indian Laws.

Rules:
- Answer ONLY using the provided legal context.
- Explain the law in simple English.
- Mention the relevant Article, Section or Act whenever available.
- Suggest legal remedies only if supported by the legal context.
- Never invent legal information.
- If the answer is unavailable in the legal context, reply:
"Sorry, I couldn't find this information in the provided legal documents."

LEGAL CONTEXT:
{context}

USER QUESTION:
{question}
"""

            response = client.chat.completions.create(
                model="openrouter/auto",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert AI Legal Consultant. Answer only from the provided legal context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            llm_answer = response.choices[0].message.content

            # Detect relevant sources
            sources = []

            if "Constitution" in context:
                sources.append("✔ Constitution of India")

            if "Bharatiya Nyaya" in context or "BNS" in context:
                sources.append("✔ Bharatiya Nyaya Sanhita (BNS)")

            if "BNSS" in context or "Nagarik Suraksha" in context:
                sources.append("✔ Bharatiya Nagarik Suraksha Sanhita (BNSS)")

            if "Bharatiya Sakshya" in context or "BSA" in context:
                sources.append("✔ Bharatiya Sakshya Adhiniyam (BSA)")

            if "Consumer" in context:
                sources.append("✔ Consumer Protection Act, 2019")

            if not sources:
                sources.append("✔ Legal Documents")

            source_text = "\n".join(sources)

            answer = f"""
⚖️ LEGAL ANALYSIS

{llm_answer}

────────────────────────────────

📚 RELEVANT SOURCES

{source_text}

────────────────────────────────

⚠️ DISCLAIMER

This response is generated using the legal documents available in the system.
It is intended for educational and informational purposes only.
Please consult a qualified legal professional before taking any legal action.
"""

        except Exception as e:

            answer = f"❌ Error: {str(e)}"

    return render_template(
        "index.html",
        answer=answer,
        question=question
    )


if __name__ == "__main__":
    app.run(debug=True)