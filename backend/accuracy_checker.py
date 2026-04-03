import warnings
import os
import time
import json
import re
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from config import GROQ_API_KEY, EMBEDDING_MODEL

GROQ_MODELS = ["llama-3.1-8b-instant", "gemma2-9b-it"]


def print_header(title):
    print("\n" + "=" * 60)
    print("   " + title)
    print("=" * 60)


def get_test_questions():
    return [
        {
            "question": "Which role suits me if I know Python and SQL?",
            "expected_roles": ["Python_Developer", "Software_Developer", "Database_Administrator"],
            "expected_keywords": ["python", "developer", "sql", "database"],
        },
        {
            "question": "I know JavaScript, React, HTML, CSS - what job?",
            "expected_roles": ["Front_End_Developer", "Web_Developer"],
            "expected_keywords": ["frontend", "front end", "web", "react", "developer"],
        },
        {
            "question": "Java, Spring Boot, Microservices - suggest role",
            "expected_roles": ["Java_Developer", "Software_Developer"],
            "expected_keywords": ["java", "developer", "spring", "backend"],
        },
        {
            "question": "I have experience in machine learning and Python",
            "expected_roles": ["Python_Developer", "Software_Developer"],
            "expected_keywords": ["python", "developer", "machine learning", "data"],
        },
        {
            "question": "Network security and firewall management skills",
            "expected_roles": ["Security_Analyst", "Network_Administrator"],
            "expected_keywords": ["security", "network", "administrator", "analyst"],
        },
        {
            "question": "Django and REST APIs, what job should I apply for?",
            "expected_roles": ["Python_Developer", "Software_Developer", "Web_Developer"],
            "expected_keywords": ["python", "developer", "django", "backend", "web"],
        },
        {
            "question": "I manage databases, Oracle and PostgreSQL expert",
            "expected_roles": ["Database_Administrator"],
            "expected_keywords": ["database", "administrator", "oracle", "postgresql"],
        },
        {
            "question": "Linux, Docker, Kubernetes, CI/CD - what role?",
            "expected_roles": ["Systems_Administrator", "Software_Developer"],
            "expected_keywords": ["systems", "administrator", "devops", "developer"],
        },
        {
            "question": "Project management with Agile and Scrum experience",
            "expected_roles": ["Project_manager"],
            "expected_keywords": ["project", "manager", "agile", "scrum"],
        },
        {
            "question": "Node.js, Express, MongoDB, full stack development",
            "expected_roles": ["Web_Developer", "Software_Developer"],
            "expected_keywords": ["web", "developer", "full stack", "node"],
        },
    ]


def load_chatbot():
    """Load chatbot with proper error handling."""
    print("Loading chatbot...")

    # Check FAISS index exists
    faiss_path = os.path.join(os.path.dirname(__file__), "faiss_index", "index.faiss")
    if not os.path.exists(faiss_path):
        # Try to find it
        for path in ["faiss_index", "../faiss_index", "data/faiss_index"]:
            check = os.path.join(path, "index.faiss")
            if os.path.exists(check):
                print("   Found FAISS at: " + path)
                break
        else:
            print("   ERROR: faiss_index not found!")
            print("   Run: python build_faiss_fast.py first")
            return None

    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings

        emb = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        store = FAISS.load_local(
            "faiss_index", emb,
            allow_dangerous_deserialization=True
        )

        retriever = store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5}
        )

        client = Groq(api_key=GROQ_API_KEY)

        print("   Chatbot loaded!")
        return {"retriever": retriever, "client": client}

    except Exception as e:
        print("   ERROR: " + str(e)[:200])
        return None


def ask_chatbot(bot, question):
    """Ask question to chatbot and get response."""
    try:
        # Retrieve context
        docs = bot["retriever"].invoke(question)
        context = "\n\n".join([d.page_content for d in docs])

        # Build prompt
        prompt = (
            "You are a Career Advisor with IT resume knowledge.\n\n"
            "CONTEXT:\n" + context + "\n\n"
            "QUESTION: " + question + "\n\n"
            "Recommend matching job roles. Available roles: "
            "Software Developer, Front End Developer, Network Administrator, "
            "Web Developer, Project Manager, Database Administrator, "
            "Security Analyst, Systems Administrator, Python Developer, Java Developer\n\n"
            "ANSWER:"
        )

        # Ask Groq
        r = bot["client"].chat.completions.create(
            model=GROQ_MODELS[0],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512, temperature=0.3
        )

        return r.choices[0].message.content.strip()

    except Exception as e:
        if "429" in str(e):
            time.sleep(10)
            try:
                r = bot["client"].chat.completions.create(
                    model=GROQ_MODELS[0],
                    messages=[{"role": "user", "content": question}],
                    max_tokens=256
                )
                return r.choices[0].message.content.strip()
            except:
                return "Rate limited"
        return "Error: " + str(e)[:100]


def run_accuracy_test(bot):
    questions = get_test_questions()
    results = []

    print_header("RUNNING ACCURACY TEST")
    print("   Testing " + str(len(questions)) + " questions\n")

    for i, q in enumerate(questions):
        question = q["question"]
        print("   [" + str(i + 1) + "/" + str(len(questions)) + "] " + question[:55])

        answer = ask_chatbot(bot, question)
        answer_lower = answer.lower()

        # Check roles
        roles_found = []
        roles_missed = []
        for role in q["expected_roles"]:
            role_words = role.lower().replace("_", " ")
            if role_words in answer_lower or role.lower() in answer_lower:
                roles_found.append(role)
            else:
                roles_missed.append(role)

        role_acc = len(roles_found) / max(len(q["expected_roles"]), 1) * 100

        # Check keywords
        kw_found = []
        kw_missed = []
        for kw in q["expected_keywords"]:
            if kw in answer_lower:
                kw_found.append(kw)
            else:
                kw_missed.append(kw)

        kw_acc = len(kw_found) / max(len(q["expected_keywords"]), 1) * 100

        # Check quality
        has_content = len(answer) > 100
        no_error = "error" not in answer_lower and "rate limit" not in answer_lower
        mentions_role = any(r in answer_lower for r in ["developer", "administrator", "analyst", "manager", "engineer"])
        quality = (has_content * 33 + no_error * 34 + mentions_role * 33)

        overall = (role_acc + kw_acc + quality) / 3

        result = {
            "question": question,
            "answer": answer,
            "role_accuracy": round(role_acc, 1),
            "keyword_accuracy": round(kw_acc, 1),
            "quality_score": round(quality, 1),
            "overall": round(overall, 1),
            "roles_found": roles_found,
            "roles_missed": roles_missed,
            "keywords_found": kw_found,
            "keywords_missed": kw_missed,
        }
        results.append(result)

        if overall >= 80:
            status = "OK"
        elif overall >= 50:
            status = "!!"
        else:
            status = "XX"

        print("      " + status + " Role: " + str(round(role_acc)) + "% | KW: " + str(round(kw_acc)) + "% | Quality: " + str(round(quality)) + "% | Overall: " + str(round(overall)) + "%")

        time.sleep(3)

    return results


def llm_accuracy_test(bot, results):
    print_header("LLM-BASED ACCURACY EVALUATION")
    print("   Using Groq to judge each answer...\n")

    llm_scores = []

    for i, r in enumerate(results):
        print("   Judging " + str(i + 1) + "/" + str(len(results)), end="\r")

        prompt = (
            "Rate this chatbot response accuracy from 1 to 10.\n\n"
            "QUESTION: " + r["question"] + "\n\n"
            "EXPECTED ROLES: " + ", ".join(r.get("roles_found", []) + r.get("roles_missed", [])) + "\n\n"
            "ANSWER (first 300 chars): " + r["answer"][:300] + "\n\n"
            "ROLES FOUND: " + str(r["roles_found"]) + "\n"
            "KEYWORDS FOUND: " + str(r["keywords_found"]) + "\n\n"
            "Reply with ONLY a number 1-10."
        )

        try:
            response = bot["client"].chat.completions.create(
                model=GROQ_MODELS[0],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10, temperature=0.1
            )
            text = response.choices[0].message.content.strip()
            nums = re.findall(r"(\d+)", text)
            score = int(nums[0]) if nums else 5
            score = min(max(score, 1), 10)
        except:
            score = 5

        llm_scores.append(score)
        time.sleep(3)

    print("\n")
    return llm_scores


def print_report(results, llm_scores=None):
    print_header("DETAILED ACCURACY REPORT")

    print("\n   PER-QUESTION RESULTS:")
    print("   " + "-" * 75)
    print("   {:<45} {:>8} {:>8} {:>8} {:>8}".format("Question", "Role%", "KW%", "Quality%", "Overall%"))
    print("   " + "-" * 75)

    for r in results:
        q = r["question"][:43]
        print("   {:<45} {:>7.0f} {:>7.0f} {:>7.0f} {:>7.0f}".format(
            q, r["role_accuracy"], r["keyword_accuracy"],
            r["quality_score"], r["overall"]
        ))

    avg_role = sum(r["role_accuracy"] for r in results) / len(results)
    avg_kw = sum(r["keyword_accuracy"] for r in results) / len(results)
    avg_quality = sum(r["quality_score"] for r in results) / len(results)
    avg_overall = sum(r["overall"] for r in results) / len(results)

    print("\n   " + "-" * 75)

    print_header("ACCURACY METRICS")

    metrics = [
        ("ROLE PREDICTION", avg_role, "Did it recommend the correct job role?"),
        ("KEYWORD MATCH", avg_kw, "Did it mention relevant skills/terms?"),
        ("RESPONSE QUALITY", avg_quality, "Was the response complete and error-free?"),
    ]

    for name, score, desc in metrics:
        bar = "#" * int(score / 5) + "." * (20 - int(score / 5))
        print("\n   " + name)
        print("   " + desc)
        print("   [" + bar + "]  " + str(round(score, 1)) + "%")

    if llm_scores:
        avg_llm = sum(llm_scores) / len(llm_scores) * 10
        bar = "#" * int(avg_llm / 5) + "." * (20 - int(avg_llm / 5))
        print("\n   LLM JUDGE SCORE")
        print("   How accurate does AI think the answers are?")
        print("   [" + bar + "]  " + str(round(avg_llm, 1)) + "%")

    if llm_scores:
        final = (avg_overall + avg_llm) / 2
    else:
        final = avg_overall

    print("\n   " + "=" * 50)
    bar = "#" * int(final / 5) + "." * (20 - int(final / 5))
    print("\n   OVERALL ACCURACY")
    print("   [" + bar + "]  " + str(round(final, 1)) + "%")

    if final >= 90:
        grade = "A+ (Excellent)"
    elif final >= 80:
        grade = "A (Very Good)"
    elif final >= 70:
        grade = "B (Good)"
    elif final >= 60:
        grade = "C (Satisfactory)"
    elif final >= 50:
        grade = "D (Needs Work)"
    else:
        grade = "F (Poor)"

    print("   GRADE: " + grade)

    print("\n   STRENGTHS:")
    found = False
    for r in results:
        if r["overall"] >= 80:
            print("   + " + r["question"][:50])
            found = True
    if not found:
        print("   No questions scored above 80%")

    print("\n   WEAKNESSES:")
    for r in results:
        if r["overall"] < 60:
            print("   - " + r["question"][:50])
            if r["roles_missed"]:
                print("     Missing: " + ", ".join(r["roles_missed"]))

    # Save results
    output = {
        "total_questions": len(results),
        "final_accuracy": round(final, 1),
        "grade": grade,
        "metrics": {
            "role_accuracy": round(avg_role, 1),
            "keyword_accuracy": round(avg_kw, 1),
            "quality_score": round(avg_quality, 1),
        },
        "per_question": [
            {
                "question": r["question"],
                "role_accuracy": r["role_accuracy"],
                "keyword_accuracy": r["keyword_accuracy"],
                "quality_score": r["quality_score"],
                "overall": r["overall"],
                "roles_found": r["roles_found"],
                "roles_missed": r["roles_missed"],
            }
            for r in results
        ]
    }

    if llm_scores:
        output["llm_judge_avg"] = round(avg_llm, 1)

    with open("accuracy_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n   Saved to: accuracy_results.json")
    return final


def main():
    print_header("RAG CHATBOT ACCURACY CHECKER")
    print("""
   This tool measures HOW ACCURATE your chatbot is.

   It tests 10 questions where we KNOW the correct answer,
   then checks if the chatbot gave the right answer.

   Metrics:
     1. Role Prediction  - Did it recommend the right job?
     2. Keyword Match    - Did it mention relevant terms?
     3. Response Quality - Was the answer complete?
     4. LLM Judge        - AI rates the answer quality

   Options:
     1. Full Test (with LLM judge) - 5 minutes
     2. Quick Test (no LLM judge) - 2 minutes
     3. Exit
    """)

    choice = input("   Choice (1-3): ").strip()

    if choice == "1" or choice == "2":
        bot = load_chatbot()
        if bot is None:
            print("\n   Cannot run test without chatbot.")
            print("   Make sure faiss_index/ exists.")
            print("   Run: python build_faiss_fast.py")
            return

        results = run_accuracy_test(bot)

        if choice == "1":
            llm_scores = llm_accuracy_test(bot, results)
            print_report(results, llm_scores)
        else:
            print_report(results)

    elif choice == "3":
        print("   Bye!")


if __name__ == "__main__":
    main()