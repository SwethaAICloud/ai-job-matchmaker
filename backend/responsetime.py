
# Chatbot Response Time Checker

## Create `response_time_checker.py`

import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import json
import statistics


def print_header(title):
    print("\n" + "=" * 60)
    print("   " + title)
    print("=" * 60)


def print_bar(label, value, max_val, unit="s"):
    filled = int(value / max_val * 20)
    bar = "#" * filled + "." * (20 - filled)
    print("   [" + bar + "]  " + str(round(value, 2)) + unit + "  " + label)


# ============================================================
# TEST QUESTIONS
# ============================================================

test_questions = [
    "Which role suits me if I know Python and SQL?",
    "I know JavaScript, React, HTML, CSS - what job?",
    "Java, Spring Boot, Microservices - suggest role",
    "Compare frontend vs backend development",
    "What salary can I expect with Python 3 years?",
    "Rate my profile: Python Django AWS 3 years backend",
    "I am a fresher with basic Python, what should I do?",
    "How do I become a senior developer?",
    "What skills do I need for Python Developer?",
    "Will I get hired with React and Node.js?",
]


# ============================================================
# METHOD 1: Test Flask API Response Time
# ============================================================

def test_api_response_time():
    """Test response time through Flask API."""
    import requests

    print_header("API RESPONSE TIME TEST")
    print("   Testing through Flask API (http://127.0.0.1:9090)")
    print("   Make sure app.py is running!\n")

    url = "http://127.0.0.1:9090/chat"
    times = []
    results = []

    for i, question in enumerate(test_questions):
        print("   [" + str(i+1) + "/" + str(len(test_questions)) + "] " + question[:50], end="")

        start = time.time()
        try:
            r = requests.post(url, json={"message": question}, timeout=60)
            elapsed = time.time() - start
            response = r.json().get("response", "")
            status = "OK"
            answer_len = len(response)
        except requests.ConnectionError:
            elapsed = 0
            status = "FAIL (app not running)"
            answer_len = 0
        except requests.Timeout:
            elapsed = 60
            status = "TIMEOUT"
            answer_len = 0
        except Exception as e:
            elapsed = time.time() - start
            status = "ERROR"
            answer_len = 0

        times.append(elapsed)
        results.append({
            "question": question,
            "time": round(elapsed, 2),
            "status": status,
            "answer_length": answer_len
        })

        print("  " + str(round(elapsed, 2)) + "s  " + status)
        time.sleep(1)

    return times, results


# ============================================================
# METHOD 2: Test Direct Chatbot Response Time
# ============================================================

def test_direct_response_time():
    """Test response time by calling chatbot directly."""

    print_header("DIRECT CHATBOT RESPONSE TIME TEST")
    print("   Testing chatbot directly (no Flask overhead)\n")

    # Import and initialize chatbot
    print("   Loading chatbot...")
    load_start = time.time()

    from app import NaturalChatbot
    bot = NaturalChatbot()

    load_time = time.time() - load_start
    print("   Chatbot loaded in " + str(round(load_time, 2)) + "s\n")

    times = []
    results = []

    # Breakdown times
    search_times = []
    llm_times = []
    format_times = []

    for i, question in enumerate(test_questions):
        print("   [" + str(i+1) + "/" + str(len(test_questions)) + "] " + question[:50], end="")

        # Total time
        total_start = time.time()

        # Search time
        search_start = time.time()
        docs = bot.hybrid_search(question)
        search_time = time.time() - search_start

        # Context building
        parts = []
        for j, doc in enumerate(docs):
            jobs = doc.metadata.get('jobs', 'Unknown')
            skills = doc.metadata.get('skills', '')[:200]
            parts.append("Profile " + str(j+1) + ": Role: " + jobs.replace('_',' ') + " Skills: " + skills + " " + doc.page_content[:400])
        context = "\n\n".join(parts)

        # Prompt building
        prompt = bot.build_prompt(question, context)

        # LLM time
        llm_start = time.time()
        answer = bot.llm.invoke(prompt)
        llm_time = time.time() - llm_start

        # Format time
        format_start = time.time()
        answer = bot.clean_markdown(answer)
        answer = bot.validate_response(answer, question)
        format_time = time.time() - format_start

        total_time = time.time() - total_start

        times.append(total_time)
        search_times.append(search_time)
        llm_times.append(llm_time)
        format_times.append(format_time)

        results.append({
            "question": question,
            "total_time": round(total_time, 2),
            "search_time": round(search_time, 3),
            "llm_time": round(llm_time, 2),
            "format_time": round(format_time, 4),
            "answer_length": len(answer)
        })

        print("  " + str(round(total_time, 2)) + "s (search:" + str(round(search_time, 3)) + " llm:" + str(round(llm_time, 2)) + ")")
        time.sleep(2)

    return times, results, search_times, llm_times, format_times, load_time


# ============================================================
# PRINT RESULTS
# ============================================================

def print_api_results(times, results):
    if not times or all(t == 0 for t in times):
        print("\n   No valid results. Is the Flask app running?")
        return

    valid_times = [t for t in times if t > 0]
    if not valid_times:
        return

    avg = statistics.mean(valid_times)
    median = statistics.median(valid_times)
    fastest = min(valid_times)
    slowest = max(valid_times)
    max_time = max(slowest, 1)

    print_header("API RESPONSE TIME RESULTS")

    print("\n   PER-QUESTION TIMES:")
    print("   " + "-" * 65)
    for r in results:
        q = r["question"][:40]
        t = r["time"]
        s = r["status"]
        bar = "#" * int(t / max_time * 15) + "." * (15 - int(t / max_time * 15))
        print("   [" + bar + "] " + str(t) + "s  " + s + "  " + q)

    print("\n   SUMMARY:")
    print("   " + "-" * 40)
    print_bar("Average", avg, max_time)
    print_bar("Median", median, max_time)
    print_bar("Fastest", fastest, max_time)
    print_bar("Slowest", slowest, max_time)

    print("\n   RATING:")
    if avg < 3:
        print("   EXCELLENT - Under 3 seconds average")
    elif avg < 5:
        print("   GOOD - Under 5 seconds average")
    elif avg < 10:
        print("   ACCEPTABLE - Under 10 seconds average")
    else:
        print("   SLOW - Over 10 seconds average. Consider optimization.")


def print_direct_results(times, results, search_times, llm_times, format_times, load_time):
    avg = statistics.mean(times)
    median = statistics.median(times)
    fastest = min(times)
    slowest = max(times)
    max_time = max(slowest, 1)

    avg_search = statistics.mean(search_times)
    avg_llm = statistics.mean(llm_times)
    avg_format = statistics.mean(format_times)

    print_header("DIRECT RESPONSE TIME RESULTS")

    print("\n   CHATBOT LOAD TIME: " + str(round(load_time, 2)) + "s")

    print("\n   PER-QUESTION BREAKDOWN:")
    print("   " + "-" * 75)
    print("   {:<38} {:>7} {:>8} {:>7} {:>7}".format("Question", "Total", "Search", "LLM", "Format"))
    print("   " + "-" * 75)

    for r in results:
        q = r["question"][:36]
        print("   {:<38} {:>6}s {:>7}s {:>6}s {:>6}s".format(
            q,
            str(r["total_time"]),
            str(r["search_time"]),
            str(r["llm_time"]),
            str(r["format_time"])
        ))

    print("\n   TIME BREAKDOWN (Average):")
    print("   " + "-" * 50)

    total = avg_search + avg_llm + avg_format
    if total > 0:
        search_pct = avg_search / total * 100
        llm_pct = avg_llm / total * 100
        format_pct = avg_format / total * 100
    else:
        search_pct = llm_pct = format_pct = 0

    print_bar("FAISS Search  (" + str(round(search_pct)) + "%)", avg_search, max_time)
    print_bar("Groq LLM      (" + str(round(llm_pct)) + "%)", avg_llm, max_time)
    print_bar("Post-Process   (" + str(round(format_pct)) + "%)", avg_format, max_time)

    print("\n   OVERALL STATS:")
    print("   " + "-" * 40)
    print_bar("Average Response", avg, max_time)
    print_bar("Median Response", median, max_time)
    print_bar("Fastest Response", fastest, max_time)
    print_bar("Slowest Response", slowest, max_time)

    print("\n   WHERE TIME IS SPENT:")
    print("   " + "-" * 40)
    print("   FAISS Search:    " + str(round(avg_search * 1000)) + " ms (" + str(round(search_pct)) + "%)")
    print("   Groq LLM Call:   " + str(round(avg_llm, 2)) + " s (" + str(round(llm_pct)) + "%)")
    print("   Post-Processing: " + str(round(avg_format * 1000, 1)) + " ms (" + str(round(format_pct)) + "%)")
    print("   Total Average:   " + str(round(avg, 2)) + " s")

    print("\n   BOTTLENECK: ", end="")
    if llm_pct > 80:
        print("Groq LLM (network latency)")
        print("   FIX: Use faster model or local LLM")
    elif search_pct > 50:
        print("FAISS Search (too many chunks)")
        print("   FIX: Reduce chunk count or optimize index")
    else:
        print("Well balanced - no major bottleneck")

    print("\n   RATING:")
    if avg < 3:
        print("   EXCELLENT - Under 3 seconds")
    elif avg < 5:
        print("   GOOD - Under 5 seconds")
    elif avg < 10:
        print("   ACCEPTABLE - Under 10 seconds")
    else:
        print("   SLOW - Over 10 seconds")

    # Save results
    output = {
        "load_time": round(load_time, 2),
        "average_response": round(avg, 2),
        "median_response": round(median, 2),
        "fastest": round(fastest, 2),
        "slowest": round(slowest, 2),
        "breakdown": {
            "search_avg_ms": round(avg_search * 1000),
            "llm_avg_s": round(avg_llm, 2),
            "format_avg_ms": round(avg_format * 1000, 1),
            "search_pct": round(search_pct, 1),
            "llm_pct": round(llm_pct, 1),
            "format_pct": round(format_pct, 1),
        },
        "per_question": results
    }

    with open("response_time_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n   Saved to: response_time_results.json")


# ============================================================
# MAIN
# ============================================================

def main():
    print_header("CAREERBUDDY RESPONSE TIME CHECKER")
    print("""
   Tests how fast the chatbot responds.

   Options:
   1. Test Flask API (needs app.py running)
   2. Test Direct Chatbot (loads chatbot here)
   3. Both
   4. Exit
    """)

    choice = input("   Choice (1-4): ").strip()

    if choice == "1":
        times, results = test_api_response_time()
        print_api_results(times, results)

    elif choice == "2":
        times, results, st, lt, ft, load = test_direct_response_time()
        print_direct_results(times, results, st, lt, ft, load)

    elif choice == "3":
        print("\n   --- API TEST ---")
        api_times, api_results = test_api_response_time()
        print_api_results(api_times, api_results)

        print("\n   --- DIRECT TEST ---")
        times, results, st, lt, ft, load = test_direct_response_time()
        print_direct_results(times, results, st, lt, ft, load)

    elif choice == "4":
        print("   Bye!")


if __name__ == "__main__":
    main()
