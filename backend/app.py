
import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import re
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import GROQ_API_KEY, EMBEDDING_MODEL, GROQ_MODELS

app = Flask(__name__)
app.secret_key = "secretkey123"
CORS(app, resources={r"/*": {"origins": "*"}})

chatbot = None


class SimpleGroqLLM:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.models = GROQ_MODELS

    def invoke(self, prompt):
        for model in self.models:
            try:
                r = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are CareerBuddy, a career advisor. "
                                "CRITICAL RULES:\n"
                                "1. ONLY use information from the provided REFERENCE DATA\n"
                                "2. Do NOT make up ANY facts, numbers, or claims\n"
                                "3. Do NOT add information not found in the context\n"
                                "4. If the context does not have enough info, explicitly say 'Based on available data'\n"
                                "5. Every recommendation MUST be traceable to the reference data\n"
                                "6. Do NOT hallucinate or invent details\n"
                                "7. When you mention a skill, it MUST appear in the reference data\n"
                                "8. When you recommend a role, the reference data MUST support it\n"
                                "9. Keep answers concise, under 250 words\n"
                                "10. Be helpful and encouraging but STAY GROUNDED in the data"
                            )
                        },
                        {
                            "role": "user",
                            "content": str(prompt)
                        }
                    ],
                    max_tokens=1200,
                    temperature=0.02
                )
                return r.choices[0].message.content.strip()
            except:
                continue
        return "Error generating response. Please try again."


class NaturalChatbot:
    def __init__(self):
        print("Loading FAISS...")
        emb = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.store = FAISS.load_local(
            "faiss_index", emb,
            allow_dangerous_deserialization=True
        )
        self.llm = SimpleGroqLLM(api_key=GROQ_API_KEY)
        self.retriever = self.store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.4}
        )
        self.last_call = 0
        print("Ready!")

    def respond(self, user_input):
        u = str(user_input).strip()
        if not u:
            return "Please ask about jobs, skills, or careers."
        if u.lower() in ["exit", "quit", "bye"]:
            return "Goodbye! Best of luck with your career!"
        if u.lower() in ["help", "hi", "hello", "hey"]:
            return self.help_text()
        if u.lower() in ["reset", "clear"]:
            return "Chat reset! Ask me anything about your career."
        elapsed = time.time() - self.last_call
        if elapsed < 3:
            time.sleep(3 - elapsed)
        try:
            docs = self.hybrid_search(u)
            context_parts = []
            for i, doc in enumerate(docs):
                jobs = doc.metadata.get('jobs', 'Unknown')
                skills = doc.metadata.get('skills', '')[:200]
                content = doc.page_content[:400]
                part = ("EVIDENCE " + str(i+1) + ":\n"
                    "  Role: " + jobs.replace('_', ' ') + "\n"
                    "  Skills found: " + skills + "\n"
                    "  Profile: " + content)
                context_parts.append(part)
            context = "\n\n".join(context_parts)
            prompt = self.build_prompt(u, context)
            answer = self.llm.invoke(prompt)
            answer = self.clean_markdown(answer)
            answer = self.validate_response(answer, u)
            self.last_call = time.time()
            return answer
        except Exception as e:
            if "429" in str(e):
                time.sleep(10)
                return "Rate limited. Please wait a moment."
            return "Error: " + str(e)[:200]

    def expand_query(self, question):
        expansions = {
            'python': 'python django flask pandas numpy backend developer',
            'java': 'java spring boot maven junit backend developer',
            'javascript': 'javascript react angular vue node frontend developer',
            'react': 'react javascript frontend ui component web developer',
            'sql': 'sql database postgresql mysql oracle query developer',
            'html': 'html css frontend web design responsive developer',
            'docker': 'docker kubernetes container devops deployment',
            'linux': 'linux bash shell ubuntu server administration',
            'aws': 'aws cloud amazon ec2 s3 lambda deployment',
            'azure': 'azure cloud microsoft devops infrastructure',
            'network': 'network cisco firewall tcp ip routing administrator',
            'security': 'security penetration testing firewall vulnerability analyst',
            'agile': 'agile scrum project management sprint kanban manager',
            'database': 'database sql oracle postgresql mysql administration dba',
            'node': 'node nodejs express backend javascript server',
            'django': 'django python web backend rest api developer',
            'spring': 'spring boot java microservices backend developer',
            'mobile': 'mobile ios android react native flutter app developer',
            'frontend': 'frontend react angular vue javascript html css developer',
            'backend': 'backend python java node api server database developer',
            'devops': 'devops docker kubernetes ci cd jenkins aws linux',
            'fresher': 'fresher entry level junior beginner graduate',
            'machine learning': 'machine learning python tensorflow pytorch data science',
            'cloud': 'cloud aws azure gcp infrastructure deployment',
            'full stack': 'full stack frontend backend react node python developer',
        }
        expanded = question
        t = question.lower()
        for keyword, expansion in expansions.items():
            if keyword in t:
                expanded = expanded + " " + expansion
        return expanded

    def hybrid_search(self, question, k=8):
        expanded = self.expand_query(question)
        semantic_docs = self.retriever.invoke(expanded)
        skills = [
            'python', 'java', 'javascript', 'react', 'sql', 'html', 'css',
            'django', 'flask', 'spring', 'node', 'angular', 'docker', 'linux',
            'aws', 'azure', 'network', 'security', 'database', 'agile', 'scrum',
            'mongodb', 'kubernetes', 'vue', 'typescript', 'machine learning',
            'devops', 'cloud', 'api', 'frontend', 'backend', 'full stack',
        ]
        query_skills = [s for s in skills if s in question.lower()]
        if not query_skills:
            return semantic_docs[:k]
        scored = []
        for doc in semantic_docs:
            content = doc.page_content.lower()
            jobs = doc.metadata.get('jobs', '').lower()
            skill_matches = sum(1 for s in query_skills if s in content)
            job_bonus = sum(1 for s in query_skills if s in jobs) * 2
            score = (skill_matches + job_bonus) / max(len(query_skills), 1)
            scored.append((doc, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored[:k]]

    def validate_response(self, answer, question):
        answer_lower = answer.lower()
        roles = [
            'software developer', 'front end developer', 'network administrator',
            'web developer', 'project manager', 'database administrator',
            'security analyst', 'systems administrator', 'python developer',
            'java developer', 'cloud engineer', 'devops engineer',
        ]
        has_role = any(role in answer_lower for role in roles)
        asks_role = any(w in question.lower() for w in ['role', 'job', 'suit', 'fit', 'match', 'recommend', 'career'])
        skills_in_q = any(s in question.lower() for s in ['python', 'java', 'javascript', 'react', 'sql', 'html', 'node', 'docker', 'linux', 'aws', 'azure', 'network', 'security', 'database', 'agile', 'django', 'flask', 'spring', 'cloud', 'devops'])
        if (asks_role or skills_in_q) and not has_role:
            skill_role_map = {
                'python': 'Python Developer', 'java': 'Java Developer',
                'javascript': 'Front End Developer', 'react': 'Front End Developer',
                'sql': 'Database Administrator', 'network': 'Network Administrator',
                'security': 'Security Analyst', 'linux': 'Systems Administrator',
                'docker': 'Systems Administrator', 'agile': 'Project Manager',
                'html': 'Web Developer', 'node': 'Web Developer',
                'aws': 'Cloud Engineer', 'azure': 'Cloud Engineer',
                'django': 'Python Developer', 'spring': 'Java Developer',
            }
            suggested = []
            for skill, role in skill_role_map.items():
                if skill in question.lower() and role not in suggested:
                    suggested.append(role)
            if suggested:
                answer = answer + "\n\nBased on the data, consider: " + ", ".join(suggested[:3])
        return answer

    def clean_markdown(self, text):
        text = re.sub(r'^#{1,3}\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        return text.strip()

    def build_prompt(self, question, context):
        t = question.lower()
        strict = ("\n\nSTRICT FAITHFULNESS RULES:\n"
            "- ONLY state facts found in the EVIDENCE sections above\n"
            "- For EVERY claim, it must appear in the evidence\n"
            "- If evidence does not support a claim, do NOT make it\n"
            "- Prefix uncertain statements with 'Based on available data'\n"
            "- Do NOT invent skills, roles, or details not in evidence\n"
            "- Do NOT list individual resumes or show Evidence 1, 2, etc\n"
            "- Combine all evidence into ONE unified answer\n"
            "- Keep under 250 words\n"
            "- NEVER use markdown symbols like # * or `\n"
            "- Use UPPERCASE for section headings\n"
            "- Use [check] for checkmarks, [arrow] for arrows\n"
            "- Use numbered lists and | for tables\n")
        roles = ("\n\nROLE MATCHING (use only if supported by evidence):\n"
            "- Python + SQL = Python Developer or Database Administrator\n"
            "- JavaScript + React = Front End Developer\n"
            "- Java + Spring = Java Developer\n"
            "- Docker + Kubernetes = Systems Administrator\n"
            "- Network + Firewall = Network Administrator\n"
            "- Security + Penetration = Security Analyst\n"
            "- Agile + Scrum = Project Manager\n"
            "- AWS/Azure = Cloud Engineer or Systems Administrator\n")
        if any(w in t for w in ["compare", "vs", "versus", "better", "difference"]):
            return ("EVIDENCE FROM PROFILES:\n" + context + "\n\n"
                "QUESTION: " + question + "\n\n"
                "Using ONLY the evidence above, provide:\n"
                "ROLE COMPARISON citing evidence for each role\n"
                "SIDE BY SIDE table\n"
                "RECOMMENDATION supported by evidence\n"
                "NEXT STEPS 1. 2. 3."
                + strict + roles)
        elif any(w in t for w in ["review", "rate", "score", "feedback", "grade"]):
            return ("EVIDENCE (strong profiles for comparison):\n" + context + "\n\n"
                "QUESTION: " + question + "\n\n"
                "Comparing with profiles in evidence, provide:\n"
                "ASSESSMENT grade based on comparison with evidence\n"
                "STRENGTHS [check] found in evidence comparison\n"
                "GAPS [arrow] what evidence profiles have that user lacks\n"
                "KEYWORDS found in successful evidence profiles\n"
                "ACTIONS 1. 2. 3. based on evidence patterns"
                + strict)
        elif any(w in t for w in ["salary", "pay", "earn", "money"]):
            return ("QUESTION: " + question + "\n\n"
                "VERIFIED SALARY DATA:\n"
                "US: Software Dev $70K-$170K | Frontend $65K-$155K | Python $75K-$175K\n"
                "Java $72K-$170K | Web $55K-$145K | DevOps $80K-$175K | Cloud $85K-$180K\n"
                "Security $65K-$160K | Network $50K-$120K | DB Admin $65K-$145K\n"
                "Systems $55K-$135K | PM $68K-$155K | ML $95K-$200K\n"
                "Ireland EUR: Junior 30K-45K | Mid 50K-75K | Senior 75K-120K\n"
                "India LPA: Fresher 3-6 | Junior 6-12 | Mid 12-25 | Senior 25-50\n\n"
                "PROFILE EVIDENCE:\n" + context + "\n\n"
                "Using ONLY the salary data AND evidence above:\n"
                "SALARY ESTIMATE one sentence\n"
                "YOUR RANGE table Level | Annual | Monthly\n"
                "BY ROLE 2-3 roles with ranges FROM the data above only\n"
                "VALUE BOOSTERS [check] 2-3 from evidence\n"
                "EARN MORE 1. 2. 3.\n"
                "IMPORTANT: Use ONLY numbers from the salary data above."
                + strict)
        elif any(w in t for w in ["career path", "roadmap", "become", "senior", "grow"]):
            return ("EVIDENCE (career trajectories):\n" + context + "\n\n"
                "QUESTION: " + question + "\n\n"
                "Using ONLY patterns found in evidence above:\n"
                "CURRENT LEVEL based on evidence comparison\n"
                "ROADMAP PHASE 1/2/3 based on career patterns in evidence\n"
                "PATHS seen in the evidence profiles\n"
                "START NOW 1. 2. 3. supported by evidence"
                + strict + roles)
        elif any(w in t for w in ["hire", "chances", "probability", "get hired"]):
            return ("EVIDENCE (similar candidates):\n" + context + "\n\n"
                "QUESTION: " + question + "\n\n"
                "Comparing with evidence profiles:\n"
                "ASSESSMENT based on evidence comparison\n"
                "STANDOUT [check] supported by evidence\n"
                "CONCERNS [arrow] gaps compared to evidence profiles\n"
                "TARGET ROLES 1. 2. 3. from evidence\n"
                "ACTIONS [check] based on evidence patterns"
                + strict + roles)
        elif any(w in t for w in ["need", "require", "gap", "learn", "missing"]):
            return ("EVIDENCE (profiles in target role):\n" + context + "\n\n"
                "QUESTION: " + question + "\n\n"
                "Using ONLY skills found in evidence above:\n"
                "ASSESSMENT target role from evidence\n"
                "YOU HAVE [check] skills that appear in evidence\n"
                "YOU NEED skills that evidence profiles have\n"
                "PLAN based on priority skills from evidence\n"
                "PROJECTS based on what evidence profiles mention"
                + strict + roles)
        else:
            return ("EVIDENCE FROM PROFILES:\n" + context + "\n\n"
                "QUESTION: " + question + "\n\n"
                "Using ONLY the evidence above, provide:\n"
                "SUGGESTED ROLES each supported by specific evidence\n"
                "1. Role citing which evidence supports it\n"
                "2. Role with evidence\n"
                "3. Role with evidence\n\n"
                "GROWTH ROLES table Role | Skills from evidence\n"
                "RECOMMENDATIONS [check] based on evidence patterns\n"
                "TOP PICK role with strongest evidence support\n\n"
                "Available roles: Software Developer, Front End Developer, "
                "Network Administrator, Web Developer, Project Manager, "
                "Database Administrator, Security Analyst, Systems Administrator, "
                "Python Developer, Java Developer"
                + strict + roles)

    def help_text(self):
        return ("Welcome to CareerBuddy!\n\nSmarter careers start here.\n\n"
            "WHAT I CAN DO\n\n"
            "[check] Find your ideal IT role\n"
            "[check] Compare career paths\n"
            "[check] Score your resume\n"
            "[check] Estimate salary\n"
            "[check] Plan career growth\n"
            "[check] Identify skill gaps\n"
            "[check] Assess hire chances\n\n"
            "Just type your question naturally!")


def get_bot():
    global chatbot
    if chatbot is None:
        chatbot = NaturalChatbot()
    return chatbot


PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CareerBuddy - Smarter Careers Start Here</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;background:#F3F6FB;color:#1a202c;min-height:100vh}
.app{max-width:800px;margin:0 auto;padding:16px}

/* Header */
.header{text-align:center;padding:24px 0 16px}
.logo-row{display:flex;align-items:center;justify-content:center;gap:12px;margin-bottom:6px}
.logo-icon{width:44px;height:44px;background:#0A66C2;border-radius:12px;display:flex;align-items:center;justify-content:center;position:relative;box-shadow:0 4px 12px rgba(10,102,194,0.25)}
.logo-icon svg{width:24px;height:24px}
.logo-name{font-size:26px;font-weight:800;color:#0A66C2;letter-spacing:-0.5px}
.tagline{font-size:13px;color:#6B7B8D;font-weight:400;margin-top:2px;font-style:italic}
.tech-bar{display:flex;gap:6px;justify-content:center;margin-top:10px}
.tech-tag{padding:3px 8px;border-radius:4px;font-size:10px;font-weight:600}
.t1{background:#E8F4FD;color:#0A66C2}
.t2{background:#E6F9F0;color:#0D9E5F}
.t3{background:#FFF3E0;color:#E67E22}
.t4{background:#F3E8FD;color:#8E44AD}

/* Chat */
.chat{background:#fff;border:1px solid #E2E8F0;border-radius:16px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,0.06);margin-top:12px}
.chat-top{padding:12px 18px;border-bottom:1px solid #E2E8F0;display:flex;align-items:center;justify-content:space-between;background:#FAFBFC}
.chat-top-left{display:flex;align-items:center;gap:10px}
.ai-badge{width:32px;height:32px;background:#0A66C2;border-radius:10px;display:flex;align-items:center;justify-content:center}
.ai-badge svg{width:18px;height:18px}
.chat-name{font-size:14px;font-weight:700;color:#1A202C}
.chat-stat{font-size:11px;color:#8B95A5;display:flex;align-items:center;gap:4px}
.online-dot{width:6px;height:6px;background:#0D9E5F;border-radius:50%;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
.clr{background:none;border:1px solid #E2E8F0;color:#8B95A5;padding:5px 10px;border-radius:6px;font-size:11px;cursor:pointer;font-family:inherit;font-weight:500}
.clr:hover{background:#FFF0F0;color:#E53E3E;border-color:#FEB2B2}

/* Messages */
.msgs{height:480px;overflow-y:auto;padding:18px;background:#F8FAFC}
.msgs::-webkit-scrollbar{width:3px}
.msgs::-webkit-scrollbar-thumb{background:#CBD5E0;border-radius:10px}

/* Welcome */
.welcome{display:flex;flex-direction:column;align-items:center;padding:40px 20px;text-align:center}
.welcome-logo{width:64px;height:64px;background:#0A66C2;border-radius:18px;display:flex;align-items:center;justify-content:center;margin-bottom:16px;box-shadow:0 6px 20px rgba(10,102,194,0.2)}
.welcome-logo svg{width:34px;height:34px}
.welcome h2{font-size:20px;color:#1A202C;font-weight:700;margin-bottom:4px}
.welcome p{font-size:13px;color:#8B95A5;line-height:1.6;max-width:360px}
.welcome-hint{margin-top:20px;font-size:12px;color:#A0AEC0;background:#F0F4F8;padding:10px 16px;border-radius:8px;max-width:380px}

/* Messages */
.mr{display:flex;gap:8px;margin-bottom:14px}
.mr.user{flex-direction:row-reverse}
.mav{width:26px;height:26px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex-shrink:0;margin-top:2px;color:#fff}
.mav.bot{background:#0A66C2}
.mav.user{background:#E67E22}
.mbody{max-width:78%}
.mt{padding:11px 15px;border-radius:14px;font-size:13px;line-height:1.8}
.mr.bot .mt{background:#fff;border:1px solid #E2E8F0;border-top-left-radius:4px;color:#2D3748}
.mr.user .mt{background:#0A66C2;border-top-right-radius:4px;color:#fff}
.mtime{font-size:9px;color:#A0AEC0;margin-top:2px;padding:0 4px}
.mr.user .mtime{text-align:right}

/* Typing */
.typ{display:none;padding:0 18px 8px}
.typ-inner{display:flex;align-items:center;gap:8px}
.typ-av{width:26px;height:26px;background:#0A66C2;border-radius:8px;display:flex;align-items:center;justify-content:center}
.typ-av svg{width:14px;height:14px}
.typ-dots{display:flex;gap:4px;background:#fff;padding:7px 14px;border-radius:14px;border:1px solid #E2E8F0}
.typ-dots span{width:5px;height:5px;border-radius:50%;animation:db 1.4s infinite}
.typ-dots span:nth-child(1){background:#0A66C2}
.typ-dots span:nth-child(2){background:#0D9E5F;animation-delay:.15s}
.typ-dots span:nth-child(3){background:#E67E22;animation-delay:.3s}
@keyframes db{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-3px)}}

/* Input */
.ibar{padding:12px 16px;border-top:1px solid #E2E8F0;display:flex;gap:8px;background:#FAFBFC}
.ifield{flex:1;padding:11px 16px;border-radius:10px;border:1px solid #E2E8F0;background:#fff;color:#2D3748;font-size:13px;font-family:inherit}
.ifield:focus{outline:none;border-color:#0A66C2;box-shadow:0 0 0 3px rgba(10,102,194,0.1)}
.ifield::placeholder{color:#A0AEC0}
.sbtn{width:40px;height:40px;border-radius:10px;border:none;background:#0A66C2;color:#fff;font-size:15px;cursor:pointer;display:flex;align-items:center;justify-content:center;box-shadow:0 2px 8px rgba(10,102,194,0.2)}
.sbtn:hover{background:#084B8A}

/* Footer */
.foot{text-align:center;padding:8px 0 14px;font-size:10px;color:#A0AEC0}
.foot b{color:#0A66C2}

@media(max-width:600px){.msgs{height:380px}.mbody{max-width:88%}}
</style>
</head>
<body>
<div class="app">

<!-- Header -->
<div class="header">
<div class="logo-row">
<div class="logo-icon">
<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
<circle cx="12" cy="12" r="10"/>
<polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>
</svg>
</div>
<span class="logo-name">CareerBuddy</span>
</div>
<div class="tagline">Smarter careers start here</div>
<div class="tech-bar">
<span class="tech-tag t1">LangChain</span>
<span class="tech-tag t2">FAISS</span>
<span class="tech-tag t3">Groq</span>
<span class="tech-tag t4">29K+ Resumes</span>
</div>
</div>

<!-- Chat -->
<div class="chat">
<div class="chat-top">
<div class="chat-top-left">
<div class="ai-badge">
<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
<circle cx="12" cy="12" r="10"/>
<polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>
</svg>
</div>
<div>
<div class="chat-name">CareerBuddy</div>
<div class="chat-stat"><span class="online-dot"></span> Ready to help</div>
</div>
</div>
<button class="clr" id="clearBtn">Clear Chat</button>
</div>

<div class="msgs" id="messages">
<div class="welcome" id="welcome">
<div class="welcome-logo">
<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<circle cx="12" cy="12" r="10"/>
<polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>
</svg>
</div>
<h2>Hey there! I'm CareerBuddy</h2>
<p>I analyze thousands of IT resumes to help you find the right role, bridge skill gaps, and plan your career growth.</p>
<div class="welcome-hint">Try asking: "Which role suits me if I know Python and SQL?" or "What salary can I expect in Ireland?"</div>
</div>
</div>

<div class="typ" id="typing">
<div class="typ-inner">
<div class="typ-av">
<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
<circle cx="12" cy="12" r="10"/>
<polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>
</svg>
</div>
<div class="typ-dots"><span></span><span></span><span></span></div>
</div>
</div>

<div class="ibar">
<input type="text" class="ifield" id="inputField" placeholder="Ask me anything about your career..." autofocus>
<button class="sbtn" id="sendBtn">&#10148;</button>
</div>
</div>

<div class="foot">Powered by <b>CareerBuddy</b> &mdash; Smarter careers start here</div>
</div>

<script>
var inp=document.getElementById("inputField");
var msgs=document.getElementById("messages");
var typ=document.getElementById("typing");

document.getElementById("sendBtn").addEventListener("click",doSend);
inp.addEventListener("keydown",function(e){if(e.key==="Enter"){e.preventDefault();doSend()}});
document.getElementById("clearBtn").addEventListener("click",function(){
    msgs.innerHTML="<div class='welcome' id='welcome'><div class='welcome-logo'><svg viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='10'/><polygon points='16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76'/></svg></div><h2>Fresh start!</h2><p>Ask me anything about your career.</p></div>";
    fetch("/reset",{method:"POST"});
});

function doSend(){
    var msg=inp.value.trim();
    if(!msg)return;
    inp.value="";
    var w=document.getElementById("welcome");
    if(w)w.remove();
    addMsg(msg,"user");
    typ.style.display="block";
    sb();
    fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:msg})})
    .then(function(r){return r.json()})
    .then(function(d){typ.style.display="none";addMsg(d.response,"bot")})
    .catch(function(e){typ.style.display="none";addMsg("Connection error.","bot")});
}

function fmt(text){
    var lines=text.split("\n");
    var html="";
    for(var i=0;i<lines.length;i++){
        var line=lines[i];
        var t=line.trim();
        if(!t){html+="<div style='height:6px'></div>";continue}
        if(t.includes("|")&&t.split("|").length>=3){
            var cells=t.split("|").map(function(c){return c.trim()}).filter(function(c){return c});
            if(cells.every(function(c){return/^[\-\s:]+$/.test(c)}))continue;
            html+="<div style='display:flex;gap:0;margin:1px 0'>";
            cells.forEach(function(cell){html+="<div style='flex:1;padding:5px 8px;background:"+(i%2===0?"#E8F4FD":"#fff")+";border:1px solid #E2E8F0;font-size:12px;color:#4A5568'>"+esc(cell)+"</div>"});
            html+="</div>";continue;
        }
        if(t.length>3&&t.length<80&&t===t.toUpperCase()&&t.match(/[A-Z]{3,}/)){
            html+="<div style='font-size:12px;font-weight:700;color:#0A66C2;margin:14px 0 5px;letter-spacing:0.3px;border-bottom:2px solid #E8F4FD;padding-bottom:3px'>"+esc(t)+"</div>";continue;
        }
        if(t.startsWith("[check]")||t.startsWith("\u2705")){
            var c=t.replace(/^\[check\]\s*/,"").replace(/^\u2705\s*/,"");
            html+="<div style='margin:3px 0 3px 8px;color:#2D3748;font-size:13px;display:flex;gap:7px'><span style='color:#0D9E5F;flex-shrink:0'>\u2705</span><span>"+esc(c)+"</span></div>";continue;
        }
        if(t.startsWith("[arrow]")||t.startsWith("\u27A1")){
            var c=t.replace(/^\[arrow\]\s*/,"").replace(/^\u27A1\uFE0F?\s*/,"");
            html+="<div style='margin:3px 0 3px 8px;color:#4A5568;font-size:13px;display:flex;gap:7px'><span style='color:#E67E22;flex-shrink:0'>\u27A1\uFE0F</span><span>"+esc(c)+"</span></div>";continue;
        }
        if(t.startsWith("[up]")){
            var c=t.replace(/^\[up\]\s*/,"");
            html+="<div style='margin:3px 0 3px 8px;color:#0D9E5F;font-size:13px;display:flex;gap:7px'><span>\uD83D\uDCC8</span><span>"+esc(c)+"</span></div>";continue;
        }
        if(t.startsWith("[down]")){
            var c=t.replace(/^\[down\]\s*/,"");
            html+="<div style='margin:3px 0 3px 8px;color:#E53E3E;font-size:13px;display:flex;gap:7px'><span>\uD83D\uDCC9</span><span>"+esc(c)+"</span></div>";continue;
        }
        if(t.endsWith(":")&&t.length<80&&!t.startsWith("-")&&!t.match(/^\d/)){
            html+="<div style='font-size:13px;font-weight:600;color:#1A202C;margin:8px 0 3px'>"+esc(t)+"</div>";continue;
        }
        if(t.match(/^[\-\*\u2022]\s/)){
            var c=t.replace(/^[\-\*\u2022]\s+/,"");
            var ci=c.indexOf(":");
            if(ci>0&&ci<40){html+="<div style='margin:2px 0 2px 14px;color:#4A5568;font-size:13px'>\u2022 <span style='color:#0A66C2;font-weight:600'>"+esc(c.substring(0,ci))+":</span> "+esc(c.substring(ci+1).trim())+"</div>"}
            else{html+="<div style='margin:2px 0 2px 14px;color:#4A5568;font-size:13px'>\u2022 "+esc(c)+"</div>"}
            continue;
        }
        if(t.match(/^\d+[\.\)]\s/)){
            var m=t.match(/^(\d+)[\.\)]\s+(.*)/);
            if(m){var c=m[2];var ci=c.indexOf(":");
            if(ci>0&&ci<40){html+="<div style='margin:3px 0 3px 4px;color:#4A5568;font-size:13px;display:flex;gap:7px'><span style='color:#0A66C2;font-weight:700;flex-shrink:0;min-width:16px'>"+m[1]+".</span><span><span style='color:#1A202C;font-weight:600'>"+esc(c.substring(0,ci))+":</span> "+esc(c.substring(ci+1).trim())+"</span></div>"}
            else{html+="<div style='margin:3px 0 3px 4px;color:#4A5568;font-size:13px;display:flex;gap:7px'><span style='color:#0A66C2;font-weight:700;flex-shrink:0;min-width:16px'>"+m[1]+".</span><span>"+esc(c)+"</span></div>"}}
            continue;
        }
        if(line.match(/^\s{2,}/)&&t.length>0){
            var ci=t.indexOf(":");
            if(ci>0&&ci<30){html+="<div style='margin:1px 0 1px 26px;font-size:12px;color:#6B7B8D'><span style='color:#4A5568;font-weight:500'>"+esc(t.substring(0,ci))+":</span> "+esc(t.substring(ci+1).trim())+"</div>"}
            else{html+="<div style='margin:1px 0 1px 26px;font-size:12px;color:#6B7B8D'>"+esc(t)+"</div>"}
            continue;
        }
        if(t.match(/\$[\d,]+/)){html+="<div style='margin:2px 0;color:#0D9E5F;font-weight:600;font-size:13px'>"+esc(t)+"</div>";continue}
        html+="<div style='margin:2px 0;color:#4A5568;line-height:1.7;font-size:13px'>"+esc(t)+"</div>";
    }
    return html;
}

function esc(t){return t.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}

function addMsg(text,type){
    var row=document.createElement("div");
    row.className="mr "+type;
    var av=document.createElement("div");
    av.className="mav "+type;
    if(type==="bot"){
        av.innerHTML="<svg viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round' width='14' height='14'><circle cx='12' cy='12' r='10'/><polygon points='16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76'/></svg>";
    } else {
        av.textContent="You";
    }
    var body=document.createElement("div");
    body.className="mbody";
    var bub=document.createElement("div");
    bub.className="mt";
    if(type==="bot"){bub.innerHTML=fmt(text)}else{bub.textContent=text}
    var tm=document.createElement("div");
    tm.className="mtime";
    var now=new Date();
    tm.textContent=(type==="bot"?"CareerBuddy":"You")+" \u00b7 "+now.getHours().toString().padStart(2,"0")+":"+now.getMinutes().toString().padStart(2,"0");
    body.appendChild(bub);
    body.appendChild(tm);
    row.appendChild(av);
    row.appendChild(body);
    msgs.appendChild(row);
    sb();
}

function sb(){setTimeout(function(){msgs.scrollTop=msgs.scrollHeight},50)}
</script>
</body>
</html>"""


@app.route("/")
def home():
    return render_template_string(PAGE)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        msg = data.get("message", "")
        response = get_bot().respond(msg)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": "Error: " + str(e)[:200]})

@app.route("/reset", methods=["POST"])
def reset():
    return jsonify({"status": "ok"})

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"response": "No file uploaded"})
    file = request.files["file"]
    return jsonify({"response": "Resume " + file.filename + " received."})

if __name__ == "__main__":

    print("CareerBuddy running at http://127.0.0.1:9090")
    app.run(debug=False, port=9090, host="127.0.0.1")