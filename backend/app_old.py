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
                                "You are CareerBuddy, a warm and knowledgeable career advisor. "
                                "Use the provided reference data as your PRIMARY source. "
                                "You may add general career advice and industry knowledge to make answers helpful. "
                                "Always be specific, actionable, and encouraging. "
                                "Give detailed answers with multiple options and explanations."
                            )
                        },
                        {
                            "role": "user",
                            "content": str(prompt)
                        }
                    ],
                    max_tokens=1500,
                    temperature=0.15
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
                part = ("Profile " + str(i+1) + ":\n"
                    "  Role: " + jobs.replace('_', ' ') + "\n"
                    "  Skills: " + skills + "\n"
                    "  Details: " + content)
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
        quality_rules = ("\n\nOUTPUT RULES:\n"
            "- Give DETAILED answers with multiple roles and explanations\n"
            "- For each role: explain WHY it fits, what you would DO daily, key technologies\n"
            "- Always include UPSKILLING suggestions with specific resources\n"
            "- Always include ACTIONABLE next steps\n"
            "- Use the reference data as your primary source\n"
            "- You may add general career knowledge to enrich the answer\n"
            "- NEVER use markdown symbols like # * or `\n"
            "- Use UPPERCASE for section headings\n"
            "- Use [check] for checkmark items, [arrow] for improvements\n"
            "- Use numbered lists and | for tables\n"
            "- Do NOT list individual resumes or show Profile 1, 2, etc\n"
            "- Combine all information into ONE unified answer\n"
            "- Be warm, specific, and encouraging\n")
        role_rules = ("\n\nROLE MATCHING:\n"
            "- Python + SQL = Python Developer or Database Administrator\n"
            "- JavaScript + React = Front End Developer\n"
            "- Java + Spring = Java Developer\n"
            "- Docker + Kubernetes = Systems Administrator or DevOps\n"
            "- Network + Firewall = Network Administrator\n"
            "- Security + Penetration = Security Analyst\n"
            "- Agile + Scrum = Project Manager\n"
            "- Database + Oracle = Database Administrator\n"
            "- AWS/Azure = Cloud Engineer or Systems Administrator\n"
            "- Always suggest the MOST SPECIFIC role first\n"
            "- Always list at least 2-3 matching roles\n")
        if any(w in t for w in ["compare", "vs", "versus", "better", "difference"]):
            return ("You are CareerBuddy. Help compare career paths.\n\n"
                "REFERENCE DATA:\n" + context + "\n\n"
                "QUESTION: " + question + "\n\n"
                "Provide a DETAILED comparison:\n\n"
                "ROLE COMPARISON\n\n"
                "First role:\n"
                "- What you would do day-to-day\n"
                "- Key technologies and tools\n"
                "- Salary range\n"
                "- Career growth path\n"
                "- Who this role is best for\n\n"
                "Second role:\n"
                "- Same details as above\n\n"
                "SIDE BY SIDE\n"
                "Aspect | Role 1 | Role 2 table\n\n"
                "MY RECOMMENDATION\n"
                "Clear recommendation with detailed reasoning\n\n"
                "UPSKILLING PATH\n"
                "[check] Skills to learn for each role\n"
                "[check] Free resources and courses\n"
                "[check] Certifications that help\n\n"
                "NEXT STEPS\n"
                "1. Most important action\n"
                "2. Second priority\n"
                "3. Long term goal"
                + quality_rules + role_rules)
        elif any(w in t for w in ["review", "rate", "score", "feedback", "grade"]):
            return ("You are CareerBuddy. Review this profile thoroughly.\n\n"
                "REFERENCE DATA (strong profiles for comparison):\n" + context + "\n\n"
                "QUESTION: " + question + "\n\n"
                "Provide a DETAILED review:\n\n"
                "PROFILE ASSESSMENT\n"
                "Overall Grade: [A/B/C/D] with explanation\n\n"
                "STRENGTHS\n"
                "[check] Each strength with WHY it matters to employers\n"
                "[check] At least 3-4 strengths\n\n"
                "AREAS TO IMPROVE\n"
                "[arrow] Each area with SPECIFIC action to fix it\n"
                "[arrow] Include what to learn and where\n\n"
                "MISSING SKILLS\n"
                "Skills that strong profiles have but you are missing:\n"
                "- Skill 1 with free resource to learn it\n"
                "- Skill 2 with resource\n\n"
                "ATS KEYWORDS\n"
                "Keywords to add for applicant tracking systems\n\n"
                "UPSKILLING PLAN\n"
                "1. This week: quick improvement\n"
                "2. This month: skill to learn with course name\n"
                "3. Next 3 months: certification or project goal\n\n"
                "PRO TIP\n"
                "One insider tip most candidates miss"
                + quality_rules)
        elif any(w in t for w in ["salary", "pay", "earn", "money"]):
            return ("You are CareerBuddy, a salary advisor.\n\n"
                "QUESTION: " + question + "\n\n"
                "SALARY DATA:\n"
                "US: Software Dev $70K-$170K | Frontend $65K-$155K | Python $75K-$175K\n"
                "Java $72K-$170K | Web $55K-$145K | DevOps $80K-$175K | Cloud $85K-$180K\n"
                "Security $65K-$160K | Network $50K-$120K | DB Admin $65K-$145K\n"
                "Systems $55K-$135K | PM $68K-$155K | ML $95K-$200K\n\n"
                "Ireland EUR: Junior 30K-45K | Mid 50K-75K | Senior 75K-120K\n"
                "India LPA: Fresher 3-6 | Junior 6-12 | Mid 12-25 | Senior 25-50\n\n"
                "Boosters: AWS cert +10-15% | Masters +5-10% | FAANG +30-50%\n\n"
                "REFERENCE PROFILES:\n" + context + "\n\n"
                "Provide a DETAILED salary analysis:\n\n"
                "SALARY ESTIMATE\n"
                "Your expected range based on skills and experience\n\n"
                "YOUR SALARY RANGE\n"
                "Level | Annual | Monthly table\n"
                "Show the relevant currency (USD/EUR/LPA based on question)\n\n"
                "BY MATCHING ROLE\n"
                "- 2-3 roles with specific salary ranges\n\n"
                "WHAT BOOSTS YOUR VALUE\n"
                "[check] 3-4 factors with estimated boost percentage\n\n"
                "WHAT MIGHT LIMIT\n"
                "[arrow] Limitations with how to overcome each\n\n"
                "HOW TO INCREASE YOUR SALARY\n"
                "1. Quick wins (0-6 months) with specific action\n"
                "2. Medium term (6-18 months) with certification or skill\n"
                "3. Long game (2+ years) with career move\n\n"
                "NEGOTIATION TIPS\n"
                "[check] 2-3 practical negotiation tips\n\n"
                "MARKET INSIGHT\n"
                "Current demand and trends for their skills"
                + quality_rules)
        elif any(w in t for w in ["career path", "roadmap", "become", "senior", "grow"]):
            return ("You are CareerBuddy. Create a detailed career roadmap.\n\n"
                "REFERENCE DATA:\n" + context + "\n\n"
                "QUESTION: " + question + "\n\n"
                "Provide a DETAILED career plan:\n\n"
                "WHERE YOU ARE NOW\n"
                "Current level: Entry/Junior/Mid/Senior with reasoning\n"
                "Best fit right now: Role name with explanation\n\n"
                "YOUR CAREER ROADMAP\n\n"
                "PHASE 1: Foundation (Now to Year 1)\n"
                "- Target role and why\n"
                "- 3-4 skills to build with free resources\n"
                "- 2 projects to complete\n"
                "- Certifications to get\n"
                "- Expected salary range\n\n"
                "PHASE 2: Growth (Year 1-3)\n"
                "- Next role target\n"
                "- Advanced skills to add\n"
                "- Specialization area\n"
                "- Expected salary range\n\n"
                "PHASE 3: Senior Level (Year 3-5)\n"
                "- Senior role target\n"
                "- Leadership skills\n"
                "- Industry recognition\n"
                "- Expected salary range\n\n"
                "ALTERNATIVE PATHS\n"
                "- Technical track: Stay hands-on\n"
                "- Management track: Lead teams\n"
                "- Specialist track: Deep expertise\n\n"
                "UPSKILLING RESOURCES\n"
                "[check] Free courses and platforms\n"
                "[check] Certifications worth getting\n"
                "[check] Communities to join\n\n"
                "START TODAY\n"
                "1. Most important action right now\n"
                "2. This week\n"
                "3. This month"
                + quality_rules + role_rules)
        elif any(w in t for w in ["hire", "chances", "probability", "get hired"]):
            return ("You are CareerBuddy. Give an honest hiring assessment.\n\n"
                "REFERENCE DATA:\n" + context + "\n\n"
                "QUESTION: " + question + "\n\n"
                "Provide a DETAILED assessment:\n\n"
                "HIRING ASSESSMENT\n"
                "Verdict: STRONG CANDIDATE / COMPETITIVE / GETTING THERE / EARLY STAGE\n"
                "2-3 sentences about overall position\n\n"
                "WHAT MAKES YOU STAND OUT\n"
                "[check] 3-4 strengths with WHY employers value each\n\n"
                "AREAS OF CONCERN\n"
                "[arrow] Each concern with SPECIFIC action to address it\n\n"
                "BEST ROLES TO TARGET\n"
                "1. Primary role with why and success likelihood\n"
                "2. Second option with reasoning\n"
                "3. Backup role with explanation\n\n"
                "BOOST YOUR CHANCES\n"
                "[check] Most impactful action with expected result\n"
                "[check] Second action\n"
                "[check] Third action\n\n"
                "UPSKILLING TO IMPROVE CHANCES\n"
                "- Skill to learn with free resource\n"
                "- Certification to get\n"
                "- Project to build\n\n"
                "INTERVIEW PREPARATION\n"
                "- Technical topics to review\n"
                "- Common questions to prepare\n"
                "- Portfolio tips\n\n"
                "INSIDER TIP\n"
                "One thing about hiring most people dont know"
                + quality_rules + role_rules)
        elif any(w in t for w in ["need", "require", "gap", "learn", "missing"]):
            return ("You are CareerBuddy. Analyze skills thoroughly.\n\n"
                "REFERENCE DATA:\n" + context + "\n\n"
                "QUESTION: " + question + "\n\n"
                "Provide a DETAILED skill analysis:\n\n"
                "SKILLS ASSESSMENT\n"
                "Target role: with explanation\n"
                "Readiness: Almost there / Halfway / Just starting\n\n"
                "SKILLS YOU ALREADY HAVE\n"
                "[check] Each skill with how it applies to the target role\n"
                "[check] At least 3-4 skills\n\n"
                "SKILLS YOU NEED TO LEARN\n\n"
                "Priority | Skill | Time to Learn | Free Resource\n"
                "1 (Critical) | Skill | X weeks | Specific course or platform\n"
                "2 (Important) | Skill | X weeks | Resource\n"
                "3 (Helpful) | Skill | X weeks | Resource\n"
                "4 (Bonus) | Skill | X weeks | Resource\n\n"
                "LEARNING ROADMAP\n"
                "Month 1: What to learn first and why\n"
                "Month 2: Next skill, builds on month 1\n"
                "Month 3: Advanced skill + build a project\n\n"
                "PROJECTS TO BUILD\n"
                "1. Project idea with what it demonstrates to employers\n"
                "2. Second project with what it proves\n\n"
                "CERTIFICATIONS\n"
                "[check] Certification name with cost and why it matters\n"
                "[check] Second certification\n\n"
                "TIMELINE\n"
                "With focused effort: about X months to be interview-ready"
                + quality_rules + role_rules)
        else:
            return ("You are CareerBuddy. Help find the perfect IT role.\n\n"
                "REFERENCE DATA:\n" + context + "\n\n"
                "QUESTION: " + question + "\n\n"
                "Provide a DETAILED role recommendation:\n\n"
                "Start with 1-2 friendly sentences addressing their question.\n\n"
                "BEST MATCHING ROLES\n\n"
                "1. [Primary Role]\n"
                "   - Why this fits your skills\n"
                "   - What you would do day-to-day\n"
                "   - Key technologies you would use\n"
                "   - Salary range\n\n"
                "2. [Second Role]\n"
                "   - Why this is also a good fit\n"
                "   - Daily responsibilities\n"
                "   - Technologies\n"
                "   - Salary range\n\n"
                "3. [Third Role]\n"
                "   - Why to consider this\n"
                "   - What you would do\n\n"
                "GROWTH OPPORTUNITIES\n"
                "Role | Additional Skills Needed | Time to Learn table\n"
                "Show 3-4 growth roles with specific skills\n\n"
                "UPSKILLING RECOMMENDATIONS\n"
                "[check] Most important skill to learn next with free resource\n"
                "[check] Second skill with course or platform name\n"
                "[check] Certification worth getting\n"
                "[check] Project idea to build for portfolio\n"
                "[check] Community to join for networking\n\n"
                "MOST IN-DEMAND RIGHT NOW\n"
                "Which role has highest demand and why\n"
                "2-3 sentences with market context\n\n"
                "QUICK TIP\n"
                "One unique piece of advice\n\n"
                "Available roles: Software Developer, Front End Developer, "
                "Network Administrator, Web Developer, Project Manager, "
                "Database Administrator, Security Analyst, Systems Administrator, "
                "Python Developer, Java Developer"
                + quality_rules + role_rules)

    def help_text(self):
        return ("Welcome to CareerBuddy!\n\nSmarter careers start here.\n\n"
            "WHAT I CAN DO\n\n"
            "[check] Find your ideal IT role based on your skills\n"
            "[check] Compare career paths side by side\n"
            "[check] Score and review your resume\n"
            "[check] Estimate salary for any role and country\n"
            "[check] Create a personalized career roadmap\n"
            "[check] Identify skill gaps with learning resources\n"
            "[check] Assess your hiring chances\n\n"
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
*{margin:0;padding:0;box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;background:#F3F6FB;color:#1a202c;min-height:100vh}.app{max-width:800px;margin:0 auto;padding:16px}.header{text-align:center;padding:24px 0 16px}.logo-row{display:flex;align-items:center;justify-content:center;gap:12px;margin-bottom:6px}.logo-icon{width:44px;height:44px;background:#0A66C2;border-radius:12px;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 12px rgba(10,102,194,0.25)}.logo-icon svg{width:24px;height:24px}.logo-name{font-size:26px;font-weight:800;color:#0A66C2}.tagline{font-size:13px;color:#6B7B8D;font-style:italic}.tech-bar{display:flex;gap:6px;justify-content:center;margin-top:10px}.tech-tag{padding:3px 8px;border-radius:4px;font-size:10px;font-weight:600}.t1{background:#E8F4FD;color:#0A66C2}.t2{background:#E6F9F0;color:#0D9E5F}.t3{background:#FFF3E0;color:#E67E22}.t4{background:#F3E8FD;color:#8E44AD}.chat{background:#fff;border:1px solid #E2E8F0;border-radius:16px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,0.06);margin-top:12px}.chat-top{padding:12px 18px;border-bottom:1px solid #E2E8F0;display:flex;align-items:center;justify-content:space-between;background:#FAFBFC}.chat-top-left{display:flex;align-items:center;gap:10px}.ai-badge{width:32px;height:32px;background:#0A66C2;border-radius:10px;display:flex;align-items:center;justify-content:center}.ai-badge svg{width:18px;height:18px}.chat-name{font-size:14px;font-weight:700;color:#1A202C}.chat-stat{font-size:11px;color:#8B95A5;display:flex;align-items:center;gap:4px}.online-dot{width:6px;height:6px;background:#0D9E5F;border-radius:50%;animation:pulse 2s infinite}@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}.clr{background:none;border:1px solid #E2E8F0;color:#8B95A5;padding:5px 10px;border-radius:6px;font-size:11px;cursor:pointer;font-family:inherit}.clr:hover{background:#FFF0F0;color:#E53E3E}.msgs{height:480px;overflow-y:auto;padding:18px;background:#F8FAFC}.msgs::-webkit-scrollbar{width:3px}.msgs::-webkit-scrollbar-thumb{background:#CBD5E0;border-radius:10px}.welcome{display:flex;flex-direction:column;align-items:center;padding:40px 20px;text-align:center}.welcome-logo{width:64px;height:64px;background:#0A66C2;border-radius:18px;display:flex;align-items:center;justify-content:center;margin-bottom:16px;box-shadow:0 6px 20px rgba(10,102,194,0.2)}.welcome-logo svg{width:34px;height:34px}.mr{display:flex;gap:8px;margin-bottom:14px}.mr.user{flex-direction:row-reverse}.mav{width:26px;height:26px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex-shrink:0;margin-top:2px;color:#fff}.mav.bot{background:#0A66C2}.mav.user{background:#E67E22}.mbody{max-width:78%}.mt{padding:11px 15px;border-radius:14px;font-size:13px;line-height:1.8}.mr.bot .mt{background:#fff;border:1px solid #E2E8F0;border-top-left-radius:4px;color:#2D3748}.mr.user .mt{background:#0A66C2;border-top-right-radius:4px;color:#fff}.mtime{font-size:9px;color:#A0AEC0;margin-top:2px;padding:0 4px}.mr.user .mtime{text-align:right}.typ{display:none;padding:0 18px 8px}.typ-inner{display:flex;align-items:center;gap:8px}.typ-av{width:26px;height:26px;background:#0A66C2;border-radius:8px;display:flex;align-items:center;justify-content:center}.typ-av svg{width:14px;height:14px}.typ-dots{display:flex;gap:4px;background:#fff;padding:7px 14px;border-radius:14px;border:1px solid #E2E8F0}.typ-dots span{width:5px;height:5px;border-radius:50%;animation:db 1.4s infinite}.typ-dots span:nth-child(1){background:#0A66C2}.typ-dots span:nth-child(2){background:#0D9E5F;animation-delay:.15s}.typ-dots span:nth-child(3){background:#E67E22;animation-delay:.3s}@keyframes db{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-3px)}}.ibar{padding:12px 16px;border-top:1px solid #E2E8F0;display:flex;gap:8px;background:#FAFBFC}.ifield{flex:1;padding:11px 16px;border-radius:10px;border:1px solid #E2E8F0;background:#fff;color:#2D3748;font-size:13px;font-family:inherit}.ifield:focus{outline:none;border-color:#0A66C2;box-shadow:0 0 0 3px rgba(10,102,194,0.1)}.ifield::placeholder{color:#A0AEC0}.sbtn{width:40px;height:40px;border-radius:10px;border:none;background:#0A66C2;color:#fff;font-size:15px;cursor:pointer;display:flex;align-items:center;justify-content:center;box-shadow:0 2px 8px rgba(10,102,194,0.2)}.sbtn:hover{background:#084B8A}.foot{text-align:center;padding:8px 0 14px;font-size:10px;color:#A0AEC0}.foot b{color:#0A66C2}@media(max-width:600px){.msgs{height:380px}.mbody{max-width:88%}}
</style>
</head>
<body>
<div class="app">
<div class="header"><div class="logo-row"><div class="logo-icon"><svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg></div><span class="logo-name">CareerBuddy</span></div><div class="tagline">Smarter careers start here</div><div class="tech-bar"><span class="tech-tag t1">LangChain</span><span class="tech-tag t2">FAISS</span><span class="tech-tag t3">Groq</span><span class="tech-tag t4">29K+ Resumes</span></div></div>
<div class="chat"><div class="chat-top"><div class="chat-top-left"><div class="ai-badge"><svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg></div><div><div class="chat-name">CareerBuddy</div><div class="chat-stat"><span class="online-dot"></span> Ready to help</div></div></div><button class="clr" id="clearBtn">Clear Chat</button></div>
<div class="msgs" id="messages"><div class="welcome" id="welcome"><div class="welcome-logo"><svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg></div><h2 style="font-size:20px;color:#1A202C;margin-bottom:4px">Hey there! I'm CareerBuddy</h2><p style="font-size:14px;color:#0A66C2;font-style:italic;margin-bottom:8px">Smarter careers start here</p><p style="font-size:13px;color:#8B95A5;line-height:1.6;max-width:380px">I analyze thousands of IT resumes to help you find the right role, bridge skill gaps, and plan your career growth.</p><div style="margin-top:20px;font-size:12px;color:#A0AEC0;background:#F0F4F8;padding:10px 16px;border-radius:8px;max-width:400px">Try: "Which role suits me if I know Python and SQL?" or "What salary can I expect in Ireland?"</div></div></div>
<div class="typ" id="typing"><div class="typ-inner"><div class="typ-av"><svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg></div><div class="typ-dots"><span></span><span></span><span></span></div></div></div>
<div class="ibar"><input type="text" class="ifield" id="inputField" placeholder="Ask me anything about your career..." autofocus><button class="sbtn" id="sendBtn">&#10148;</button></div></div>
<div class="foot">Powered by <b>CareerBuddy</b></div>
</div>
<script>
var inp=document.getElementById("inputField");var msgs=document.getElementById("messages");var typ=document.getElementById("typing");
document.getElementById("sendBtn").addEventListener("click",doSend);inp.addEventListener("keydown",function(e){if(e.key==="Enter"){e.preventDefault();doSend()}});document.getElementById("clearBtn").addEventListener("click",function(){msgs.innerHTML="<div class='welcome' id='welcome'><div class='welcome-logo'><svg viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='10'/><polygon points='16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76'/></svg></div><h2 style='font-size:20px;color:#1A202C'>Fresh start!</h2><p style='font-size:13px;color:#8B95A5'>Ask me anything about your career.</p></div>";fetch("/reset",{method:"POST"})});
function doSend(){var msg=inp.value.trim();if(!msg)return;inp.value="";var w=document.getElementById("welcome");if(w)w.remove();addMsg(msg,"user");typ.style.display="block";sb();fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:msg})}).then(function(r){return r.json()}).then(function(d){typ.style.display="none";addMsg(d.response,"bot")}).catch(function(e){typ.style.display="none";addMsg("Connection error.","bot")})}
function fmt(text){var lines=text.split("\n");var html="";for(var i=0;i<lines.length;i++){var line=lines[i];var t=line.trim();if(!t){html+="<div style='height:6px'></div>";continue}if(t.includes("|")&&t.split("|").length>=3){var cells=t.split("|").map(function(c){return c.trim()}).filter(function(c){return c});if(cells.every(function(c){return/^[-\s:]+$/.test(c)}))continue;html+="<div style='display:flex;gap:0;margin:1px 0'>";cells.forEach(function(cell){html+="<div style='flex:1;padding:5px 8px;background:"+(i%2===0?"#E8F4FD":"#fff")+";border:1px solid #E2E8F0;font-size:12px;color:#4A5568'>"+esc(cell)+"</div>"});html+="</div>";continue}if(t.length>3&&t.length<80&&t===t.toUpperCase()&&t.match(/[A-Z]{3,}/)){html+="<div style='font-size:12px;font-weight:700;color:#0A66C2;margin:14px 0 5px;letter-spacing:0.3px;border-bottom:2px solid #E8F4FD;padding-bottom:3px'>"+esc(t)+"</div>";continue}if(t.startsWith("[check]")||t.startsWith("\u2705")){var c=t.replace(/^\[check\]\s*/,"").replace(/^\u2705\s*/,"");html+="<div style='margin:3px 0 3px 8px;color:#2D3748;font-size:13px;display:flex;gap:7px'><span style='color:#0D9E5F;flex-shrink:0'>\u2705</span><span>"+esc(c)+"</span></div>";continue}if(t.startsWith("[arrow]")||t.startsWith("\u27A1")){var c=t.replace(/^\[arrow\]\s*/,"").replace(/^\u27A1\uFE0F?\s*/,"");html+="<div style='margin:3px 0 3px 8px;color:#4A5568;font-size:13px;display:flex;gap:7px'><span style='color:#E67E22;flex-shrink:0'>\u27A1\uFE0F</span><span>"+esc(c)+"</span></div>";continue}if(t.startsWith("[up]")){var c=t.replace(/^\[up\]\s*/,"");html+="<div style='margin:3px 0 3px 8px;color:#0D9E5F;font-size:13px;display:flex;gap:7px'><span>\uD83D\uDCC8</span><span>"+esc(c)+"</span></div>";continue}if(t.startsWith("[down]")){var c=t.replace(/^\[down\]\s*/,"");html+="<div style='margin:3px 0 3px 8px;color:#E53E3E;font-size:13px;display:flex;gap:7px'><span>\uD83D\uDCC9</span><span>"+esc(c)+"</span></div>";continue}if(t.endsWith(":")&&t.length<80&&!t.startsWith("-")&&!t.match(/^\d/)){html+="<div style='font-size:13px;font-weight:600;color:#1A202C;margin:8px 0 3px'>"+esc(t)+"</div>";continue}if(t.match(/^[-*\u2022]\s/)){var c=t.replace(/^[-*\u2022]\s+/,"");var ci=c.indexOf(":");if(ci>0&&ci<40){html+="<div style='margin:2px 0 2px 14px;color:#4A5568;font-size:13px'>\u2022 <span style='color:#0A66C2;font-weight:600'>"+esc(c.substring(0,ci))+":</span> "+esc(c.substring(ci+1).trim())+"</div>"}else{html+="<div style='margin:2px 0 2px 14px;color:#4A5568;font-size:13px'>\u2022 "+esc(c)+"</div>"}continue}if(t.match(/^\d+[.)]\s/)){var m=t.match(/^(\d+)[.)]\s+(.*)/);if(m){var c=m[2];var ci=c.indexOf(":");if(ci>0&&ci<40){html+="<div style='margin:3px 0 3px 4px;color:#4A5568;font-size:13px;display:flex;gap:7px'><span style='color:#0A66C2;font-weight:700;flex-shrink:0;min-width:16px'>"+m[1]+".</span><span><span style='color:#1A202C;font-weight:600'>"+esc(c.substring(0,ci))+":</span> "+esc(c.substring(ci+1).trim())+"</span></div>"}else{html+="<div style='margin:3px 0 3px 4px;color:#4A5568;font-size:13px;display:flex;gap:7px'><span style='color:#0A66C2;font-weight:700;flex-shrink:0;min-width:16px'>"+m[1]+".</span><span>"+esc(c)+"</span></div>"}}continue}if(line.match(/^\s{2,}/)&&t.length>0){var ci=t.indexOf(":");if(ci>0&&ci<30){html+="<div style='margin:1px 0 1px 26px;font-size:12px;color:#6B7B8D'><span style='color:#4A5568;font-weight:500'>"+esc(t.substring(0,ci))+":</span> "+esc(t.substring(ci+1).trim())+"</div>"}else{html+="<div style='margin:1px 0 1px 26px;font-size:12px;color:#6B7B8D'>"+esc(t)+"</div>"}continue}if(t.match(/\$[\d,]+/)){html+="<div style='margin:2px 0;color:#0D9E5F;font-weight:600;font-size:13px'>"+esc(t)+"</div>";continue}html+="<div style='margin:2px 0;color:#4A5568;line-height:1.7;font-size:13px'>"+esc(t)+"</div>"}return html}
function esc(t){return t.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}
function addMsg(text,type){var row=document.createElement("div");row.className="mr "+type;var av=document.createElement("div");av.className="mav "+type;if(type==="bot"){av.innerHTML="<svg viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round' width='14' height='14'><circle cx='12' cy='12' r='10'/><polygon points='16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76'/></svg>"}else{av.textContent="You"}var body=document.createElement("div");body.className="mbody";var bub=document.createElement("div");bub.className="mt";if(type==="bot"){bub.innerHTML=fmt(text)}else{bub.textContent=text}var tm=document.createElement("div");tm.className="mtime";var now=new Date();tm.textContent=(type==="bot"?"CareerBuddy":"You")+" \u00b7 "+now.getHours().toString().padStart(2,"0")+":"+now.getMinutes().toString().padStart(2,"0");body.appendChild(bub);body.appendChild(tm);row.appendChild(av);row.appendChild(body);msgs.appendChild(row);sb()}
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