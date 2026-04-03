import React, { useState, useEffect, useRef } from "react";
import { sendMessage, resetChat } from "./utils/api";
import "./App.css";

function App() {
    const [sidebar, setSidebar] = useState(true);
    const [dark, setDark] = useState(false);
    const [sessions, setSessions] = useState([
        { id: "1", title: "New Chat", date: new Date().toLocaleDateString(), msgs: [] }
    ]);
    const [active, setActive] = useState("1");
    const [loading, setLoading] = useState(false);
    const [search, setSearch] = useState("");
    const [time, setTime] = useState(new Date());
    const [inputText, setInputText] = useState("");
    const endRef = useRef(null);
    const fileRef = useRef(null);

    useEffect(() => { const t = setInterval(() => setTime(new Date()), 1000); return () => clearInterval(t); }, []);
    useEffect(() => { document.body.className = dark ? "dark" : ""; }, [dark]);
    useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [sessions, loading]);

    const current = sessions.find(s => s.id === active);
    const msgs = current ? current.msgs : [];
    const filtered = sessions.filter(s => s.title.toLowerCase().includes(search.toLowerCase()));

    const doSend = async (text) => {
        if (!text || loading) return;
        const userMsg = { type: "user", text, time: new Date(), liked: null, comment: "", copied: false };
        setSessions(prev => prev.map(s => {
            if (s.id === active) {
                const title = s.msgs.length === 0 ? text.slice(0, 35) : s.title;
                return { ...s, title, msgs: [...s.msgs, userMsg] };
            }
            return s;
        }));
        setLoading(true);
        const response = await sendMessage(text);
        const botMsg = { type: "bot", text: response, time: new Date(), liked: null, comment: "", copied: false };
        setSessions(prev => prev.map(s => {
            if (s.id === active) return { ...s, msgs: [...s.msgs, botMsg] };
            return s;
        }));
        setLoading(false);
    };

    const newChat = () => {
        const id = Date.now().toString();
        setSessions(prev => [{ id, title: "New Chat", date: new Date().toLocaleDateString(), msgs: [] }, ...prev]);
        setActive(id);
        resetChat();
    };

    const delChat = (id) => {
        setSessions(prev => prev.filter(s => s.id !== id));
        if (active === id) {
            const rest = sessions.filter(s => s.id !== id);
            if (rest.length > 0) setActive(rest[0].id); else newChat();
        }
    };

    const refresh = () => {
        resetChat();
        setSessions(prev => prev.map(s => s.id === active ? { ...s, msgs: [] } : s));
    };

    const updateMsg = (idx, updates) => {
        setSessions(prev => prev.map(s => {
            if (s.id === active) { const m = [...s.msgs]; m[idx] = { ...m[idx], ...updates }; return { ...s, msgs: m }; }
            return s;
        }));
    };

    const copyMsg = (text, idx) => { navigator.clipboard.writeText(text); updateMsg(idx, { copied: true }); setTimeout(() => updateMsg(idx, { copied: false }), 2000); };
    const likeMsg = (idx, val) => { updateMsg(idx, { liked: msgs[idx].liked === val ? null : val }); };
    const commentMsg = (idx, comment) => { updateMsg(idx, { comment }); };

    const handleVoice = () => {
        if (!("webkitSpeechRecognition" in window || "SpeechRecognition" in window)) { alert("Voice not supported."); return; }
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        const rec = new SR(); rec.lang = "en-US";
        rec.onresult = (e) => setInputText(prev => prev + " " + e.results[0][0].transcript);
        rec.start();
    };

    const handleFile = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        doSend("I uploaded my resume: " + file.name + ". Please analyze it.");
        e.target.value = "";
    };

    const fmtTime = (d) => new Date(d).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });

    const formatText = (text) => {
        return text.split("\n").map((line, i) => {
            const t = line.trim();
            if (!t) return <div key={i} style={{ height: 6 }} />;
            if (t.includes("|") && t.split("|").length >= 3) {
                const cells = t.split("|").map(c => c.trim()).filter(c => c);
                if (cells.every(c => /^[-\s:]+$/.test(c))) return null;
                return <div key={i} style={{ display: "flex", gap: 0, margin: "1px 0" }}>{cells.map((cell, j) => <div key={j} style={{ flex: 1, padding: "5px 8px", background: i % 2 === 0 ? "#E8F4FD" : "#fff", border: "1px solid #E2E8F0", fontSize: 12, color: "#4A5568" }}>{cell}</div>)}</div>;
            }
            if (t.length > 3 && t.length < 80 && t === t.toUpperCase() && t.match(/[A-Z]{3,}/)) return <div key={i} style={{ fontSize: 12, fontWeight: 700, color: "#0A66C2", margin: "14px 0 5px", borderBottom: "2px solid #E8F4FD", paddingBottom: 3 }}>{t}</div>;
            if (t.startsWith("[check]") || t.startsWith("\u2705")) return <div key={i} style={{ margin: "3px 0 3px 8px", color: "#2D3748", fontSize: 13, display: "flex", gap: 7 }}><span style={{ color: "#0D9E5F", flexShrink: 0 }}>✅</span><span>{t.replace(/^\[check\]\s*/, "").replace(/^✅\s*/, "")}</span></div>;
            if (t.startsWith("[arrow]") || t.startsWith("\u27A1")) return <div key={i} style={{ margin: "3px 0 3px 8px", color: "#4A5568", fontSize: 13, display: "flex", gap: 7 }}><span style={{ color: "#E67E22", flexShrink: 0 }}>➡️</span><span>{t.replace(/^\[arrow\]\s*/, "").replace(/^➡️?\s*/, "")}</span></div>;
            if (t.endsWith(":") && t.length < 80 && !t.startsWith("-")) return <div key={i} style={{ fontSize: 13, fontWeight: 600, color: "#1A202C", margin: "8px 0 3px" }}>{t}</div>;
            if (t.match(/^[-*•]\s/)) { const c = t.replace(/^[-*•]\s+/, ""); const ci = c.indexOf(":"); if (ci > 0 && ci < 40) return <div key={i} style={{ margin: "2px 0 2px 14px", color: "#4A5568", fontSize: 13 }}>• <span style={{ color: "#0A66C2", fontWeight: 600 }}>{c.substring(0, ci)}:</span> {c.substring(ci + 1).trim()}</div>; return <div key={i} style={{ margin: "2px 0 2px 14px", color: "#4A5568", fontSize: 13 }}>• {c}</div>; }
            if (t.match(/^\d+[.)]\s/)) { const m = t.match(/^(\d+)[.)]\s+(.*)/); if (m) { const c = m[2]; const ci = c.indexOf(":"); if (ci > 0 && ci < 40) return <div key={i} style={{ margin: "3px 0 3px 4px", color: "#4A5568", fontSize: 13, display: "flex", gap: 7 }}><span style={{ color: "#0A66C2", fontWeight: 700, flexShrink: 0, minWidth: 16 }}>{m[1]}.</span><span><span style={{ color: "#1A202C", fontWeight: 600 }}>{c.substring(0, ci)}:</span> {c.substring(ci + 1).trim()}</span></div>; return <div key={i} style={{ margin: "3px 0 3px 4px", color: "#4A5568", fontSize: 13, display: "flex", gap: 7 }}><span style={{ color: "#0A66C2", fontWeight: 700, flexShrink: 0, minWidth: 16 }}>{m[1]}.</span><span>{c}</span></div>; } }
            if (t.match(/\$[\d,]+/)) return <div key={i} style={{ margin: "2px 0", color: "#0D9E5F", fontWeight: 600, fontSize: 13 }}>{t}</div>;
            return <div key={i} style={{ margin: "2px 0", color: "#4A5568", lineHeight: 1.7, fontSize: 13 }}>{t}</div>;
        });
    };

    return (
        <div className="app-layout">
            {sidebar ? (
                <div className="sidebar" style={{ borderRight: "1px solid #E2E8F0" }}>
                    <div className="sidebar-top">
                        <div className="sidebar-head">
                            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                <div style={{ width: 28, height: 28, background: "#0A66C2", borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", color: "white", fontSize: 11, fontWeight: 700 }}>CB</div>
                                <span style={{ fontSize: 16, fontWeight: 700, color: "#0A66C2" }}>CareerBuddy</span>
                            </div>
                            <button className="toggle-btn" onClick={() => setSidebar(false)}>✕</button>
                        </div>
                        <button className="new-btn" style={{ borderColor: "#0A66C2", color: "#0A66C2" }} onClick={newChat}>+ New Chat</button>
                        <input className="search-input" placeholder="Search chats..." value={search} onChange={e => setSearch(e.target.value)} />
                    </div>
                    <div className="chat-list">
                        {filtered.map(s => (
                            <div key={s.id} className={"chat-item" + (s.id === active ? " active" : "")} onClick={() => setActive(s.id)} style={s.id === active ? { borderLeftColor: "#0A66C2" } : {}}>
                                <div className="chat-item-title">{s.title}</div>
                                <div className="chat-item-date">{s.date}</div>
                                <button className="chat-item-del" onClick={e => { e.stopPropagation(); delChat(s.id); }}>×</button>
                            </div>
                        ))}
                    </div>
                    <div className="sidebar-bottom">
                        <div className="mem-label">Memory</div>
                        <div className="mem-bar"><div className="mem-fill" style={{ background: "#0A66C2" }}></div></div>
                        <div className="mem-text">Context: 8 docs per query</div>
                        <div className="sidebar-foot">LangChain + FAISS + Groq</div>
                    </div>
                </div>
            ) : (
                <div className="sidebar-closed"><button className="toggle-btn" onClick={() => setSidebar(true)}>☰</button></div>
            )}

            <div className="main-area">
                <div className="topbar">
                    <div>
                        <div className="topbar-date">{time.toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric" })}</div>
                        <div className="topbar-time">{time.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" })}</div>
                    </div>
                    <div className="topbar-btns">
                        <button className="topbar-btn" onClick={refresh} title="Refresh">↻</button>
                        <button className="topbar-btn" onClick={() => setDark(!dark)} title="Theme">{dark ? "☀" : "☾"}</button>
                    </div>
                </div>

                <div className="chat-area">
                    {msgs.length === 0 ? (
                        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "50px 20px", textAlign: "center" }}>
                            <div style={{ width: 56, height: 56, background: "#0A66C2", borderRadius: 16, display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 16, boxShadow: "0 6px 20px rgba(10,102,194,0.2)" }}>
                                <svg viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="30" height="30"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>
                            </div>
                            <h2 style={{ fontSize: 20, fontWeight: 700, color: "#1A202C", marginBottom: 4 }}>Hey there! I'm CareerBuddy</h2>
                            <p style={{ fontSize: 14, color: "#0A66C2", fontStyle: "italic", marginBottom: 8 }}>Smarter careers start here</p>
                            <p style={{ fontSize: 13, color: "#8B95A5", lineHeight: 1.6, maxWidth: 380 }}>I analyze thousands of IT resumes to help you find the right role, bridge skill gaps, and plan your career growth.</p>
                            <div style={{ marginTop: 20, fontSize: 12, color: "#A0AEC0", background: "#F0F4F8", padding: "10px 16px", borderRadius: 8, maxWidth: 400 }}>
                                Try: "Which role suits me if I know Python and SQL?" or "What salary can I expect in Ireland?"
                            </div>
                        </div>
                    ) : (
                        msgs.map((m, i) => (
                            <div key={i}>
                                <div className={"msg-row " + m.type}>
                                    <div className={"msg-av " + m.type} style={m.type === "bot" ? { background: "#0A66C2" } : {}}>
                                        {m.type === "user" ? "You" : <svg viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" width="14" height="14"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>}
                                    </div>
                                    <div className="msg-body">
                                        <div className="msg-bubble" style={m.type === "user" ? { background: "#0A66C2" } : {}}>{m.type === "bot" ? formatText(m.text) : m.text}</div>
                                        {m.type === "bot" && (
                                            <div className="msg-actions">
                                                <button className={"act-btn" + (m.copied ? " copied" : "")} onClick={() => copyMsg(m.text, i)}>{m.copied ? "✓ Copied" : "Copy"}</button>
                                                <button className={"act-btn" + (m.liked === true ? " liked" : "")} onClick={() => likeMsg(i, true)}>{m.liked === true ? "♥" : "♡"} Like</button>
                                                <button className={"act-btn" + (m.liked === false ? " disliked" : "")} onClick={() => likeMsg(i, false)}>↓ Dislike</button>
                                                <CommentBtn msg={m} idx={i} onComment={commentMsg} />
                                            </div>
                                        )}
                                        <div className="msg-time">{m.type === "bot" ? "CareerBuddy" : "You"} · {fmtTime(m.time)}</div>
                                    </div>
                                </div>
                            </div>
                        ))
                    )}
                    {loading && <div className="typing-row"><div className="typing-av" style={{ background: "#0A66C2" }}>CB</div><div className="typing-dots"><span style={{ background: "#0A66C2" }}></span><span style={{ background: "#0D9E5F" }}></span><span style={{ background: "#E67E22" }}></span></div></div>}
                    <div ref={endRef} />
                </div>

                <div className="input-bar">
                    <div className="input-row">
                        <button className="input-act" onClick={() => fileRef.current.click()} title="Upload Resume">📎</button>
                        <input type="file" ref={fileRef} style={{ display: "none" }} accept=".pdf,.doc,.docx" onChange={handleFile} />
                        <button className="input-act" onClick={handleVoice} title="Voice">🎤</button>
                        <textarea className="text-input" placeholder="Ask me anything about your career..." value={inputText} onChange={e => setInputText(e.target.value)} onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); doSend(inputText); setInputText(""); } }} rows={1} style={{ borderColor: inputText ? "#0A66C2" : undefined }} />
                        <button className="send-btn" onClick={() => { doSend(inputText); setInputText(""); }} disabled={loading || !inputText.trim()} style={{ background: "#0A66C2" }}>➤</button>
                    </div>
                    <div className="input-info"><span>PDF/DOC upload</span><span>·</span><span>Voice input</span><span>·</span><span>Enter to send</span></div>
                </div>
            </div>
        </div>
    );
}

function CommentBtn({ msg, idx, onComment }) {
    const [open, setOpen] = useState(false);
    const [text, setText] = useState("");
    return (
        <>
            <button className="act-btn" onClick={() => setOpen(!open)}>💬</button>
            {open && <div className="comment-box"><input className="comment-input" placeholder="Comment..." value={text} onChange={e => setText(e.target.value)} onKeyDown={e => { if (e.key === "Enter") { onComment(idx, text); setText(""); setOpen(false); } }} /><button className="comment-send" style={{ background: "#0A66C2" }} onClick={() => { onComment(idx, text); setText(""); setOpen(false); }}>Send</button></div>}
            {msg.comment && <div className="saved-comment">💬 {msg.comment}</div>}
        </>
    );
}

export default App;