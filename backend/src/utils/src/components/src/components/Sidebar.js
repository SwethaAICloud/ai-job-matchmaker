import React, { useState } from "react";
import { useTheme } from "./ThemeContext";

function Sidebar({ sessions, activeSession, onSelectSession, onNewChat, onDeleteSession, sidebarOpen, setSidebarOpen }) {
    const [search, setSearch] = useState("");
    const { dark } = useTheme();

    const filtered = sessions.filter((s) =>
        s.title.toLowerCase().includes(search.toLowerCase())
    );

    if (!sidebarOpen) {
        return (
            <div className={`sidebar-collapsed ${dark ? "dark" : "light"}`}>
                <button className="sidebar-toggle" onClick={() => setSidebarOpen(true)} title="Open sidebar">
                    ☰
                </button>
            </div>
        );
    }

    return (
        <div className={`sidebar ${dark ? "dark" : "light"}`}>
            {/* Top */}
            <div className="sidebar-top">
                <div className="sidebar-header">
                    <span className="sidebar-title">CareerAI</span>
                    <button className="sidebar-toggle" onClick={() => setSidebarOpen(false)} title="Hide sidebar">
                        ✕
                    </button>
                </div>

                <button className="new-chat-btn" onClick={onNewChat}>
                    + New Chat
                </button>

                <div className="search-box">
                    <input
                        type="text"
                        placeholder="Search chats..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        className="search-input"
                    />
                </div>
            </div>

            {/* Chat list */}
            <div className="chat-list">
                {filtered.length === 0 ? (
                    <div className="no-chats">No chats found</div>
                ) : (
                    filtered.map((session) => (
                        <div
                            key={session.id}
                            className={`chat-item ${session.id === activeSession ? "active" : ""}`}
                            onClick={() => onSelectSession(session.id)}
                        >
                            <div className="chat-item-title">{session.title}</div>
                            <div className="chat-item-date">{session.date}</div>
                            <button
                                className="chat-item-delete"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    onDeleteSession(session.id);
                                }}
                                title="Delete"
                            >
                                ×
                            </button>
                        </div>
                    ))
                )}
            </div>

            {/* Bottom info */}
            <div className="sidebar-bottom">
                <div className="memory-info">
                    <div className="memory-label">Memory</div>
                    <div className="memory-bar">
                        <div className="memory-fill" style={{ width: "35%" }}></div>
                    </div>
                    <div className="memory-text">Context: 5 docs retrieved per query</div>
                </div>
                <div className="sidebar-footer">
                    <span>LangChain + FAISS + Groq</span>
                </div>
            </div>
        </div>
    );
}

export default Sidebar;