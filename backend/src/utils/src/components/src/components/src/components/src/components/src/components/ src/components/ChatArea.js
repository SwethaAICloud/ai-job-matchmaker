import React, { useRef, useEffect } from "react";
import MessageBubble from "./MessageBubble";
import { useTheme } from "./ThemeContext";

function ChatArea({ messages, isLoading }) {
    const endRef = useRef(null);
    const { dark } = useTheme();

    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, isLoading]);

    const formatTime = (date) => {
        return new Date(date).toLocaleTimeString("en-US", {
            hour: "2-digit",
            minute: "2-digit",
        });
    };

    const suggestions = [
        "Which role for Python and SQL?",
        "3 years React — what job?",
        "Compare frontend vs backend",
        "Skills for Python Developer",
        "Salary with Java 5 years",
        "Rate my profile",
        "Fresher with Python",
        "Career path to senior dev",
    ];

    return (
        <div className={`chat-area ${dark ? "dark" : "light"}`}>
            {messages.length === 0 ? (
                <div className="welcome-screen">
                    <div className="welcome-badge">CA</div>
                    <h2>How can I help your career?</h2>
                    <p>I analyze 29,000+ real IT resumes for personalized advice.</p>

                    <div className="suggestion-grid">
                        {suggestions.map((s, i) => (
                            <button
                                key={i}
                                className="suggestion-card"
                                onClick={() => {
                                    const event = new CustomEvent("suggestion", { detail: s });
                                    window.dispatchEvent(event);
                                }}
                            >
                                {s}
                            </button>
                        ))}
                    </div>
                </div>
            ) : (
                <div className="messages-list">
                    {messages.map((msg, i) => (
                        <MessageBubble
                            key={i}
                            message={msg.text}
                            type={msg.type}
                            timestamp={formatTime(msg.time)}
                        />
                    ))}

                    {isLoading && (
                        <div className="typing-indicator">
                            <div className="typing-avatar">AI</div>
                            <div className="typing-dots">
                                <span></span>
                                <span></span>
                                <span></span>
                            </div>
                        </div>
                    )}

                    <div ref={endRef} />
                </div>
            )}
        </div>
    );
}

export default ChatArea;