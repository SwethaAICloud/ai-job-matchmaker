import React, { useState, useEffect } from "react";
import Sidebar from "./components/Sidebar";
import TopBar from "./components/TopBar";
import ChatArea from "./components/ChatArea";
import InputBar from "./components/InputBar";
import { ThemeProvider } from "./components/ThemeContext";
import { sendMessage, resetChat } from "./utils/api";
import "./App.css";

function App() {
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const [sessions, setSessions] = useState([
        { id: "1", title: "New Chat", date: new Date().toLocaleDateString(), messages: [] },
    ]);
    const [activeSession, setActiveSession] = useState("1");
    const [isLoading, setIsLoading] = useState(false);

    const currentSession = sessions.find((s) => s.id === activeSession);
    const messages = currentSession ? currentSession.messages : [];

    useEffect(() => {
        const handleSuggestion = (e) => {
            handleSend(e.detail);
        };
        window.addEventListener("suggestion", handleSuggestion);
        return () => window.removeEventListener("suggestion", handleSuggestion);
    }, [activeSession]);

    const handleSend = async (text) => {
        const userMsg = { type: "user", text, time: new Date() };

        setSessions((prev) =>
            prev.map((s) => {
                if (s.id === activeSession) {
                    const newTitle = s.messages.length === 0 ? text.slice(0, 40) : s.title;
                    return { ...s, title: newTitle, messages: [...s.messages, userMsg] };
                }
                return s;
            })
        );

        setIsLoading(true);

        const response = await sendMessage(text);

        const botMsg = { type: "bot", text: response, time: new Date() };

        setSessions((prev) =>
            prev.map((s) => {
                if (s.id === activeSession) {
                    return { ...s, messages: [...s.messages, botMsg] };
                }
                return s;
            })
        );

        setIsLoading(false);
    };

    const handleUpload = async (file) => {
        const text = "I uploaded my resume: " + file.name + ". Please analyze it.";
        handleSend(text);
    };

    const handleNewChat = () => {
        const id = Date.now().toString();
        setSessions((prev) => [
            { id, title: "New Chat", date: new Date().toLocaleDateString(), messages: [] },
            ...prev,
        ]);
        setActiveSession(id);
        resetChat();
    };

    const handleDeleteSession = (id) => {
        setSessions((prev) => prev.filter((s) => s.id !== id));
        if (activeSession === id) {
            const remaining = sessions.filter((s) => s.id !== id);
            if (remaining.length > 0) {
                setActiveSession(remaining[0].id);
            } else {
                handleNewChat();
            }
        }
    };

    const handleRefresh = () => {
        resetChat();
        setSessions((prev) =>
            prev.map((s) => {
                if (s.id === activeSession) {
                    return { ...s, messages: [] };
                }
                return s;
            })
        );
    };

    return (
        <ThemeProvider>
            <div className="app-layout">
                <Sidebar
                    sessions={sessions}
                    activeSession={activeSession}
                    onSelectSession={setActiveSession}
                    onNewChat={handleNewChat}
                    onDeleteSession={handleDeleteSession}
                    sidebarOpen={sidebarOpen}
                    setSidebarOpen={setSidebarOpen}
                />

                <div className="main-panel">
                    <TopBar onRefresh={handleRefresh} />
                    <ChatArea messages={messages} isLoading={isLoading} />
                    <InputBar
                        onSend={handleSend}
                        onUpload={handleUpload}
                        isLoading={isLoading}
                    />
                </div>
            </div>
        </ThemeProvider>
    );
}

export default App;