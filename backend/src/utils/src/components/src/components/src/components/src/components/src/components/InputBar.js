import React, { useState, useRef } from "react";
import { useTheme } from "./ThemeContext";

function InputBar({ onSend, onUpload, isLoading }) {
    const [text, setText] = useState("");
    const [isRecording, setIsRecording] = useState(false);
    const fileRef = useRef(null);
    const { dark } = useTheme();

    const handleSend = () => {
        if (!text.trim() || isLoading) return;
        onSend(text.trim());
        setText("");
    };

    const handleKeyDown = (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const validTypes = [
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ];

        if (!validTypes.includes(file.type)) {
            alert("Please upload a PDF or DOC/DOCX file.");
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            alert("File too large. Max 10MB.");
            return;
        }

        onUpload(file);
        e.target.value = "";
    };

    const handleVoice = () => {
        if (!("webkitSpeechRecognition" in window || "SpeechRecognition" in window)) {
            alert("Voice input not supported in this browser. Try Chrome.");
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        recognition.lang = "en-US";
        recognition.continuous = false;

        if (isRecording) {
            recognition.stop();
            setIsRecording(false);
            return;
        }

        setIsRecording(true);

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            setText((prev) => prev + " " + transcript);
            setIsRecording(false);
        };

        recognition.onerror = () => {
            setIsRecording(false);
        };

        recognition.onend = () => {
            setIsRecording(false);
        };

        recognition.start();
    };

    return (
        <div className={`input-bar ${dark ? "dark" : "light"}`}>
            <div className="input-row">
                {/* Upload button */}
                <button
                    className="input-action-btn"
                    onClick={() => fileRef.current.click()}
                    title="Upload Resume (PDF/DOC)"
                >
                    📎
                </button>
                <input
                    type="file"
                    ref={fileRef}
                    style={{ display: "none" }}
                    accept=".pdf,.doc,.docx"
                    onChange={handleFileChange}
                />

                {/* Voice button */}
                <button
                    className={`input-action-btn ${isRecording ? "recording" : ""}`}
                    onClick={handleVoice}
                    title={isRecording ? "Stop recording" : "Voice input"}
                >
                    {isRecording ? "⏹" : "🎤"}
                </button>

                {/* Text input */}
                <textarea
                    className="text-input"
                    placeholder="Ask about jobs, skills, career..."
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    onKeyDown={handleKeyDown}
                    rows={1}
                    disabled={isLoading}
                />

                {/* Send button */}
                <button
                    className={`send-btn ${isLoading ? "loading" : ""}`}
                    onClick={handleSend}
                    disabled={isLoading || !text.trim()}
                >
                    {isLoading ? "..." : "➤"}
                </button>
            </div>

            <div className="input-info">
                <span>PDF/DOC upload supported</span>
                <span>•</span>
                <span>Voice input available</span>
                <span>•</span>
                <span>Press Enter to send</span>
            </div>
        </div>
    );
}

export default InputBar;