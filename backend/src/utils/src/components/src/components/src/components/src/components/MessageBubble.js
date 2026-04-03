import React, { useState } from "react";
import { useTheme } from "./ThemeContext";

function MessageBubble({ message, type, timestamp }) {
    const { dark } = useTheme();
    const [liked, setLiked] = useState(null);
    const [copied, setCopied] = useState(false);
    const [showComment, setShowComment] = useState(false);
    const [comment, setComment] = useState("");
    const [savedComment, setSavedComment] = useState("");

    const handleCopy = () => {
        navigator.clipboard.writeText(message);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const handleComment = () => {
        setSavedComment(comment);
        setComment("");
        setShowComment(false);
    };

    const formatMarkdown = (text) => {
        let h = text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");

        h = h.replace(/^### \*\*(.*?)\*\*/gm, "<h3>$1</h3>");
        h = h.replace(/^### (.*)/gm, "<h3>$1</h3>");
        h = h.replace(/^## (.*)/gm, "<h2>$1</h2>");
        h = h.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        h = h.replace(/`(.*?)`/g, "<code>$1</code>");
        h = h.replace(/^[\-\*] (.*)/gm, "<li>$1</li>");
        h = h.replace(/^\d+\. (.*)/gm, "<li>$1</li>");
        h = h.replace(/(<li>[\s\S]*?<\/li>\n?)+/g, (m) => "<ul>" + m + "</ul>");
        h = h.replace(/^-{3,}$/gm, "<hr>");
        h = h.replace(/\n\n+/g, "</p><p>");
        h = h.replace(/\n/g, "<br>");
        if (!h.startsWith("<h") && !h.startsWith("<ul")) h = "<p>" + h + "</p>";
        h = h.replace(/<p><\/p>/g, "");
        h = h.replace(/<p><h/g, "<h");
        h = h.replace(/<\/h(\d)><\/p>/g, "</h$1>");
        h = h.replace(/<p><ul>/g, "<ul>");
        h = h.replace(/<\/ul><\/p>/g, "</ul>");

        return h;
    };

    return (
        <div className={`msg-row ${type} ${dark ? "dark" : "light"}`}>
            <div className={`msg-avatar ${type}`}>
                {type === "user" ? "U" : "AI"}
            </div>

            <div className="msg-body">
                <div className={`msg-bubble ${type}`}>
                    {type === "bot" ? (
                        <div dangerouslySetInnerHTML={{ __html: formatMarkdown(message) }} />
                    ) : (
                        message
                    )}
                </div>

                {/* Actions (only for bot messages) */}
                {type === "bot" && (
                    <div className="msg-actions">
                        <button
                            className={`action-btn ${copied ? "active" : ""}`}
                            onClick={handleCopy}
                            title="Copy"
                        >
                            {copied ? "✓ Copied" : "⎘ Copy"}
                        </button>

                        <button
                            className={`action-btn ${liked === true ? "active liked" : ""}`}
                            onClick={() => setLiked(liked === true ? null : true)}
                            title="Like"
                        >
                            {liked === true ? "♥" : "♡"} Like
                        </button>

                        <button
                            className={`action-btn ${liked === false ? "active disliked" : ""}`}
                            onClick={() => setLiked(liked === false ? null : false)}
                            title="Dislike"
                        >
                            {liked === false ? "✗" : "↓"} Dislike
                        </button>

                        <button
                            className="action-btn"
                            onClick={() => setShowComment(!showComment)}
                            title="Comment"
                        >
                            💬 Comment
                        </button>
                    </div>
                )}

                {/* Comment box */}
                {showComment && (
                    <div className="comment-box">
                        <input
                            type="text"
                            placeholder="Add a comment..."
                            value={comment}
                            onChange={(e) => setComment(e.target.value)}
                            className="comment-input"
                            onKeyDown={(e) => { if (e.key === "Enter") handleComment(); }}
                        />
                        <button className="comment-submit" onClick={handleComment}>
                            Send
                        </button>
                    </div>
                )}

                {savedComment && (
                    <div className="saved-comment">
                        💬 {savedComment}
                    </div>
                )}

                <div className="msg-time">{timestamp}</div>
            </div>
        </div>
    );
}

export default MessageBubble;