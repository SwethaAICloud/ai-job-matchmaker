const API = "http://127.0.0.1:9090";

export async function sendMessage(message) {
    try {
        const r = await fetch(API + "/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: message })
        });
        const data = await r.json();
        return data.response;
    } catch (e) {
        return "Connection error. Is the backend running?";
    }
}

export async function resetChat() {
    try {
        await fetch(API + "/reset", { method: "POST" });
    } catch (e) {
        console.log("Reset failed");
    }
}