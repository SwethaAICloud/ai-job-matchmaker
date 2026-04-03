const API_URL = "http://127.0.0.1:9090";

export async function sendMessage(message) {
    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message }),
        });
        const data = await response.json();
        return data.response;
    } catch (error) {
        return "Connection error. Is the backend running?";
    }
}

export async function resetChat() {
    try {
        await fetch(`${API_URL}/reset`, { method: "POST" });
    } catch (error) {
        console.error("Reset failed");
    }
}

export async function uploadResume(file) {
    try {
        const formData = new FormData();
        formData.append("file", file);
        const response = await fetch(`${API_URL}/upload`, {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        return data.response;
    } catch (error) {
        return "Upload failed. Try again.";
    }
}