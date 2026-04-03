import React, { useState, useEffect } from "react";
import { useTheme } from "./ThemeContext";

function TopBar({ onRefresh }) {
    const { dark, toggle } = useTheme();
    const [time, setTime] = useState(new Date());

    useEffect(() => {
        const timer = setInterval(() => setTime(new Date()), 1000);
        return () => clearInterval(timer);
    }, []);

    const formatDate = (d) => {
        const options = { weekday: "long", year: "numeric", month: "long", day: "numeric" };
        return d.toLocaleDateString("en-US", options);
    };

    const formatTime = (d) => {
        return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
    };

    return (
        <div className={`topbar ${dark ? "dark" : "light"}`}>
            <div className="topbar-left">
                <div className="topbar-date">{formatDate(time)}</div>
                <div className="topbar-time">{formatTime(time)}</div>
            </div>
            <div className="topbar-right">
                <button className="topbar-btn" onClick={onRefresh} title="Refresh">
                    ↻
                </button>
                <button className="topbar-btn" onClick={toggle} title={dark ? "Light mode" : "Dark mode"}>
                    {dark ? "☀" : "☾"}
                </button>
            </div>
        </div>
    );
}

export default TopBar;