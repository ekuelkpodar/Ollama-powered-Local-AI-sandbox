/**
 * Shared utilities for Local Ollama Agents web UI.
 */

async function apiFetch(path, options = {}) {
    const url = `/api${path}`;
    const defaults = {
        headers: { 'Content-Type': 'application/json' },
    };
    const merged = { ...defaults, ...options };
    if (merged.body && typeof merged.body === 'object' && !(merged.body instanceof FormData)) {
        merged.body = JSON.stringify(merged.body);
    }
    const resp = await fetch(url, merged);
    return resp.json();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function autoResizeTextarea(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 150) + 'px';
}
