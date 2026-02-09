/**
 * Main Alpine.js application for Local Ollama Agents.
 */

document.addEventListener('alpine:init', () => {
    // Global app state
    Alpine.store('app', {
        currentView: 'chat',  // 'chat' or 'settings'
    });
});
