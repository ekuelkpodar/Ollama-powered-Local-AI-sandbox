/**
 * Chat component logic for Alpine.js.
 */

function chatComponent() {
    return {
        sessionId: null,
        messages: [],
        input: '',
        isProcessing: false,
        streamingContent: '',
        ollamaStatus: 'checking',

        async init() {
            await this.checkOllama();
        },

        async checkOllama() {
            try {
                const data = await apiFetch('/models');
                this.ollamaStatus = data.error ? 'offline' : 'online';
            } catch {
                this.ollamaStatus = 'offline';
            }
        },

        async send() {
            const msg = this.input.trim();
            if (!msg || this.isProcessing) return;

            this.input = '';
            this.messages.push({ role: 'user', content: msg });
            this.isProcessing = true;
            this.streamingContent = '';

            try {
                const result = await apiFetch('/chat/send', {
                    method: 'POST',
                    body: { message: msg, session_id: this.sessionId },
                });

                this.sessionId = result.session_id;

                // Connect to SSE stream
                await this.streamResponse(result.session_id);
            } catch (e) {
                this.messages.push({
                    role: 'system',
                    content: `Error: ${e.message}`,
                });
                this.isProcessing = false;
            }
        },

        async streamResponse(sessionId) {
            return new Promise((resolve) => {
                const eventSource = new EventSource(`/api/chat/stream/${sessionId}`);
                let currentContent = '';

                eventSource.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);

                        if (data.type === 'chunk') {
                            currentContent += data.content;
                            this.streamingContent = currentContent;
                            this.scrollToBottom();
                        } else if (data.type === 'done') {
                            eventSource.close();
                            this.messages.push({
                                role: 'assistant',
                                content: data.content,
                                reasoning: currentContent,
                            });
                            this.streamingContent = '';
                            this.isProcessing = false;
                            this.scrollToBottom();
                            resolve();
                        } else if (data.type === 'error') {
                            eventSource.close();
                            this.messages.push({
                                role: 'system',
                                content: `Error: ${data.content}`,
                            });
                            this.streamingContent = '';
                            this.isProcessing = false;
                            resolve();
                        }
                    } catch (e) {
                        console.error('SSE parse error:', e);
                    }
                };

                eventSource.onerror = () => {
                    eventSource.close();
                    if (this.isProcessing) {
                        this.messages.push({
                            role: 'system',
                            content: 'Connection lost. Please try again.',
                        });
                        this.streamingContent = '';
                        this.isProcessing = false;
                    }
                    resolve();
                };
            });
        },

        newSession() {
            this.sessionId = null;
            this.messages = [];
            this.streamingContent = '';
            this.isProcessing = false;
        },

        scrollToBottom() {
            this.$nextTick(() => {
                const container = document.querySelector('.chat-messages');
                if (container) {
                    container.scrollTop = container.scrollHeight;
                }
            });
        },

        handleKeydown(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                this.send();
            }
        },
    };
}
