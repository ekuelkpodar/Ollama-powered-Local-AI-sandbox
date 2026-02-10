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
                window.dispatchEvent(new CustomEvent('session-activated', {
                    detail: { session_id: this.sessionId },
                }));
                window.dispatchEvent(new Event('sessions-refresh'));

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
                            this.refreshHistory();
                            window.dispatchEvent(new Event('sessions-refresh'));
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
                    this.refreshHistory();
                    window.dispatchEvent(new Event('sessions-refresh'));
                    resolve();
                };
            });
        },

        newSession() {
            this.sessionId = null;
            this.messages = [];
            this.streamingContent = '';
            this.isProcessing = false;
            window.dispatchEvent(new Event('session-cleared'));
            window.dispatchEvent(new Event('sessions-refresh'));
        },

        async loadSession(sessionId) {
            try {
                const data = await apiFetch(`/chat/history/${sessionId}`);
                this.sessionId = sessionId;
                this.messages = this.buildTimeline(data.history || [], data.tool_calls || []);
                this.streamingContent = '';
                this.isProcessing = false;
                this.scrollToBottom();
                this.highlightBlocks();
                window.dispatchEvent(new CustomEvent('session-activated', {
                    detail: { session_id: sessionId },
                }));
            } catch (e) {
                this.messages.push({
                    role: 'system',
                    content: `Error: ${e.message}`,
                });
            }
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

        async refreshHistory() {
            if (!this.sessionId) return;
            try {
                const data = await apiFetch(`/chat/history/${this.sessionId}`);
                this.messages = this.buildTimeline(data.history || [], data.tool_calls || []);
                this.highlightBlocks();
            } catch (e) {
                console.error('Failed to refresh history', e);
            }
        },

        buildTimeline(history, toolCalls) {
            const hasToolCalls = toolCalls && toolCalls.length > 0;
            const messages = (history || [])
                .filter((m) => {
                    if (!hasToolCalls) return true;
                    if (m.role !== 'system') return true;
                    return !this.isToolResultMessage(m.content || '');
                })
                .map((m, idx) => ({
                    role: m.role,
                    content: m.content,
                    reasoning: m.reasoning,
                    created_at: m.created_at || `msg-${idx}`,
                }));

            const tools = (toolCalls || []).map((t, idx) => this.buildToolMessage(t, idx));
            const merged = [...messages, ...tools];

            merged.sort((a, b) => {
                const ta = this.parseTimestamp(a.created_at);
                const tb = this.parseTimestamp(b.created_at);
                if (ta && tb) return ta - tb;
                if (ta && !tb) return -1;
                if (!ta && tb) return 1;
                return 0;
            });

            return merged;
        },

        buildToolMessage(toolCall, index) {
            const toolName = toolCall.tool_name || 'tool';
            const args = toolCall.args || {};
            const result = toolCall.result || '';
            const createdAt = toolCall.created_at || `tool-${index}`;
            const runtime = args.runtime || '';
            const codeLang = runtime === 'shell' ? 'bash' : runtime === 'node' ? 'javascript' : 'python';
            const delegation = this.parseDelegation(result);
            const memoryResults = this.parseMemoryResults(result);
            const status = this.toolStatus(result);

            return {
                role: 'tool',
                tool_name: toolName,
                tool_args: args,
                tool_result: result,
                created_at: createdAt,
                tool_label: delegation ? `Delegated to Agent ${delegation.agentId}` : `Tool: ${toolName}`,
                tool_subtitle: delegation ? delegation.summary : '',
                args_pretty: this.prettyJson(args),
                result_pretty: result,
                code_language: codeLang,
                result_language: toolName === 'code_execution' ? codeLang : 'plaintext',
                memory_results: memoryResults,
                is_code: toolName === 'code_execution',
                tool_status: status,
                tool_status_label: status === 'error' ? 'Error' : status === 'warning' ? 'Warning' : 'Success',
            };
        },

        parseTimestamp(value) {
            if (!value) return null;
            if (value.startsWith('msg-') || value.startsWith('tool-')) return null;
            const normalized = value.replace(' UTC', 'Z');
            const date = new Date(normalized);
            if (Number.isNaN(date.getTime())) return null;
            return date.getTime();
        },

        isToolResultMessage(content) {
            return content.startsWith('[Tool ') && content.includes(' result]');
        },

        prettyJson(obj) {
            try {
                return JSON.stringify(obj, null, 2);
            } catch {
                return String(obj);
            }
        },

        parseDelegation(result) {
            const match = result.match(/Subordinate Agent (\\d+) result:/i);
            if (!match) return null;
            const agentId = match[1];
            const summary = result.split('\n').slice(1).join('\n').trim();
            return { agentId, summary };
        },

        parseMemoryResults(result) {
            if (!result || !result.startsWith('Memories found:')) return [];
            const lines = result.split('\\n').slice(1);
            const parsed = [];
            const regex = /^- \\[(.+?)\\/(.+?)\\] \\(score: ([0-9.]+)(?:, importance: ([0-9.]+))?\\) (.+)$/;
            for (const line of lines) {
                const match = line.match(regex);
                if (!match) continue;
                parsed.push({
                    namespace: match[1],
                    area: match[2],
                    score: match[3],
                    importance: match[4],
                    content: match[5],
                });
            }
            return parsed;
        },

        toolStatus(result) {
            if (!result) return 'ok';
            const lower = result.toLowerCase();
            if (lower.includes('timed out')) return 'warning';
            if (lower.startsWith('[error') || lower.includes(' error') || lower.includes('failed')) {
                return 'error';
            }
            return 'ok';
        },

        highlightBlocks() {
            this.$nextTick(() => {
                if (!window.hljs) return;
                document.querySelectorAll('pre code').forEach((block) => {
                    window.hljs.highlightElement(block);
                });
            });
        },
    };
}
