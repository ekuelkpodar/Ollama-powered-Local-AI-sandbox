/**
 * Sessions sidebar component.
 */

function sessionsComponent() {
    return {
        sessions: [],
        search: '',
        activeSessionId: null,
        editingId: null,
        editTitle: '',

        async init() {
            await this.refresh();
            window.addEventListener('session-activated', (e) => {
                this.activeSessionId = e.detail.session_id;
                this.refresh();
            });
            window.addEventListener('session-cleared', () => {
                this.activeSessionId = null;
                this.refresh();
            });
            window.addEventListener('sessions-refresh', () => {
                this.refresh();
            });
        },

        get filteredSessions() {
            if (!this.search.trim()) return this.sessions;
            const q = this.search.toLowerCase();
            return this.sessions.filter((s) => {
                const title = (s.title || '').toLowerCase();
                const id = (s.session_id || '').toLowerCase();
                return title.includes(q) || id.includes(q);
            });
        },

        async refresh() {
            try {
                const data = await apiFetch('/chat/sessions');
                this.sessions = data.sessions || [];
            } catch (e) {
                console.error('Failed to fetch sessions', e);
            }
        },

        openSession(session) {
            this.activeSessionId = session.session_id;
            window.dispatchEvent(new CustomEvent('load-session', {
                detail: { session_id: session.session_id },
            }));
        },

        startRename(session) {
            this.editingId = session.session_id;
            this.editTitle = session.title || '';
        },

        cancelRename() {
            this.editingId = null;
            this.editTitle = '';
        },

        async saveRename(session) {
            const title = this.editTitle.trim();
            if (!title) return;
            try {
                await apiFetch(`/chat/session/${session.session_id}`, {
                    method: 'PATCH',
                    body: { title },
                });
                this.editingId = null;
                this.editTitle = '';
                await this.refresh();
            } catch (e) {
                console.error('Rename failed', e);
            }
        },

        async deleteSession(session) {
            if (!confirm(`Delete session "${session.title || session.session_id}"?`)) {
                return;
            }
            try {
                await apiFetch(`/chat/session/${session.session_id}`, { method: 'DELETE' });
                if (this.activeSessionId === session.session_id) {
                    window.dispatchEvent(new Event('new-session'));
                }
                await this.refresh();
            } catch (e) {
                console.error('Delete failed', e);
            }
        },

        formatUpdated(session) {
            const stamp = session.updated_at || session.created_at;
            if (!stamp) return 'Active';
            const date = new Date(stamp.replace(' UTC', 'Z'));
            if (Number.isNaN(date.getTime())) return stamp;
            return date.toLocaleDateString();
        },

        sessionLabel(session) {
            return session.title || 'New chat';
        },
    };
}
