/**
 * Settings panel logic.
 */

function settingsComponent() {
    return {
        open: false,
        settings: {},
        models: [],
        saving: false,
        message: '',
        rawConfig: '',
        rawMessage: '',
        extensions: [],
        extensionMessage: '',
        pullModelName: '',
        pullJob: null,
        pullMessage: '',
        memoryNamespace: '',
        memoryNamespaces: [],
        memorySearch: '',
        memoryResults: [],
        memoryMessage: '',

        async init() {
            await this.loadSettings();
            await this.loadModels();
            await this.loadRawConfig();
            await this.loadExtensions();
            await this.loadMemoryList();
        },

        async loadSettings() {
            try {
                this.settings = await apiFetch('/settings');
            } catch (e) {
                console.error('Failed to load settings:', e);
            }
        },

        async loadModels() {
            try {
                const data = await apiFetch('/models');
                this.models = data.models || [];
            } catch (e) {
                console.error('Failed to load models:', e);
                this.models = [];
            }
        },

        async save() {
            this.saving = true;
            this.message = '';
            try {
                const result = await apiFetch('/settings', {
                    method: 'POST',
                    body: this.settings,
                });
                this.message = result.message || 'Settings saved!';
            } catch (e) {
                this.message = 'Failed to save settings';
            }
            this.saving = false;
        },

        async loadRawConfig() {
            try {
                const data = await apiFetch('/settings/raw');
                this.rawConfig = JSON.stringify(data.config || {}, null, 2);
                this.memoryNamespaces = data.config?.memory?.namespaces || [];
                if (!this.memoryNamespace && this.memoryNamespaces.length) {
                    this.memoryNamespace = this.memoryNamespaces[0];
                }
            } catch (e) {
                console.error('Failed to load raw config', e);
            }
        },

        async saveRawConfig() {
            this.rawMessage = '';
            try {
                const parsed = JSON.parse(this.rawConfig || '{}');
                this.memoryNamespaces = parsed?.memory?.namespaces || [];
                if (this.memoryNamespaces.length && !this.memoryNamespace) {
                    this.memoryNamespace = this.memoryNamespaces[0];
                }
                const result = await apiFetch('/settings/raw', {
                    method: 'POST',
                    body: parsed,
                });
                this.rawMessage = result.message || 'Saved. Restart to apply changes.';
                this.loadExtensions();
                this.loadMemoryList();
            } catch (e) {
                this.rawMessage = 'Invalid JSON or save failed';
            }
        },

        async loadExtensions() {
            try {
                const data = await apiFetch('/extensions');
                this.extensions = data.extensions || [];
            } catch (e) {
                console.error('Failed to load extensions', e);
            }
        },

        async toggleExtension(ext, event) {
            const enabled = event.target.checked;
            try {
                await apiFetch('/extensions', {
                    method: 'POST',
                    body: { extensions: { [ext.name]: enabled } },
                });
                ext.enabled = enabled;
                this.extensionMessage = 'Restart to apply extension changes.';
            } catch (e) {
                console.error('Failed to update extension', e);
            }
        },

        async startModelPull() {
            const model = (this.pullModelName || '').trim();
            if (!model) return;
            this.pullMessage = '';
            try {
                const result = await apiFetch('/models/pull', {
                    method: 'POST',
                    body: { model },
                });
                this.pullJob = { job_id: result.job_id, status: 'running', output: [] };
                this.pollPull(result.job_id);
            } catch (e) {
                this.pullMessage = 'Failed to start pull';
            }
        },

        async pollPull(jobId) {
            try {
                const job = await apiFetch(`/models/pull/${jobId}`);
                this.pullJob = job;
                if (job.status === 'running') {
                    setTimeout(() => this.pollPull(jobId), 2000);
                } else {
                    this.pullMessage = job.status === 'completed'
                        ? 'Model pull complete'
                        : (job.error || 'Model pull failed');
                    if (job.status === 'completed') {
                        this.pullModelName = '';
                    }
                    this.loadModels();
                }
            } catch (e) {
                this.pullMessage = 'Failed to fetch pull status';
            }
        },

        async loadMemoryList() {
            try {
                const params = new URLSearchParams();
                if (this.memoryNamespace) params.set('namespace', this.memoryNamespace);
                const data = await apiFetch(`/memory/list?${params.toString()}`);
                this.memoryResults = data.results || [];
            } catch (e) {
                console.error('Failed to load memory list', e);
            }
        },

        async searchMemory() {
            const query = (this.memorySearch || '').trim();
            if (!query) {
                this.loadMemoryList();
                return;
            }
            try {
                const params = new URLSearchParams({ q: query });
                if (this.memoryNamespace) params.set('namespace', this.memoryNamespace);
                const data = await apiFetch(`/memory/search?${params.toString()}`);
                this.memoryResults = data.results || [];
            } catch (e) {
                console.error('Failed to search memory', e);
            }
        },

        async deleteMemory(mem) {
            if (!confirm('Delete this memory?')) return;
            try {
                await apiFetch('/memory/delete', {
                    method: 'POST',
                    body: {
                        memory_id: mem.memory_id,
                        namespace: mem.namespace || this.memoryNamespace,
                    },
                });
                this.memoryMessage = 'Memory deleted';
                this.searchMemory();
            } catch (e) {
                this.memoryMessage = 'Failed to delete memory';
            }
        },

        toggle() {
            this.open = !this.open;
            if (this.open) {
                this.loadModels();
                this.loadExtensions();
                this.loadMemoryList();
            }
        }
    };
}
