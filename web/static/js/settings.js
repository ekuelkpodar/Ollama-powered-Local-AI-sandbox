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

        async init() {
            await this.loadSettings();
            await this.loadModels();
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

        toggle() {
            this.open = !this.open;
            if (this.open) {
                this.loadModels();
            }
        }
    };
}
