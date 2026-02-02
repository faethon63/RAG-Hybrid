import { useState, useEffect } from 'react';
import { useSettingsStore, MODEL_OPTIONS, MODE_OPTIONS } from '../../stores/settingsStore';
import { CloseIcon, LoaderIcon, CheckIcon } from '../common/icons';

export function SettingsPanel() {
  const showSettings = useSettingsStore((s) => s.showSettings);
  const setShowSettings = useSettingsStore((s) => s.setShowSettings);
  const globalSettings = useSettingsStore((s) => s.globalSettings);
  const loadSettings = useSettingsStore((s) => s.loadSettings);
  const saveSettings = useSettingsStore((s) => s.saveSettings);
  const showThinking = useSettingsStore((s) => s.showThinking);
  const setShowThinking = useSettingsStore((s) => s.setShowThinking);
  const mode = useSettingsStore((s) => s.mode);
  const setMode = useSettingsStore((s) => s.setMode);
  const model = useSettingsStore((s) => s.model);
  const setModel = useSettingsStore((s) => s.setModel);

  const [defaultMode, setDefaultMode] = useState('auto');
  const [defaultModel, setDefaultModel] = useState('auto');
  const [globalInstructions, setGlobalInstructions] = useState('');
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (globalSettings) {
      setDefaultMode(globalSettings.default_mode || 'auto');
      setDefaultModel(globalSettings.default_model || 'auto');
      setGlobalInstructions(globalSettings.global_instructions || '');
    }
  }, [globalSettings]);

  useEffect(() => {
    if (showSettings && !globalSettings) {
      loadSettings();
    }
  }, [showSettings, globalSettings, loadSettings]);

  const handleSave = async () => {
    setSaving(true);
    try {
      await saveSettings({
        default_mode: defaultMode,
        default_model: defaultModel,
        global_instructions: globalInstructions,
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (err) {
      console.error('Failed to save settings:', err);
    } finally {
      setSaving(false);
    }
  };

  if (!showSettings) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50"
        onClick={() => setShowSettings(false)}
      />

      {/* Panel */}
      <div className="relative w-full max-w-lg bg-[var(--color-background)] border border-[var(--color-border)] rounded-xl shadow-2xl max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-[var(--color-background)] border-b border-[var(--color-border)] px-6 py-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Settings</h2>
          <button
            onClick={() => setShowSettings(false)}
            className="p-1 hover:bg-[var(--color-surface)] rounded transition-colors"
          >
            <CloseIcon className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Current session settings */}
          <section>
            <h3 className="text-sm font-medium text-[var(--color-text-secondary)] uppercase tracking-wide mb-3">
              Current Session
            </h3>

            <div className="space-y-4">
              {/* Mode selector */}
              <div>
                <label className="block text-sm mb-2">Mode</label>
                <select
                  value={mode}
                  onChange={(e) => setMode(e.target.value)}
                  className="w-full px-3 py-2 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg text-[var(--color-text)] focus:outline-none focus:border-[var(--color-primary)]"
                >
                  {MODE_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
                <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                  {MODE_OPTIONS.find((o) => o.value === mode)?.description}
                </p>
              </div>

              {/* Model selector */}
              <div>
                <label className="block text-sm mb-2">Model</label>
                <select
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  className="w-full px-3 py-2 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg text-[var(--color-text)] focus:outline-none focus:border-[var(--color-primary)]"
                >
                  {MODEL_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Show thinking toggle */}
              <div className="flex items-center justify-between">
                <label className="text-sm">Show thinking blocks</label>
                <button
                  onClick={() => setShowThinking(!showThinking)}
                  className={`w-10 h-6 rounded-full transition-colors ${
                    showThinking
                      ? 'bg-[var(--color-primary)]'
                      : 'bg-[var(--color-border)]'
                  }`}
                >
                  <div
                    className={`w-4 h-4 bg-white rounded-full transition-transform mx-1 ${
                      showThinking ? 'translate-x-4' : 'translate-x-0'
                    }`}
                  />
                </button>
              </div>
            </div>
          </section>

          <div className="border-t border-[var(--color-border)]" />

          {/* Default settings (saved to backend) */}
          <section>
            <h3 className="text-sm font-medium text-[var(--color-text-secondary)] uppercase tracking-wide mb-3">
              Default Settings
            </h3>

            <div className="space-y-4">
              {/* Default mode */}
              <div>
                <label className="block text-sm mb-2">Default Mode</label>
                <select
                  value={defaultMode}
                  onChange={(e) => setDefaultMode(e.target.value)}
                  className="w-full px-3 py-2 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg text-[var(--color-text)] focus:outline-none focus:border-[var(--color-primary)]"
                >
                  {MODE_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Default model */}
              <div>
                <label className="block text-sm mb-2">Default Model</label>
                <select
                  value={defaultModel}
                  onChange={(e) => setDefaultModel(e.target.value)}
                  className="w-full px-3 py-2 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg text-[var(--color-text)] focus:outline-none focus:border-[var(--color-primary)]"
                >
                  {MODEL_OPTIONS.filter((o) => o.value !== 'auto').map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Global instructions */}
              <div>
                <label className="block text-sm mb-2">Global Instructions</label>
                <textarea
                  value={globalInstructions}
                  onChange={(e) => setGlobalInstructions(e.target.value)}
                  placeholder="Instructions that apply to all queries..."
                  rows={4}
                  className="w-full px-3 py-2 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg text-[var(--color-text)] placeholder-[var(--color-text-secondary)] resize-none focus:outline-none focus:border-[var(--color-primary)]"
                />
              </div>
            </div>
          </section>
        </div>

        {/* Footer */}
        <div className="sticky bottom-0 bg-[var(--color-background)] border-t border-[var(--color-border)] px-6 py-4 flex justify-end gap-3">
          <button
            onClick={() => setShowSettings(false)}
            className="px-4 py-2 text-sm hover:bg-[var(--color-surface)] rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            className="px-4 py-2 text-sm bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] text-white rounded-lg transition-colors flex items-center gap-2"
          >
            {saving ? (
              <>
                <LoaderIcon className="w-4 h-4 animate-spin" />
                <span>Saving...</span>
              </>
            ) : saved ? (
              <>
                <CheckIcon className="w-4 h-4" />
                <span>Saved</span>
              </>
            ) : (
              <span>Save</span>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
