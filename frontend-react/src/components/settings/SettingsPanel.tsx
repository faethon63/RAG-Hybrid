import { useState, useEffect } from 'react';
import { useSettingsStore, MODEL_OPTIONS, MODE_OPTIONS, getAvailableModelOptions, getAvailableModeOptions } from '../../stores/settingsStore';
import { CloseIcon, LoaderIcon, CheckIcon, MonitorIcon, AlertIcon, RefreshIcon } from '../common/icons';
import { api } from '../../api/client';
import { isPushSupported, unsubscribeFromPush } from '../../utils/pushNotifications';

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

  const health = useSettingsStore((s) => s.health);

  const [defaultMode, setDefaultMode] = useState('auto');
  const [defaultModel, setDefaultModel] = useState('auto');
  const [globalInstructions, setGlobalInstructions] = useState('');
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  // Remote Control status
  const [rcStatus, setRcStatus] = useState<{
    pc_online: boolean;
    claude_code_running: boolean;
    hostname?: string;
    last_seen?: string;
    reason?: string;
  } | null>(null);
  const [rcLoading, setRcLoading] = useState(false);
  const [pushResetting, setPushResetting] = useState(false);
  const [pushResetDone, setPushResetDone] = useState(false);

  // Filter options based on Ollama availability
  const ollamaAvailable = health?.services?.ollama ?? false;
  const availableModelOptions = getAvailableModelOptions(ollamaAvailable);
  const availableModeOptions = getAvailableModeOptions(ollamaAvailable);

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

  const fetchRcStatus = async () => {
    setRcLoading(true);
    try {
      const status = await api.getRemoteControlStatus();
      setRcStatus(status);
    } catch {
      setRcStatus({ pc_online: false, claude_code_running: false, reason: 'Failed to check' });
    } finally {
      setRcLoading(false);
    }
  };

  useEffect(() => {
    if (showSettings) {
      fetchRcStatus();
    }
  }, [showSettings]);

  // Reset to 'auto' if currently selected option requires Ollama but it's unavailable
  useEffect(() => {
    if (!ollamaAvailable) {
      const currentModeOption = MODE_OPTIONS.find(o => o.value === mode);
      const currentModelOption = MODEL_OPTIONS.find(o => o.value === model);

      if (currentModeOption?.requiresOllama) {
        setMode('auto');
      }
      if (currentModelOption?.requiresOllama) {
        setModel('auto');
      }
    }
  }, [ollamaAvailable, mode, model, setMode, setModel]);

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
                  {availableModeOptions.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
                <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                  {availableModeOptions.find((o) => o.value === mode)?.description}
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
                  {availableModelOptions.map((opt) => (
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
                  {availableModeOptions.map((opt) => (
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
                  {availableModelOptions.filter((o) => o.value !== 'auto').map((opt) => (
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

          {/* Notifications */}
          {isPushSupported() && (
            <>
              <div className="border-t border-[var(--color-border)]" />
              <section>
                <h3 className="text-sm font-medium text-[var(--color-text-secondary)] uppercase tracking-wide mb-3">
                  Push Notifications
                </h3>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm">Reset push subscription</p>
                    <p className="text-xs text-[var(--color-text-secondary)]">
                      Clears browser subscription so you can re-register
                    </p>
                  </div>
                  <button
                    onClick={async () => {
                      setPushResetting(true);
                      await unsubscribeFromPush();
                      localStorage.removeItem('push-banner-dismissed');
                      setPushResetting(false);
                      setPushResetDone(true);
                      setTimeout(() => setPushResetDone(false), 3000);
                    }}
                    disabled={pushResetting}
                    className="px-3 py-1.5 text-sm border border-[var(--color-border)] rounded-lg hover:bg-[var(--color-surface)] transition-colors disabled:opacity-50 flex items-center gap-2"
                  >
                    {pushResetting ? (
                      <><LoaderIcon className="w-3.5 h-3.5 animate-spin" /> Resetting...</>
                    ) : pushResetDone ? (
                      <><CheckIcon className="w-3.5 h-3.5 text-green-500" /> Done — reload app</>
                    ) : (
                      'Reset'
                    )}
                  </button>
                </div>
              </section>
            </>
          )}

          <div className="border-t border-[var(--color-border)]" />

          {/* Remote Control */}
          <section>
            <h3 className="text-sm font-medium text-[var(--color-text-secondary)] uppercase tracking-wide mb-3">
              Claude Code Remote Control
            </h3>

            <div className="bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg p-4">
              {rcLoading ? (
                <div className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
                  <LoaderIcon className="w-4 h-4 animate-spin" />
                  <span>Checking PC status...</span>
                </div>
              ) : rcStatus ? (
                <div className="space-y-3">
                  {/* PC status */}
                  <div className="flex items-center gap-3">
                    <MonitorIcon className={`w-5 h-5 ${rcStatus.pc_online ? 'text-green-500' : 'text-red-400'}`} />
                    <div className="flex-1">
                      <p className="text-sm font-medium">
                        PC {rcStatus.pc_online ? 'Online' : 'Offline'}
                        {rcStatus.hostname && rcStatus.pc_online && (
                          <span className="text-[var(--color-text-secondary)] font-normal"> ({rcStatus.hostname})</span>
                        )}
                      </p>
                      {rcStatus.last_seen && (
                        <p className="text-xs text-[var(--color-text-secondary)]">
                          Last seen: {new Date(rcStatus.last_seen).toLocaleString()}
                        </p>
                      )}
                    </div>
                    <button
                      onClick={fetchRcStatus}
                      className="p-1.5 hover:bg-[var(--color-surface-hover)] rounded transition-colors"
                      title="Refresh"
                    >
                      <RefreshIcon className="w-3.5 h-3.5 text-[var(--color-text-secondary)]" />
                    </button>
                  </div>

                  {/* Claude Code status */}
                  {rcStatus.pc_online && (
                    <div className="flex items-center gap-3">
                      <div className={`w-2 h-2 rounded-full ${rcStatus.claude_code_running ? 'bg-green-500' : 'bg-gray-400'}`} />
                      <p className="text-sm">
                        Claude Code: {rcStatus.claude_code_running ? (
                          <span className="text-green-500">Running</span>
                        ) : (
                          <span className="text-[var(--color-text-secondary)]">Not running</span>
                        )}
                      </p>
                    </div>
                  )}

                  {/* Action / message */}
                  {!rcStatus.pc_online ? (
                    <div className="flex items-start gap-2 bg-red-500/10 text-red-400 rounded-lg px-3 py-2 text-xs">
                      <AlertIcon className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                      <span>Your PC is offline. Turn it on to use Remote Control.</span>
                    </div>
                  ) : !rcStatus.claude_code_running ? (
                    <div className="flex items-start gap-2 bg-yellow-500/10 text-yellow-400 rounded-lg px-3 py-2 text-xs">
                      <AlertIcon className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                      <span>Claude Code is not running. Start a session on your PC first.</span>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <div className="flex items-start gap-2 bg-green-500/10 text-green-400 rounded-lg px-3 py-2 text-xs">
                        <CheckIcon className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                        <span>Ready! Open the Claude app on your phone to connect.</span>
                      </div>
                      <a
                        href="https://claude.ai/remote-control"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="block w-full text-center px-3 py-2 bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] text-white text-sm rounded-lg transition-colors"
                      >
                        Open Claude Remote Control
                      </a>
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-sm text-[var(--color-text-secondary)]">Unable to check status.</p>
              )}
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
