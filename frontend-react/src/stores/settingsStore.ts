import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { GlobalSettings, HealthStatus } from '../types/api';
import { api } from '../api/client';

interface SettingsState {
  // Current selection (per-query)
  mode: string;
  model: string;

  // Global settings from backend
  globalSettings: GlobalSettings | null;
  settingsLoading: boolean;

  // Health
  health: HealthStatus | null;
  healthLoading: boolean;

  // UI state
  showThinking: boolean;
  sidebarOpen: boolean;
  sidebarWidth: number;
  showSettings: boolean;

  // Actions
  setMode: (mode: string) => void;
  setModel: (model: string) => void;
  setShowThinking: (show: boolean) => void;
  setSidebarOpen: (open: boolean) => void;
  setSidebarWidth: (width: number) => void;
  setShowSettings: (show: boolean) => void;

  // API operations
  loadSettings: () => Promise<void>;
  saveSettings: (settings: Partial<GlobalSettings>) => Promise<void>;
  checkHealth: () => Promise<void>;
  reloadConfig: () => Promise<void>;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      mode: 'auto',
      model: 'auto',
      globalSettings: null,
      settingsLoading: false,
      health: null,
      healthLoading: false,
      showThinking: true,
      sidebarOpen: true,
      sidebarWidth: 256,
      showSettings: false,

      setMode: (mode) => set({ mode }),
      setModel: (model) => set({ model }),
      setShowThinking: (show) => set({ showThinking: show }),
      setSidebarOpen: (open) => set({ sidebarOpen: open }),
      setSidebarWidth: (width) => set({ sidebarWidth: Math.max(200, Math.min(500, width)) }),
      setShowSettings: (show) => set({ showSettings: show }),

      loadSettings: async () => {
        set({ settingsLoading: true });
        try {
          const response = await api.getSettings();
          const settings = response.settings;
          // Only load globalSettings - don't overwrite user's mode/model selection
          // (those are persisted to localStorage and should be preserved)
          set({
            globalSettings: settings,
            settingsLoading: false,
          });
        } catch (err) {
          console.error('Failed to load settings:', err);
          set({ settingsLoading: false });
        }
      },

      saveSettings: async (settings) => {
        try {
          const response = await api.updateSettings(settings);
          set({ globalSettings: response.settings });
        } catch (err) {
          console.error('Failed to save settings:', err);
          throw err;
        }
      },

      checkHealth: async () => {
        set({ healthLoading: true });
        try {
          const health = await api.health();
          set({ health, healthLoading: false });
        } catch (err) {
          console.error('Health check failed:', err);
          set({
            health: {
              status: 'offline',
              services: {
                local_rag: false,
                claude_api: false,
                perplexity_api: false,
                ollama: false,
                chromadb: false,
              },
              timestamp: new Date().toISOString(),
            },
            healthLoading: false,
          });
        }
      },

      reloadConfig: async () => {
        try {
          await api.reload();
          await get().loadSettings();
          await get().checkHealth();
        } catch (err) {
          console.error('Failed to reload config:', err);
          throw err;
        }
      },
    }),
    {
      name: 'rag-settings',
      partialize: (state) => ({
        mode: state.mode,
        model: state.model,
        showThinking: state.showThinking,
        sidebarOpen: state.sidebarOpen,
        sidebarWidth: state.sidebarWidth,
      }),
    }
  )
);

// Model options - Groq orchestrates automatically, Claude selection is tiered
export const MODEL_OPTIONS = [
  { value: 'auto', label: 'Smart (Groq orchestrates)', description: 'Groq routes queries, uses tiered Claude when needed', requiresOllama: false },
  { value: 'local', label: 'Local (Ollama)', description: 'Offline, uses local qwen2.5 model', requiresOllama: true },
];

export const MODE_OPTIONS = [
  { value: 'auto', label: 'Smart', description: 'Groq + web search + tiered Claude fallback', requiresOllama: false },
  { value: 'private', label: 'Private', description: 'Local Ollama only, no external APIs', requiresOllama: true },
  { value: 'research', label: 'Research', description: 'Deep Perplexity Pro search', requiresOllama: false },
  { value: 'deep_agent', label: 'Deep Agent', description: 'Multi-step research agent', requiresOllama: false },
];

// Helper to get available options based on health
export function getAvailableModelOptions(ollamaAvailable: boolean) {
  return MODEL_OPTIONS.filter(opt => !opt.requiresOllama || ollamaAvailable);
}

export function getAvailableModeOptions(ollamaAvailable: boolean) {
  return MODE_OPTIONS.filter(opt => !opt.requiresOllama || ollamaAvailable);
}
