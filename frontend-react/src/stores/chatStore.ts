import { create } from 'zustand';
import type { Message, Source, AgentStep, ChatSummary } from '../types/api';
import { api } from '../api/client';

interface ChatState {
  // Current conversation
  messages: Message[];
  currentChatId: string | null;
  isLoading: boolean;
  error: string | null;

  // Chat list
  chats: ChatSummary[];
  chatsLoading: boolean;

  // Last query metadata
  lastSources: Source[];
  lastAgentSteps: AgentStep[];

  // Persisted files for follow-up questions
  lastAttachedFiles: AttachedFile[];

  // Actions
  addMessage: (message: Message) => void;
  updateLastMessage: (content: string) => void;
  setMessages: (messages: Message[]) => void;
  clearMessages: () => void;
  setCurrentChatId: (id: string | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setLastSources: (sources: Source[]) => void;
  setLastAgentSteps: (steps: AgentStep[]) => void;

  // API operations
  loadChats: (project?: string | null) => Promise<void>;
  loadChat: (chatId: string) => Promise<void>;
  saveChat: (project?: string | null) => Promise<void>;
  deleteChat: (chatId: string) => Promise<void>;
  newChat: () => void;
  sendQuery: (query: string, mode: string, model: string, project: string | null, files?: AttachedFile[]) => Promise<void>;
}

interface AttachedFile {
  id: string;
  name: string;
  type: string;
  size: number;
  data: string;
  isImage: boolean;
}

let messageIdCounter = 0;
const generateMessageId = () => `msg_${Date.now()}_${++messageIdCounter}`;

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  currentChatId: null,
  isLoading: false,
  error: null,
  chats: [],
  chatsLoading: false,
  lastSources: [],
  lastAgentSteps: [],
  lastAttachedFiles: [],

  addMessage: (message) =>
    set((state) => ({
      messages: [...state.messages, message],
    })),

  updateLastMessage: (content) =>
    set((state) => {
      const messages = [...state.messages];
      if (messages.length > 0) {
        const lastMsg = messages[messages.length - 1];
        messages[messages.length - 1] = { ...lastMsg, content };
      }
      return { messages };
    }),

  setMessages: (messages) => set({ messages }),
  clearMessages: () => set({ messages: [], currentChatId: null, lastSources: [], lastAgentSteps: [], lastAttachedFiles: [] }),
  setCurrentChatId: (id) => set({ currentChatId: id }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
  setLastSources: (sources) => set({ lastSources: sources }),
  setLastAgentSteps: (steps) => set({ lastAgentSteps: steps }),

  loadChats: async (project) => {
    set({ chatsLoading: true });
    try {
      const response = await api.listChats(project || undefined);
      set({ chats: response.chats, chatsLoading: false });
    } catch (err) {
      console.error('Failed to load chats:', err);
      set({ chatsLoading: false });
    }
  },

  loadChat: async (chatId) => {
    set({ isLoading: true, error: null });
    try {
      const response = await api.getChat(chatId);
      set({
        currentChatId: chatId,
        messages: response.chat.messages,
        isLoading: false,
      });
    } catch (err) {
      console.error('Failed to load chat:', err);
      set({ error: 'Failed to load chat', isLoading: false });
    }
  },

  saveChat: async (project) => {
    const { messages, currentChatId } = get();
    if (messages.length === 0) return;

    try {
      const chatData = {
        project,
        messages: messages.map((m) => ({ role: m.role, content: m.content })),
      };

      if (currentChatId) {
        await api.updateChat(currentChatId, chatData);
      } else {
        const response = await api.createChat(chatData);
        set({ currentChatId: response.chat.id });
      }
    } catch (err) {
      console.error('Failed to save chat:', err);
    }
  },

  deleteChat: async (chatId) => {
    try {
      await api.deleteChat(chatId);
      const { currentChatId, chats } = get();
      set({ chats: chats.filter((c) => c.id !== chatId) });
      if (currentChatId === chatId) {
        set({ currentChatId: null, messages: [] });
      }
    } catch (err) {
      console.error('Failed to delete chat:', err);
    }
  },

  newChat: () => {
    set({
      messages: [],
      currentChatId: null,
      lastSources: [],
      lastAgentSteps: [],
      lastAttachedFiles: [],
      error: null,
    });
  },

  sendQuery: async (query, mode, model, project, files) => {
    const { messages, saveChat, lastAttachedFiles } = get();

    // Use new files if provided, otherwise reuse last attached files for follow-ups
    const effectiveFiles = (files && files.length > 0) ? files : lastAttachedFiles;
    const hasNewFiles = files && files.length > 0;

    // Build user message content with file info
    let userContent = query;
    const hasImages = effectiveFiles?.some(f => f.isImage);
    if (hasNewFiles && files.length > 0) {
      const fileNames = files.map(f => f.name).join(', ');
      if (!query) {
        userContent = `[Attached: ${fileNames}]`;
      } else {
        userContent = `${query}\n\n[Attached: ${fileNames}]`;
      }
    }

    // Add user message
    const userMessage: Message = {
      id: generateMessageId(),
      role: 'user',
      content: userContent,
      timestamp: new Date().toISOString(),
    };
    set((state) => ({
      messages: [...state.messages, userMessage],
      isLoading: true,
      error: null,
    }));

    try {
      // Build conversation history (last 6 messages)
      const history = messages.slice(-6).map((m) => ({
        role: m.role,
        content: m.content,
      }));

      // If images attached, force local model (vision)
      const effectiveModel = hasImages ? 'local' : model;

      // Build files payload from effective files (new or persisted)
      const filesPayload = effectiveFiles?.length > 0 ? effectiveFiles.map(f => ({
        name: f.name,
        type: f.type,
        data: f.data,
        isImage: f.isImage,
      })) : undefined;

      // Store new files for follow-up questions
      if (hasNewFiles) {
        set({ lastAttachedFiles: files });
      }

      const response = await api.query({
        query: query || 'Analyze the attached file(s)',
        mode: mode as 'auto' | 'private' | 'research' | 'deep_agent',
        model: effectiveModel,
        project,
        max_results: 5,
        include_sources: true,
        conversation_history: history,
        files: filesPayload,
      });

      // Add assistant message
      const assistantMessage: Message = {
        id: generateMessageId(),
        role: 'assistant',
        content: response.answer,
        timestamp: response.timestamp,
        metadata: {
          mode: response.mode,
          model_used: response.model_used,
          processing_time: response.processing_time,
          confidence: response.confidence,
          tokens: response.tokens,
          estimated_cost: response.estimated_cost,
          sources: response.sources,
          agent_steps: response.agent_steps,
        },
      };

      set((state) => ({
        messages: [...state.messages, assistantMessage],
        isLoading: false,
        lastSources: response.sources || [],
        lastAgentSteps: response.agent_steps || [],
      }));

      // Auto-save
      await saveChat(project);
    } catch (err) {
      console.error('Query failed:', err);
      const errorMessage = err instanceof Error ? err.message : 'Query failed';
      set({ isLoading: false, error: errorMessage });
    }
  },
}));
