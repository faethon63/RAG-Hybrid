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

  // Search
  searchQuery: string;
  searchResults: ChatSummary[];
  searchLoading: boolean;

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

  // Message editing
  editMessage: (messageId: string, newContent: string, project?: string | null) => Promise<void>;
  editAndRegenerate: (messageId: string, newContent: string, mode: string, model: string, project: string | null) => Promise<void>;
  deleteMessage: (messageId: string, project?: string | null) => Promise<void>;

  // Search
  searchChats: (query: string) => Promise<void>;
  clearSearch: () => void;
  setSearchQuery: (query: string) => void;

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
  searchQuery: '',
  searchResults: [],
  searchLoading: false,
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

  editMessage: async (messageId, newContent, project) => {
    set((state) => ({
      messages: state.messages.map((m) =>
        m.id === messageId ? { ...m, content: newContent } : m
      ),
    }));
    await get().saveChat(project);
  },

  editAndRegenerate: async (messageId, newContent, mode, model, project) => {
    const { messages, saveChat, lastAttachedFiles } = get();
    const idx = messages.findIndex((m) => m.id === messageId);
    if (idx === -1) return;

    // Keep messages up to and including the edited one, with updated content
    const truncated = messages.slice(0, idx + 1).map((m) =>
      m.id === messageId ? { ...m, content: newContent } : m
    );
    set({ messages: truncated, isLoading: true, error: null });

    try {
      // Build conversation history from messages before the edited one
      const history = truncated.slice(0, -1).slice(-6).map((m) => ({
        role: m.role,
        content: m.content,
      }));

      // Strip "[Attached: ...]" suffix to get the raw query text
      const query = newContent.replace(/\n\n\[Attached:.*\]$/, '');

      const filesPayload = lastAttachedFiles?.length > 0 ? lastAttachedFiles.map(f => ({
        name: f.name,
        type: f.type,
        data: f.data,
        isImage: f.isImage,
      })) : undefined;

      const response = await api.query({
        query: query || 'Analyze the attached file(s)',
        mode: mode as 'auto' | 'private' | 'research' | 'deep_agent',
        model,
        project,
        max_results: 5,
        include_sources: true,
        conversation_history: history,
        files: filesPayload,
      });

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
          routing_info: response.routing_info,
        },
      };

      set((state) => ({
        messages: [...state.messages, assistantMessage],
        isLoading: false,
        lastSources: response.sources || [],
        lastAgentSteps: response.agent_steps || [],
      }));

      await saveChat(project);
    } catch (err) {
      console.error('Regeneration failed:', err);
      const errorMessage = err instanceof Error ? err.message : 'Regeneration failed';
      set({ isLoading: false, error: errorMessage });
    }
  },

  deleteMessage: async (messageId, project) => {
    set((state) => {
      const idx = state.messages.findIndex((m) => m.id === messageId);
      if (idx === -1) return state;
      const msg = state.messages[idx];
      const next = state.messages[idx + 1];
      // If deleting a user message, also remove the assistant reply that follows
      const removeNext = msg.role === 'user' && next && next.role === 'assistant';
      const idsToRemove = new Set([messageId]);
      if (removeNext) idsToRemove.add(next.id);
      return { messages: state.messages.filter((m) => !idsToRemove.has(m.id)) };
    });
    await get().saveChat(project);
  },

  searchChats: async (query) => {
    if (!query || query.length < 2) {
      set({ searchResults: [], searchLoading: false });
      return;
    }
    set({ searchLoading: true });
    try {
      const response = await api.listChats(undefined, 30, query);
      set({ searchResults: response.chats, searchLoading: false });
    } catch (err) {
      console.error('Failed to search chats:', err);
      set({ searchLoading: false });
    }
  },

  clearSearch: () => set({ searchQuery: '', searchResults: [], searchLoading: false }),

  setSearchQuery: (query) => set({ searchQuery: query }),

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
      // Ensure every message has a unique ID and timestamp (server may not store them)
      const chatCreatedAt = response.chat.created_at;
      const messages = (response.chat.messages || []).map((m: Message, idx: number) => ({
        ...m,
        id: m.id || `msg_loaded_${chatId}_${idx}`,
        // Fallback: use chat created_at for first message if no timestamp stored
        timestamp: m.timestamp || (idx === 0 && chatCreatedAt ? chatCreatedAt : undefined),
      }));
      set({
        currentChatId: chatId,
        messages,
        isLoading: false,
      });
    } catch (err) {
      console.error('Failed to load chat:', err);
      set({ error: 'Failed to load chat', isLoading: false });
    }
  },

  saveChat: async (project) => {
    const { messages, currentChatId, loadChats } = get();
    if (messages.length === 0) return;

    try {
      const chatData = {
        project,
        messages: messages.map((m) => ({
          id: m.id,
          role: m.role,
          content: m.content,
          timestamp: m.timestamp,
          metadata: m.metadata,
        })),
      };

      if (currentChatId) {
        await api.updateChat(currentChatId, chatData);
      } else {
        const response = await api.createChat(chatData);
        set({ currentChatId: response.chat.id });
      }
      // Refresh chat list so sidebar shows the new/updated chat
      await loadChats(project);
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
        model,
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
          routing_info: response.routing_info,
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
