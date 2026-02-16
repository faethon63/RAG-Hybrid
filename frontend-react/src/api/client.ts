// API Client for RAG-Hybrid Backend

const API_BASE = '/api/v1';
const API_KEY = import.meta.env.VITE_API_KEY || '';

class ApiError extends Error {
  status: number;
  statusText: string;
  body?: unknown;

  constructor(status: number, statusText: string, body?: unknown) {
    super(`API Error: ${status} ${statusText}`);
    this.name = 'ApiError';
    this.status = status;
    this.statusText = statusText;
    this.body = body;
  }
}

async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${endpoint}`;

  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...(API_KEY && { 'X-API-Key': API_KEY }),
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    let body: unknown;
    try {
      body = await response.json();
    } catch {
      body = await response.text();
    }
    throw new ApiError(response.status, response.statusText, body);
  }

  return response.json();
}

export const api = {
  // Health
  health: () => request<import('../types/api').HealthStatus>('/health'),
  reload: () => request<{ status: string; message: string }>('/reload', { method: 'POST' }),

  // Query
  query: (data: import('../types/api').QueryRequest) =>
    request<import('../types/api').QueryResponse>('/query', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  // Chats
  listChats: (project?: string, limit = 50, search?: string) => {
    const params = new URLSearchParams();
    if (project) params.set('project', project);
    if (limit) params.set('limit', String(limit));
    if (search) params.set('search', search);
    return request<import('../types/api').ChatsResponse>(`/chats?${params}`);
  },

  getChat: (chatId: string) =>
    request<import('../types/api').ChatResponse>(`/chats/${chatId}`),

  createChat: (data: { name?: string; project?: string | null; messages?: Array<{ role: string; content: string }> }) =>
    request<{ status: string; chat: import('../types/api').Chat }>('/chats', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  updateChat: (chatId: string, data: { name?: string; project?: string | null; messages?: Array<{ role: string; content: string }> }) =>
    request<{ status: string; chat: import('../types/api').Chat }>(`/chats/${chatId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),

  deleteChat: (chatId: string) =>
    request<{ status: string; chat_id: string }>(`/chats/${chatId}`, { method: 'DELETE' }),

  renameChat: (chatId: string, name: string) =>
    request<{ status: string; chat_id: string; name: string }>(`/chats/${chatId}/rename`, {
      method: 'PATCH',
      body: JSON.stringify({ name }),
    }),

  // Projects
  listProjects: () =>
    request<import('../types/api').ProjectsResponse>('/projects'),

  getProject: (name: string) =>
    request<import('../types/api').ProjectDetail>(`/projects/${encodeURIComponent(name)}`),

  createProject: (data: { name: string; description?: string; system_prompt?: string; instructions?: string; allowed_paths?: string[] }) =>
    request<{ status: string; project: import('../types/api').Project }>('/projects', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  updateProject: (name: string, data: Partial<import('../types/api').ProjectConfig>) =>
    request<{ status: string; project: string; config: import('../types/api').ProjectConfig }>(`/projects/${encodeURIComponent(name)}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),

  deleteProject: (name: string) =>
    request<{ status: string; project: string; deleted: string[] }>(`/projects/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    }),

  indexProject: (name: string) =>
    request<import('../types/api').IndexResponse>(`/projects/${encodeURIComponent(name)}/index`, {
      method: 'POST',
    }),

  // Project KB files
  listProjectFiles: (name: string) =>
    request<import('../types/api').ProjectFilesResponse>(`/projects/${encodeURIComponent(name)}/files`),

  uploadProjectFiles: async (name: string, files: File[]): Promise<import('../types/api').UploadFilesResponse> => {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    const response = await fetch(`${API_BASE}/projects/${encodeURIComponent(name)}/files`, {
      method: 'POST',
      ...(API_KEY && { headers: { 'X-API-Key': API_KEY } }),
      body: formData,
    });
    if (!response.ok) {
      let body: unknown;
      try {
        body = await response.json();
      } catch {
        body = await response.text();
      }
      throw new ApiError(response.status, response.statusText, body);
    }
    return response.json();
  },

  deleteProjectFile: (name: string, filename: string) =>
    request<{ status: string; project: string; deleted: string }>(`/projects/${encodeURIComponent(name)}/files/${encodeURIComponent(filename)}`, {
      method: 'DELETE',
    }),

  getFileContent: (name: string, filename: string) =>
    request<import('../types/api').FileContentResponse>(`/projects/${encodeURIComponent(name)}/files/${encodeURIComponent(filename)}`),

  updateFileContent: (name: string, filename: string, content: string) =>
    request<import('../types/api').UpdateFileResponse>(`/projects/${encodeURIComponent(name)}/files/${encodeURIComponent(filename)}`, {
      method: 'PUT',
      body: JSON.stringify({ content }),
    }),

  downloadFile: async (name: string, filename: string): Promise<void> => {
    const url = `${API_BASE}/projects/${encodeURIComponent(name)}/files/${encodeURIComponent(filename)}/download`;
    const response = await fetch(url, {
      ...(API_KEY && { headers: { 'X-API-Key': API_KEY } }),
    });
    if (!response.ok) {
      throw new ApiError(response.status, response.statusText);
    }
    const blob = await response.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
  },

  // Data sync (local <-> VPS)
  syncPushToVps: (vpsUrl: string = 'https://rag.coopeverything.org') =>
    request<{ status: string; message: string; vps_response?: Record<string, unknown> }>(`/sync/push?vps_url=${encodeURIComponent(vpsUrl)}`, {
      method: 'POST',
    }),

  // Settings
  getSettings: () =>
    request<import('../types/api').SettingsResponse>('/settings'),

  updateSettings: (data: Partial<import('../types/api').GlobalSettings>) =>
    request<{ status: string; settings: import('../types/api').GlobalSettings }>('/settings', {
      method: 'PUT',
      body: JSON.stringify(data),
    }),

  // Document indexing
  indexDocuments: (documents: Array<{ text: string; title: string; metadata?: Record<string, unknown> }>, project?: string) =>
    request<import('../types/api').IndexResponse>('/index', {
      method: 'POST',
      body: JSON.stringify({ documents, project }),
    }),
};

export { ApiError };
