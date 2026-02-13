// API Response Types

export interface Source {
  type: 'local_doc' | 'web' | 'research' | 'agent_step' | 'tool_result';
  title: string;
  url?: string;
  snippet: string;
  score?: number;
}

export interface TokenUsage {
  input: number;
  output: number;
}

export interface AgentStep {
  tool: string;
  input: string;
  output?: string;
}

export interface RoutingInfo {
  orchestrator: string;
  tools_used: string[];
  claude_model?: string | null;
  reasoning?: string;
}

export interface AttachedFile {
  name: string;
  type: string;
  data: string; // base64 for images/PDFs, text content for text files
  isImage: boolean;
}

export interface QueryRequest {
  query: string;
  mode?: 'auto' | 'private' | 'research' | 'deep_agent';
  model?: string;
  project?: string | null;
  max_results?: number;
  include_sources?: boolean;
  conversation_history?: Array<{ role: 'user' | 'assistant'; content: string }>;
  files?: AttachedFile[];
}

export interface QueryResponse {
  query: string;
  answer: string;
  mode: string;
  sources: Source[];
  processing_time: number;
  timestamp: string;
  confidence?: number;
  model_used?: string;
  tokens?: TokenUsage;
  estimated_cost?: number;
  agent_steps?: AgentStep[];
  routing_info?: RoutingInfo;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
  metadata?: {
    mode?: string;
    model_used?: string;
    processing_time?: number;
    confidence?: number;
    tokens?: TokenUsage;
    estimated_cost?: number;
    sources?: Source[];
    agent_steps?: AgentStep[];
    routing_info?: RoutingInfo;
  };
}

export interface Chat {
  id: string;
  name: string;
  project?: string | null;
  messages: Message[];
  created_at: string;
  updated_at: string;
}

export interface ChatSummary {
  id: string;
  name: string;
  project?: string | null;
  created_at: string;
  updated_at: string;
  score?: number;
}

export interface ProjectConfig {
  description: string;
  system_prompt: string;
  instructions: string;
  allowed_paths: string[];
}

export interface Project {
  name: string;
  description: string;
  has_config: boolean;
}

export interface ProjectDetail {
  project: string;
  config: ProjectConfig;
  timestamp: string;
}

export interface GlobalSettings {
  default_model: string;
  default_mode: string;
  global_instructions: string;
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'offline';
  services: {
    local_rag: boolean;
    claude_api: boolean;
    perplexity_api: boolean;
    ollama: boolean;
    chromadb: boolean;
  };
  timestamp: string;
}

// API Response Wrappers
export interface ChatsResponse {
  chats: ChatSummary[];
  count: number;
  timestamp: string;
}

export interface ChatResponse {
  chat: Chat;
  timestamp: string;
}

export interface ProjectsResponse {
  projects: Project[];
  count: number;
  timestamp: string;
}

export interface SettingsResponse {
  settings: GlobalSettings;
  timestamp: string;
}

export interface IndexResponse {
  status: string;
  indexed_count?: number;
  indexed_chunks?: number;
  files?: string[];
  skipped?: number;
  total_files?: number;
  project?: string | null;
  message?: string;
  synced_to_vps?: boolean;
  sync_error?: string;
  timestamp: string;
}

// Project KB File types
export interface ProjectFile {
  name: string;
  size: number;
  modified: string;
  indexed: boolean;
}

export interface ProjectFilesResponse {
  project: string;
  files: ProjectFile[];
  count: number;
}

export interface UploadFilesResponse {
  status: string;
  project: string;
  uploaded: string[];
  failed: Array<{ name: string; error: string }>;
  indexed_chunks: number;
}
