import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import type { Message } from '../../types/api';
import { parseThinking, formatTokens, formatCost, formatTime, formatConfidence } from '../../utils/parseThinking';
import { useSettingsStore } from '../../stores/settingsStore';
import {
  UserIcon,
  BotIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  CopyIcon,
  CheckIcon,
  EditIcon,
  TrashIcon,
  CloseIcon,
} from '../common/icons';
import clsx from 'clsx';

interface MessageItemProps {
  message: Message;
  isFirstMessage?: boolean;
  onEdit?: (id: string, content: string, regenerate: boolean) => void;
  onDelete?: (id: string) => void;
}

export function MessageItem({ message, isFirstMessage, onEdit, onDelete }: MessageItemProps) {
  const [thinkingOpen, setThinkingOpen] = useState(false);
  const [stepsOpen, setStepsOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState('');
  const [confirmDelete, setConfirmDelete] = useState(false);
  const showThinking = useSettingsStore((s) => s.showThinking);

  const isUser = message.role === 'user';
  const { thinking, content } = parseThinking(message.content);
  const meta = message.metadata;

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleStartEdit = () => {
    setEditContent(message.content);
    setIsEditing(true);
  };

  const handleSaveEdit = () => {
    if (editContent.trim() && onEdit) {
      onEdit(message.id, editContent, isUser);
    }
    setIsEditing(false);
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    setEditContent('');
  };

  const handleDeleteClick = () => {
    if (confirmDelete) {
      onDelete?.(message.id);
      setConfirmDelete(false);
    } else {
      setConfirmDelete(true);
      setTimeout(() => setConfirmDelete(false), 3000);
    }
  };

  const handleEditKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      handleCancelEdit();
    } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleSaveEdit();
    }
  };

  return (
    <div
      className={clsx(
        'group/msg relative py-6 px-4 md:px-8',
        isUser ? 'bg-transparent' : 'bg-[var(--color-surface)]/30'
      )}
    >
      <div className="max-w-3xl mx-auto flex gap-4">
        {/* Avatar */}
        <div
          className={clsx(
            'w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0',
            isUser
              ? 'bg-[var(--color-primary)]'
              : 'bg-[var(--color-surface)] border border-[var(--color-border)]'
          )}
        >
          {isUser ? (
            <UserIcon className="w-4 h-4 text-white" />
          ) : (
            <BotIcon className="w-4 h-4 text-[var(--color-primary)]" />
          )}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {/* Role label */}
          <div className="text-sm font-medium mb-1 text-[var(--color-text-secondary)] flex items-center gap-2">
            <span>{isUser ? 'You' : 'Assistant'}</span>
            {isFirstMessage && isUser && message.timestamp && (
              <span className="text-xs font-normal text-[var(--color-text-secondary)]/60">
                {new Date(message.timestamp).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })}
              </span>
            )}
          </div>

          {/* Thinking block */}
          {thinking && showThinking && (
            <div className="mb-3">
              <button
                onClick={() => setThinkingOpen(!thinkingOpen)}
                className="flex items-center gap-1 text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text)] transition-colors"
              >
                {thinkingOpen ? (
                  <ChevronDownIcon className="w-4 h-4" />
                ) : (
                  <ChevronRightIcon className="w-4 h-4" />
                )}
                <span>Thinking</span>
              </button>
              {thinkingOpen && (
                <div className="mt-2 p-3 bg-[var(--color-surface)] rounded-lg border border-[var(--color-border)] text-sm text-[var(--color-text-secondary)]">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {thinking}
                  </ReactMarkdown>
                </div>
              )}
            </div>
          )}

          {/* Agent steps */}
          {meta?.agent_steps && meta.agent_steps.length > 0 && (
            <div className="mb-3">
              <button
                onClick={() => setStepsOpen(!stepsOpen)}
                className="flex items-center gap-1 text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text)] transition-colors"
              >
                {stepsOpen ? (
                  <ChevronDownIcon className="w-4 h-4" />
                ) : (
                  <ChevronRightIcon className="w-4 h-4" />
                )}
                <span>Agent Steps ({meta.agent_steps.length})</span>
              </button>
              {stepsOpen && (
                <div className="mt-2 space-y-2">
                  {meta.agent_steps.map((step, idx) => (
                    <div
                      key={idx}
                      className="p-3 bg-[var(--color-surface)] rounded-lg border border-[var(--color-border)] text-sm"
                    >
                      <div className="font-medium text-[var(--color-primary)]">
                        {step.tool}
                      </div>
                      <div className="text-[var(--color-text-secondary)] mt-1 truncate">
                        {step.input.slice(0, 200)}
                        {step.input.length > 200 && '...'}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Main content */}
          {isEditing ? (
            <div className="space-y-2">
              <textarea
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                onKeyDown={handleEditKeyDown}
                autoFocus
                rows={Math.max(3, editContent.split('\n').length)}
                className="w-full bg-[var(--color-background)] border border-[var(--color-border)] rounded-lg p-3 text-sm outline-none focus:border-[var(--color-primary)] resize-y"
              />
              <div className="flex items-center gap-2">
                <button
                  onClick={handleSaveEdit}
                  className="flex items-center gap-1 px-3 py-1.5 bg-[var(--color-primary)] text-white rounded-lg text-xs hover:opacity-90 transition-opacity"
                >
                  <CheckIcon className="w-3 h-3" />
                  {isUser ? 'Save & Resend' : 'Save'}
                </button>
                <button
                  onClick={handleCancelEdit}
                  className="flex items-center gap-1 px-3 py-1.5 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg text-xs hover:bg-[var(--color-surface-hover)] transition-colors"
                >
                  <CloseIcon className="w-3 h-3" />
                  Cancel
                </button>
                <span className="text-xs text-[var(--color-text-secondary)] ml-2">
                  Ctrl+Enter to save, Esc to cancel
                </span>
              </div>
            </div>
          ) : (
            <div className="markdown-content">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  code({ node, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '');
                    const inline = !match && !className;
                    return inline ? (
                      <code className={className} {...props}>
                        {children}
                      </code>
                    ) : (
                      <SyntaxHighlighter
                        style={oneDark}
                        language={match?.[1] || 'text'}
                        PreTag="div"
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
                    );
                  },
                }}
              >
                {content}
              </ReactMarkdown>
            </div>
          )}

          {/* Metadata */}
          {meta && !isUser && (
            <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-[var(--color-text-secondary)]">
              {meta.model_used && (
                <span className="px-2 py-1 bg-[var(--color-surface)] rounded">
                  {meta.model_used}
                </span>
              )}
              {meta.tokens && (
                <span>
                  {formatTokens(meta.tokens.input)} / {formatTokens(meta.tokens.output)} tokens
                </span>
              )}
              {meta.estimated_cost !== undefined && (
                <span>{formatCost(meta.estimated_cost)}</span>
              )}
              {meta.processing_time !== undefined && (
                <span>{formatTime(meta.processing_time)}</span>
              )}
              {meta.confidence !== undefined && (
                <div className="flex items-center gap-2">
                  <span>Confidence:</span>
                  <div className="w-20 h-1.5 bg-[var(--color-surface)] rounded-full overflow-hidden">
                    <div
                      className="h-full bg-[var(--color-primary)] transition-all"
                      style={{ width: `${meta.confidence * 100}%` }}
                    />
                  </div>
                  <span>{formatConfidence(meta.confidence)}</span>
                </div>
              )}
            </div>
          )}

          {/* Routing Info */}
          {meta?.routing_info && !isUser && (
            <div className="mt-2 p-2 bg-[var(--color-surface)]/50 rounded text-xs text-[var(--color-text-secondary)] border border-[var(--color-border)]/50">
              <div className="flex flex-wrap items-center gap-2">
                {meta.routing_info.tools_used && meta.routing_info.tools_used.length > 0 ? (
                  <>
                    {meta.routing_info.tools_used.map((tool, idx) => {
                      // Color code by provider type
                      const isPerplexity = tool.includes('perplexity');
                      const isClaude = tool.includes('claude');
                      const bgClass = isPerplexity
                        ? 'bg-purple-500/20 text-purple-400'
                        : isClaude
                          ? 'bg-amber-500/20 text-amber-400'
                          : 'bg-[var(--color-surface)]';
                      return (
                        <span key={idx} className={`px-1.5 py-0.5 rounded ${bgClass}`}>
                          {tool === 'perplexity_pro' ? 'Perplexity Pro' : tool === 'perplexity' ? 'Perplexity' : tool}
                        </span>
                      );
                    })}
                    {meta.routing_info.claude_model && (
                      <>
                        <span>→</span>
                        <span className="px-1.5 py-0.5 bg-amber-500/20 text-amber-400 rounded">
                          Claude {meta.routing_info.claude_model}
                        </span>
                      </>
                    )}
                  </>
                ) : (
                  <span className="px-1.5 py-0.5 bg-[var(--color-primary)]/20 rounded text-[var(--color-primary)]">
                    groq (direct)
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Action bar - hover to show */}
          {!isEditing && (
            <div className="mt-2 flex items-center gap-1 opacity-0 group-hover/msg:opacity-100 transition-opacity">
              <button
                onClick={handleStartEdit}
                className="flex items-center gap-1 px-2 py-1 text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface)] rounded transition-colors"
                title="Edit"
              >
                <EditIcon className="w-3 h-3" />
                <span>Edit</span>
              </button>
              <button
                onClick={handleDeleteClick}
                className={clsx(
                  'flex items-center gap-1 px-2 py-1 text-xs rounded transition-colors',
                  confirmDelete
                    ? 'text-red-400 bg-red-500/10 hover:bg-red-500/20'
                    : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface)]'
                )}
                title={confirmDelete ? 'Click again to confirm delete' : 'Delete'}
              >
                {confirmDelete ? (
                  <>
                    <CheckIcon className="w-3 h-3" />
                    <span>Confirm?</span>
                  </>
                ) : (
                  <>
                    <TrashIcon className="w-3 h-3" />
                    <span>Delete</span>
                  </>
                )}
              </button>
              <button
                onClick={handleCopy}
                className="flex items-center gap-1 px-2 py-1 text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface)] rounded transition-colors"
                title="Copy"
              >
                {copied ? (
                  <>
                    <CheckIcon className="w-3 h-3" />
                    <span>Copied</span>
                  </>
                ) : (
                  <>
                    <CopyIcon className="w-3 h-3" />
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
