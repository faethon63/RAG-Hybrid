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
} from '../common/icons';
import clsx from 'clsx';

interface MessageItemProps {
  message: Message;
}

export function MessageItem({ message }: MessageItemProps) {
  const [thinkingOpen, setThinkingOpen] = useState(false);
  const [stepsOpen, setStepsOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const showThinking = useSettingsStore((s) => s.showThinking);

  const isUser = message.role === 'user';
  const { thinking, content } = parseThinking(message.content);
  const meta = message.metadata;

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      className={clsx(
        'py-6 px-4 md:px-8',
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
          <div className="text-sm font-medium mb-1 text-[var(--color-text-secondary)]">
            {isUser ? 'You' : 'Assistant'}
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

          {/* Copy button */}
          {!isUser && (
            <div className="mt-3">
              <button
                onClick={handleCopy}
                className="flex items-center gap-1 text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-text)] transition-colors"
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
