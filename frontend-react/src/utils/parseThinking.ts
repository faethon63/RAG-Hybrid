/**
 * Parse thinking/reasoning blocks from assistant responses
 * Supports multiple formats:
 * - <think>...</think>
 * - <thinking>...</thinking>
 * - **Thinking:** ...
 */

interface ParsedContent {
  thinking: string | null;
  content: string;
}

export function parseThinking(text: string): ParsedContent {
  let thinking: string | null = null;
  let content = text;

  // Pattern 1: <think>...</think> (case-insensitive)
  const thinkMatch = text.match(/<think>([\s\S]*?)<\/think>/i);
  if (thinkMatch) {
    thinking = thinkMatch[1].trim();
    content = text.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
    return { thinking, content };
  }

  // Pattern 2: <thinking>...</thinking> (case-insensitive)
  const thinkingMatch = text.match(/<thinking>([\s\S]*?)<\/thinking>/i);
  if (thinkingMatch) {
    thinking = thinkingMatch[1].trim();
    content = text.replace(/<thinking>[\s\S]*?<\/thinking>/gi, '').trim();
    return { thinking, content };
  }

  // Pattern 3: Unclosed <think> until double newline
  const unclosedThink = text.match(/<think>([\s\S]*?)(?:\n\n|$)/i);
  if (unclosedThink && !text.includes('</think>')) {
    thinking = unclosedThink[1].trim();
    content = text.replace(/<think>[\s\S]*?(?:\n\n|$)/gi, '').trim();
    return { thinking, content };
  }

  // Pattern 4: **Thinking:** format
  const boldThinking = text.match(/\*\*Thinking:\*\*\s*([\s\S]*?)(?=\n\n|\*\*|$)/);
  if (boldThinking) {
    thinking = boldThinking[1].trim();
    content = text.replace(/\*\*Thinking:\*\*\s*[\s\S]*?(?=\n\n|\*\*|$)/, '').trim();
    return { thinking, content };
  }

  return { thinking: null, content };
}

/**
 * Format token counts with commas
 */
export function formatTokens(count: number): string {
  return count.toLocaleString();
}

/**
 * Format cost in USD
 */
export function formatCost(cost: number): string {
  if (cost === 0) return 'Free';
  if (cost < 0.01) return '<$0.01';
  return `$${cost.toFixed(2)}`;
}

/**
 * Format processing time
 */
export function formatTime(seconds: number): string {
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
  return `${seconds.toFixed(1)}s`;
}

/**
 * Format confidence as percentage
 */
export function formatConfidence(confidence: number): string {
  return `${Math.round(confidence * 100)}%`;
}

/**
 * Truncate text with ellipsis
 */
export function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength - 3) + '...';
}
