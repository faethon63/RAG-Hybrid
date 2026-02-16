import { useState, useEffect, useRef, useCallback } from 'react';
import { useProjectStore } from '../../stores/projectStore';
import {
  UploadIcon,
  FileIcon,
  TrashIcon,
  LoaderIcon,
  CheckIcon,
  AlertIcon,
  DownloadIcon,
  EditIcon,
} from '../common/icons';

const ALLOWED_EXTENSIONS = ['.txt', '.md', '.json', '.py', '.js', '.ts', '.html', '.css', '.yaml', '.yml', '.rst', '.csv', '.pdf'];
const TEXT_EDITABLE_EXTENSIONS = ['.txt', '.md', '.json', '.py', '.js', '.ts', '.html', '.css', '.yaml', '.yml', '.rst', '.csv'];
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB

interface FileUploadZoneProps {
  projectName: string;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function FileUploadZone({ projectName }: FileUploadZoneProps) {
  const projectFiles = useProjectStore((s) => s.projectFiles);
  const filesLoading = useProjectStore((s) => s.filesLoading);
  const uploading = useProjectStore((s) => s.uploading);
  const loadProjectFiles = useProjectStore((s) => s.loadProjectFiles);
  const uploadFiles = useProjectStore((s) => s.uploadFiles);
  const deleteFile = useProjectStore((s) => s.deleteFile);
  const openFileEditor = useProjectStore((s) => s.openFileEditor);
  const downloadFile = useProjectStore((s) => s.downloadFile);

  const [dragOver, setDragOver] = useState(false);
  const [uploadResult, setUploadResult] = useState<{ uploaded: string[]; failed: Array<{ name: string; error: string }>; indexed: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [deletingFile, setDeletingFile] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load files when project changes
  useEffect(() => {
    if (projectName) {
      loadProjectFiles(projectName);
    }
  }, [projectName, loadProjectFiles]);

  const validateFiles = useCallback((files: File[]): { valid: File[]; invalid: Array<{ name: string; error: string }> } => {
    const valid: File[] = [];
    const invalid: Array<{ name: string; error: string }> = [];

    for (const file of files) {
      const ext = '.' + file.name.split('.').pop()?.toLowerCase();
      if (!ALLOWED_EXTENSIONS.includes(ext)) {
        invalid.push({ name: file.name, error: `Extension '${ext}' not allowed` });
        continue;
      }
      if (file.size > MAX_FILE_SIZE) {
        invalid.push({ name: file.name, error: `File exceeds 10MB limit` });
        continue;
      }
      valid.push(file);
    }

    return { valid, invalid };
  }, []);

  const handleUpload = useCallback(async (files: File[]) => {
    setError(null);
    setUploadResult(null);

    const { valid, invalid } = validateFiles(files);
    if (valid.length === 0 && invalid.length > 0) {
      setError(`No valid files to upload. ${invalid.map(f => `${f.name}: ${f.error}`).join('; ')}`);
      return;
    }

    try {
      const result = await uploadFiles(projectName, valid);
      setUploadResult({
        uploaded: result.uploaded,
        failed: [...invalid, ...result.failed],
        indexed: result.indexed_chunks,
      });
      // Clear result after 5 seconds
      setTimeout(() => setUploadResult(null), 5000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    }
  }, [projectName, uploadFiles, validateFiles]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleUpload(files);
    }
  }, [handleUpload]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      handleUpload(files);
    }
    // Reset input so same file can be selected again
    e.target.value = '';
  }, [handleUpload]);

  const handleDelete = useCallback(async (filename: string) => {
    setDeletingFile(filename);
    try {
      await deleteFile(projectName, filename);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Delete failed');
    } finally {
      setDeletingFile(null);
    }
  }, [projectName, deleteFile]);

  const isTextEditable = useCallback((filename: string) => {
    const ext = '.' + filename.split('.').pop()?.toLowerCase();
    return TEXT_EDITABLE_EXTENSIONS.includes(ext);
  }, []);

  const handleFileClick = useCallback((filename: string) => {
    if (isTextEditable(filename)) {
      openFileEditor(projectName, filename);
    } else {
      downloadFile(projectName, filename);
    }
  }, [projectName, isTextEditable, openFileEditor, downloadFile]);

  const handleDownload = useCallback((filename: string) => {
    downloadFile(projectName, filename);
  }, [projectName, downloadFile]);

  return (
    <div className="space-y-3">
      {/* Drop zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`
          border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors
          ${dragOver
            ? 'border-[var(--color-primary)] bg-[var(--color-primary)]/10'
            : 'border-[var(--color-border)] hover:border-[var(--color-primary)]/50'
          }
          ${uploading ? 'opacity-50 cursor-wait' : ''}
        `}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={ALLOWED_EXTENSIONS.join(',')}
          onChange={handleFileSelect}
          className="hidden"
          disabled={uploading}
        />
        {uploading ? (
          <div className="flex items-center justify-center gap-2 text-[var(--color-text-secondary)]">
            <LoaderIcon className="w-5 h-5 animate-spin" />
            <span>Uploading...</span>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-1">
            <UploadIcon className="w-6 h-6 text-[var(--color-text-secondary)]" />
            <span className="text-sm text-[var(--color-text-secondary)]">
              Drop files here or click to browse
            </span>
            <span className="text-xs text-[var(--color-text-secondary)]/70">
              {ALLOWED_EXTENSIONS.slice(0, 6).join(', ')}...
            </span>
          </div>
        )}
      </div>

      {/* Upload result */}
      {uploadResult && (
        <div className={`text-sm rounded-lg p-2 ${uploadResult.failed.length > 0 ? 'bg-yellow-500/10 border border-yellow-500/20' : 'bg-green-500/10 border border-green-500/20'}`}>
          {uploadResult.uploaded.length > 0 && (
            <p className="text-green-400 flex items-center gap-1">
              <CheckIcon className="w-4 h-4" />
              Uploaded {uploadResult.uploaded.length} file(s), indexed {uploadResult.indexed} chunks
            </p>
          )}
          {uploadResult.failed.length > 0 && (
            <p className="text-yellow-400 flex items-center gap-1 mt-1">
              <AlertIcon className="w-4 h-4" />
              {uploadResult.failed.length} failed: {uploadResult.failed.map(f => f.name).join(', ')}
            </p>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="text-sm rounded-lg p-2 bg-red-500/10 border border-red-500/20 text-red-400 flex items-center gap-1">
          <AlertIcon className="w-4 h-4 flex-shrink-0" />
          {error}
        </div>
      )}

      {/* File list */}
      {filesLoading ? (
        <div className="flex items-center justify-center py-4 text-[var(--color-text-secondary)]">
          <LoaderIcon className="w-4 h-4 animate-spin mr-2" />
          Loading files...
        </div>
      ) : projectFiles.length > 0 ? (
        <div className="border border-[var(--color-border)] rounded-lg divide-y divide-[var(--color-border)]">
          {projectFiles.map((file) => (
            <div
              key={file.name}
              className="flex items-center gap-2 px-3 py-2 hover:bg-[var(--color-surface)]"
            >
              <FileIcon className="w-4 h-4 text-[var(--color-text-secondary)] flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <button
                  onClick={() => handleFileClick(file.name)}
                  className="text-sm truncate text-left hover:text-[var(--color-primary)] transition-colors block w-full"
                  title={isTextEditable(file.name) ? `Edit ${file.name}` : `Download ${file.name}`}
                >
                  {file.name}
                </button>
                <p className="text-xs text-[var(--color-text-secondary)]">
                  {formatFileSize(file.size)} &middot; {formatDate(file.modified)}
                </p>
              </div>
              {file.indexed && (
                <span className="text-xs px-1.5 py-0.5 rounded bg-green-500/20 text-green-400 flex-shrink-0">
                  indexed
                </span>
              )}
              {isTextEditable(file.name) && (
                <button
                  onClick={() => openFileEditor(projectName, file.name)}
                  className="p-1 hover:bg-[var(--color-primary)]/20 rounded transition-colors text-[var(--color-text-secondary)] hover:text-[var(--color-primary)] flex-shrink-0"
                  title="Edit file"
                >
                  <EditIcon className="w-4 h-4" />
                </button>
              )}
              <button
                onClick={() => handleDownload(file.name)}
                className="p-1 hover:bg-[var(--color-primary)]/20 rounded transition-colors text-[var(--color-text-secondary)] hover:text-[var(--color-primary)] flex-shrink-0"
                title="Download file"
              >
                <DownloadIcon className="w-4 h-4" />
              </button>
              <button
                onClick={() => handleDelete(file.name)}
                disabled={deletingFile === file.name}
                className="p-1 hover:bg-red-500/20 rounded transition-colors text-[var(--color-text-secondary)] hover:text-red-400 flex-shrink-0"
                title="Delete file"
              >
                {deletingFile === file.name ? (
                  <LoaderIcon className="w-4 h-4 animate-spin" />
                ) : (
                  <TrashIcon className="w-4 h-4" />
                )}
              </button>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-[var(--color-text-secondary)] text-center py-2">
          No files uploaded yet
        </p>
      )}
    </div>
  );
}
