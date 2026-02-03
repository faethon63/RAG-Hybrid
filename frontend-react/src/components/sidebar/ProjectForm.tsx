import { useState, useEffect } from 'react';
import { useProjectStore } from '../../stores/projectStore';
import { CloseIcon, LoaderIcon, CheckIcon } from '../common/icons';
import { api } from '../../api/client';

export function ProjectForm() {
  const showProjectForm = useProjectStore((s) => s.showProjectForm);
  const setShowProjectForm = useProjectStore((s) => s.setShowProjectForm);
  const editingProject = useProjectStore((s) => s.editingProject);
  const setEditingProject = useProjectStore((s) => s.setEditingProject);
  const currentProjectConfig = useProjectStore((s) => s.currentProjectConfig);
  const createProject = useProjectStore((s) => s.createProject);
  const updateProject = useProjectStore((s) => s.updateProject);
  const indexProject = useProjectStore((s) => s.indexProject);
  const loadProjectConfig = useProjectStore((s) => s.loadProjectConfig);

  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [instructions, setInstructions] = useState('');
  const [allowedPaths, setAllowedPaths] = useState('');
  const [saving, setSaving] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [indexResult, setIndexResult] = useState<string | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [syncResult, setSyncResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const isEditing = !!editingProject;
  const isOpen = showProjectForm || isEditing;

  useEffect(() => {
    if (editingProject) {
      loadProjectConfig(editingProject);
    }
  }, [editingProject, loadProjectConfig]);

  useEffect(() => {
    if (isEditing && currentProjectConfig) {
      setName(editingProject);
      setDescription(currentProjectConfig.description || '');
      setSystemPrompt(currentProjectConfig.system_prompt || '');
      setInstructions(currentProjectConfig.instructions || '');
      setAllowedPaths(currentProjectConfig.allowed_paths?.join('\n') || '');
    } else if (!isEditing) {
      setName('');
      setDescription('');
      setSystemPrompt('');
      setInstructions('');
      setAllowedPaths('');
    }
  }, [isEditing, editingProject, currentProjectConfig]);

  const handleClose = () => {
    setShowProjectForm(false);
    setEditingProject(null);
    setError(null);
    setIndexResult(null);
    setSyncResult(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setError('Project name is required');
      return;
    }

    setSaving(true);
    setError(null);

    try {
      const paths = allowedPaths
        .split('\n')
        .map((p) => p.trim())
        .filter(Boolean);

      if (isEditing) {
        await updateProject(editingProject, {
          description,
          system_prompt: systemPrompt,
          instructions,
          allowed_paths: paths,
        });
      } else {
        await createProject({
          name: name.trim(),
          description,
          system_prompt: systemPrompt,
          instructions,
          allowed_paths: paths,
        });
      }
      handleClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save project');
    } finally {
      setSaving(false);
    }
  };

  const handleIndex = async () => {
    if (!editingProject) return;

    setIndexing(true);
    setIndexResult(null);
    setSyncResult(null);
    setError(null);

    try {
      const result = await indexProject(editingProject);
      setIndexResult(
        `Indexed ${result.indexed_chunks} chunks from ${result.files.length} files`
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to index files');
    } finally {
      setIndexing(false);
    }
  };

  const handleSyncToVps = async () => {
    setSyncing(true);
    setSyncResult(null);
    setError(null);

    try {
      const result = await api.syncPushToVps();
      setSyncResult(result.message || 'Synced successfully');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to sync to VPS');
    } finally {
      setSyncing(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50" onClick={handleClose} />

      {/* Panel */}
      <div className="relative w-full max-w-lg bg-[var(--color-background)] border border-[var(--color-border)] rounded-xl shadow-2xl max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-[var(--color-background)] border-b border-[var(--color-border)] px-6 py-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold">
            {isEditing ? 'Edit Project' : 'New Project'}
          </h2>
          <button
            onClick={handleClose}
            className="p-1 hover:bg-[var(--color-surface)] rounded transition-colors"
          >
            <CloseIcon className="w-5 h-5" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {/* Name */}
          <div>
            <label className="block text-sm mb-2">
              Project Name <span className="text-red-400">*</span>
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={isEditing}
              placeholder="my-project"
              className="w-full px-3 py-2 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg text-[var(--color-text)] placeholder-[var(--color-text-secondary)] focus:outline-none focus:border-[var(--color-primary)] disabled:opacity-50"
            />
          </div>

          {/* Description */}
          <div>
            <label className="block text-sm mb-2">Description</label>
            <input
              type="text"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Brief description of this project"
              className="w-full px-3 py-2 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg text-[var(--color-text)] placeholder-[var(--color-text-secondary)] focus:outline-none focus:border-[var(--color-primary)]"
            />
          </div>

          {/* System prompt */}
          <div>
            <label className="block text-sm mb-2">System Prompt</label>
            <textarea
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              placeholder="Custom system instructions for this project..."
              rows={3}
              className="w-full px-3 py-2 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg text-[var(--color-text)] placeholder-[var(--color-text-secondary)] resize-none focus:outline-none focus:border-[var(--color-primary)]"
            />
          </div>

          {/* Additional instructions */}
          <div>
            <label className="block text-sm mb-2">Additional Instructions</label>
            <textarea
              value={instructions}
              onChange={(e) => setInstructions(e.target.value)}
              placeholder="Extra context or instructions..."
              rows={2}
              className="w-full px-3 py-2 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg text-[var(--color-text)] placeholder-[var(--color-text-secondary)] resize-none focus:outline-none focus:border-[var(--color-primary)]"
            />
          </div>

          {/* Allowed paths */}
          <div>
            <label className="block text-sm mb-2">Allowed File Paths</label>
            <textarea
              value={allowedPaths}
              onChange={(e) => setAllowedPaths(e.target.value)}
              placeholder="One path per line, e.g.:&#10;C:\Projects\my-project&#10;D:\Documents\specs"
              rows={3}
              className="w-full px-3 py-2 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg text-[var(--color-text)] placeholder-[var(--color-text-secondary)] resize-none focus:outline-none focus:border-[var(--color-primary)] font-mono text-sm"
            />
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">
              Directories containing files to index for this project
            </p>
          </div>

          {/* Index button (only for editing) */}
          {isEditing && (
            <div>
              <button
                type="button"
                onClick={handleIndex}
                disabled={indexing}
                className="px-4 py-2 text-sm bg-[var(--color-surface)] hover:bg-[var(--color-surface-hover)] border border-[var(--color-border)] rounded-lg transition-colors flex items-center gap-2"
              >
                {indexing ? (
                  <>
                    <LoaderIcon className="w-4 h-4 animate-spin" />
                    <span>Indexing...</span>
                  </>
                ) : (
                  <span>Index Files from Paths</span>
                )}
              </button>
              {indexResult && (
                <p className="text-sm text-green-400 mt-2 flex items-center gap-1">
                  <CheckIcon className="w-4 h-4" />
                  {indexResult}
                </p>
              )}
              {/* Sync to VPS button - appears after successful indexing */}
              {indexResult && (
                <button
                  type="button"
                  onClick={handleSyncToVps}
                  disabled={syncing}
                  className="mt-2 px-4 py-2 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center gap-2"
                >
                  {syncing ? (
                    <>
                      <LoaderIcon className="w-4 h-4 animate-spin" />
                      <span>Syncing to VPS...</span>
                    </>
                  ) : (
                    <span>Sync to VPS</span>
                  )}
                </button>
              )}
              {syncResult && (
                <p className="text-sm text-blue-400 mt-2 flex items-center gap-1">
                  <CheckIcon className="w-4 h-4" />
                  {syncResult}
                </p>
              )}
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm">
              {error}
            </div>
          )}
        </form>

        {/* Footer */}
        <div className="sticky bottom-0 bg-[var(--color-background)] border-t border-[var(--color-border)] px-6 py-4 flex justify-end gap-3">
          <button
            type="button"
            onClick={handleClose}
            className="px-4 py-2 text-sm hover:bg-[var(--color-surface)] rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={saving}
            className="px-4 py-2 text-sm bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] text-white rounded-lg transition-colors flex items-center gap-2"
          >
            {saving ? (
              <>
                <LoaderIcon className="w-4 h-4 animate-spin" />
                <span>Saving...</span>
              </>
            ) : (
              <span>{isEditing ? 'Save Changes' : 'Create Project'}</span>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
