import { useState, useEffect, useCallback } from 'react';
import { useProjectStore } from '../../stores/projectStore';
import { CloseIcon, LoaderIcon, CheckIcon, ChevronDownIcon } from '../common/icons';
import { FileUploadZone } from './FileUploadZone';

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
  const [error, setError] = useState<string | null>(null);

  const [expandedFields, setExpandedFields] = useState<Record<string, boolean>>({});

  const toggleExpand = useCallback((field: string) => {
    setExpandedFields((prev) => ({ ...prev, [field]: !prev[field] }));
  }, []);

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
    setError(null);

    try {
      const result = await indexProject(editingProject);
      // Build result message
      let msg = `Indexed ${result.indexed_chunks || 0} chunks from ${(result.files || []).length} new files`;
      if (result.skipped && result.skipped > 0) {
        msg += ` (${result.skipped} unchanged)`;
      }
      if (result.synced_to_vps) {
        msg += ' - Synced to VPS';
      } else if (result.sync_error) {
        msg += ` - VPS sync failed: ${result.sync_error}`;
      }
      setIndexResult(msg);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to index files');
    } finally {
      setIndexing(false);
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
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm">System Prompt</label>
              {systemPrompt && (
                <button
                  type="button"
                  onClick={() => toggleExpand('systemPrompt')}
                  className="flex items-center gap-1 text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-text)] transition-colors"
                >
                  {expandedFields.systemPrompt ? 'Collapse' : 'Expand'}
                  <ChevronDownIcon className={`w-3 h-3 transition-transform ${expandedFields.systemPrompt ? 'rotate-180' : ''}`} />
                </button>
              )}
            </div>
            <textarea
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              placeholder="Custom system instructions for this project..."
              rows={expandedFields.systemPrompt ? 12 : 3}
              className="w-full px-3 py-2 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg text-[var(--color-text)] placeholder-[var(--color-text-secondary)] resize-y focus:outline-none focus:border-[var(--color-primary)] transition-all"
            />
          </div>

          {/* Additional instructions */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm">Additional Instructions</label>
              {instructions && (
                <button
                  type="button"
                  onClick={() => toggleExpand('instructions')}
                  className="flex items-center gap-1 text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-text)] transition-colors"
                >
                  {expandedFields.instructions ? 'Collapse' : 'Expand'}
                  <ChevronDownIcon className={`w-3 h-3 transition-transform ${expandedFields.instructions ? 'rotate-180' : ''}`} />
                </button>
              )}
            </div>
            <textarea
              value={instructions}
              onChange={(e) => setInstructions(e.target.value)}
              placeholder="Extra context or instructions..."
              rows={expandedFields.instructions ? 12 : 2}
              className="w-full px-3 py-2 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg text-[var(--color-text)] placeholder-[var(--color-text-secondary)] resize-y focus:outline-none focus:border-[var(--color-primary)] transition-all"
            />
          </div>

          {/* Allowed paths */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm">Allowed File Paths</label>
              {allowedPaths && (
                <button
                  type="button"
                  onClick={() => toggleExpand('allowedPaths')}
                  className="flex items-center gap-1 text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-text)] transition-colors"
                >
                  {expandedFields.allowedPaths ? 'Collapse' : 'Expand'}
                  <ChevronDownIcon className={`w-3 h-3 transition-transform ${expandedFields.allowedPaths ? 'rotate-180' : ''}`} />
                </button>
              )}
            </div>
            <textarea
              value={allowedPaths}
              onChange={(e) => setAllowedPaths(e.target.value)}
              placeholder="One path per line, e.g.:&#10;C:\Projects\my-project&#10;D:\Documents\specs"
              rows={expandedFields.allowedPaths ? 10 : 3}
              className="w-full px-3 py-2 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg text-[var(--color-text)] placeholder-[var(--color-text-secondary)] resize-y focus:outline-none focus:border-[var(--color-primary)] font-mono text-sm transition-all"
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
            </div>
          )}

          {/* KB File Upload (only for editing) */}
          {isEditing && editingProject && (
            <div className="border-t border-[var(--color-border)] pt-4 mt-4">
              <label className="block text-sm mb-2 font-medium">
                Knowledge Base Documents
              </label>
              <p className="text-xs text-[var(--color-text-secondary)] mb-3">
                Upload files directly to this project's KB. They will be auto-indexed.
              </p>
              <FileUploadZone projectName={editingProject} />
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
