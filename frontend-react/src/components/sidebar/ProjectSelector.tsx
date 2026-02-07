import { useEffect } from 'react';
import { useProjectStore } from '../../stores/projectStore';
import { useChatStore } from '../../stores/chatStore';
import {
  PlusIcon,
  EditIcon,
  TrashIcon,
  LoaderIcon,
} from '../common/icons';
import clsx from 'clsx';

export function ProjectSelector() {
  const projects = useProjectStore((s) => s.projects);
  const projectsLoading = useProjectStore((s) => s.projectsLoading);
  const currentProject = useProjectStore((s) => s.currentProject);
  const recentProjects = useProjectStore((s) => s.recentProjects);
  const loadProjects = useProjectStore((s) => s.loadProjects);
  const setCurrentProject = useProjectStore((s) => s.setCurrentProject);
  const setShowProjectForm = useProjectStore((s) => s.setShowProjectForm);
  const setEditingProject = useProjectStore((s) => s.setEditingProject);
  const deleteProject = useProjectStore((s) => s.deleteProject);
  const newChat = useChatStore((s) => s.newChat);
  const loadChats = useChatStore((s) => s.loadChats);

  useEffect(() => {
    loadProjects();
  }, [loadProjects]);

  const handleSelectProject = (name: string | null) => {
    // Click the active project again to deselect (show all chats)
    if (name === currentProject) {
      setCurrentProject(null);
      newChat();
      loadChats(null);
      return;
    }
    setCurrentProject(name);
    newChat();
    loadChats(name);
  };

  const handleCreateNew = () => {
    setShowProjectForm(true);
  };

  const handleEditProject = (e: React.MouseEvent, name: string) => {
    e.stopPropagation();
    setEditingProject(name);
  };

  const handleDeleteProject = async (e: React.MouseEvent, name: string) => {
    e.stopPropagation();
    if (!confirm(`Delete project "${name}"? This will remove all its config, KB files, and indexed data.`)) {
      return;
    }
    try {
      await deleteProject(name);
      // If we were viewing this project, switch to all chats
      if (currentProject === name) {
        newChat();
        loadChats(null);
      }
    } catch (err) {
      console.error('Failed to delete project:', err);
    }
  };

  // Sort projects: recent first, then alphabetical
  const sortedProjects = [...projects].sort((a, b) => {
    const aRecent = recentProjects.indexOf(a.name);
    const bRecent = recentProjects.indexOf(b.name);
    if (aRecent !== -1 && bRecent !== -1) return aRecent - bRecent;
    if (aRecent !== -1) return -1;
    if (bRecent !== -1) return 1;
    return a.name.localeCompare(b.name);
  });

  return (
    <div className="space-y-1">
      {/* Header - clickable to deselect project */}
      <div className="flex items-center justify-between mb-1">
        <button
          onClick={() => { if (currentProject) { setCurrentProject(null); newChat(); loadChats(null); } }}
          className={clsx(
            'text-xs font-medium uppercase tracking-wide transition-colors',
            currentProject
              ? 'text-[var(--color-text-secondary)] hover:text-[var(--color-text)] cursor-pointer'
              : 'text-[var(--color-text-secondary)] cursor-default'
          )}
          title={currentProject ? 'Show all chats' : undefined}
        >
          Projects
        </button>
        <button
          onClick={handleCreateNew}
          className="p-0.5 hover:bg-[var(--color-surface-hover)] rounded transition-colors"
          title="New project"
        >
          <PlusIcon className="w-3.5 h-3.5 text-[var(--color-text-secondary)]" />
        </button>
      </div>

      {/* Project list */}
      {projectsLoading ? (
        <div className="flex items-center justify-center py-3">
          <LoaderIcon className="w-4 h-4 animate-spin text-[var(--color-text-secondary)]" />
        </div>
      ) : sortedProjects.length === 0 ? (
        <div className="text-xs text-[var(--color-text-secondary)] py-1 px-2">
          No projects yet
        </div>
      ) : (
        sortedProjects.map((project) => {
          const isActive = currentProject === project.name;
          return (
            <div
              key={project.name}
              onClick={() => handleSelectProject(project.name)}
              className={clsx(
                'group flex items-center gap-2 px-2 py-1.5 rounded-md cursor-pointer transition-all text-sm',
                isActive
                  ? 'bg-[var(--color-primary)]/15 text-[var(--color-primary)] border-l-2 border-[var(--color-primary)]'
                  : 'text-[var(--color-text-secondary)] hover:bg-[var(--color-surface-hover)] hover:text-[var(--color-text)]'
              )}
            >
              <span className="flex-1 truncate">{project.name}</span>
              <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  onClick={(e) => handleEditProject(e, project.name)}
                  className="p-0.5 hover:bg-[var(--color-border)] rounded transition-colors"
                  title="Edit project"
                >
                  <EditIcon className="w-3 h-3" />
                </button>
                <button
                  onClick={(e) => handleDeleteProject(e, project.name)}
                  className="p-0.5 hover:bg-red-500/20 rounded transition-colors text-[var(--color-text-secondary)] hover:text-red-400"
                  title="Delete project"
                >
                  <TrashIcon className="w-3 h-3" />
                </button>
              </div>
            </div>
          );
        })
      )}
    </div>
  );
}
