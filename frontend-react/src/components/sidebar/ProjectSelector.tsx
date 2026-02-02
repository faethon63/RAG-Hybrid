import { useEffect, useState } from 'react';
import { useProjectStore } from '../../stores/projectStore';
import { useChatStore } from '../../stores/chatStore';
import {
  ProjectIcon,
  ChevronDownIcon,
  PlusIcon,
  CloseIcon,
  LoaderIcon,
} from '../common/icons';
import clsx from 'clsx';

export function ProjectSelector() {
  const [isOpen, setIsOpen] = useState(false);
  const projects = useProjectStore((s) => s.projects);
  const projectsLoading = useProjectStore((s) => s.projectsLoading);
  const currentProject = useProjectStore((s) => s.currentProject);
  const recentProjects = useProjectStore((s) => s.recentProjects);
  const loadProjects = useProjectStore((s) => s.loadProjects);
  const setCurrentProject = useProjectStore((s) => s.setCurrentProject);
  const setShowProjectForm = useProjectStore((s) => s.setShowProjectForm);
  const newChat = useChatStore((s) => s.newChat);
  const loadChats = useChatStore((s) => s.loadChats);

  useEffect(() => {
    loadProjects();
  }, [loadProjects]);

  const handleSelectProject = (name: string | null) => {
    setCurrentProject(name);
    newChat();
    loadChats(name);
    setIsOpen(false);
  };

  const handleCreateNew = () => {
    setShowProjectForm(true);
    setIsOpen(false);
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
    <div className="relative">
      {/* Trigger button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={clsx(
          'w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-colors',
          'hover:bg-[var(--color-surface-hover)]',
          currentProject && 'bg-[var(--color-surface)]'
        )}
      >
        <ProjectIcon className="w-4 h-4 text-[var(--color-text-secondary)]" />
        <span className="flex-1 text-left text-sm truncate">
          {currentProject || 'All Projects'}
        </span>
        <ChevronDownIcon
          className={clsx(
            'w-4 h-4 text-[var(--color-text-secondary)] transition-transform',
            isOpen && 'rotate-180'
          )}
        />
      </button>

      {/* Dropdown */}
      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute left-0 right-0 mt-1 z-20 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg shadow-xl overflow-hidden">
            {/* All projects option */}
            <button
              onClick={() => handleSelectProject(null)}
              className={clsx(
                'w-full flex items-center gap-2 px-3 py-2 text-left text-sm transition-colors',
                'hover:bg-[var(--color-surface-hover)]',
                !currentProject && 'text-[var(--color-primary)]'
              )}
            >
              <span className="flex-1">All Projects</span>
              {!currentProject && <CloseIcon className="w-4 h-4" />}
            </button>

            <div className="border-t border-[var(--color-border)]" />

            {/* Project list */}
            {projectsLoading ? (
              <div className="flex items-center justify-center py-4">
                <LoaderIcon className="w-4 h-4 animate-spin" />
              </div>
            ) : sortedProjects.length === 0 ? (
              <div className="px-3 py-2 text-sm text-[var(--color-text-secondary)]">
                No projects yet
              </div>
            ) : (
              <div className="max-h-48 overflow-y-auto">
                {sortedProjects.map((project) => (
                  <button
                    key={project.name}
                    onClick={() => handleSelectProject(project.name)}
                    className={clsx(
                      'w-full flex items-center gap-2 px-3 py-2 text-left text-sm transition-colors',
                      'hover:bg-[var(--color-surface-hover)]',
                      currentProject === project.name &&
                        'text-[var(--color-primary)]'
                    )}
                  >
                    <ProjectIcon className="w-4 h-4" />
                    <span className="flex-1 truncate">{project.name}</span>
                  </button>
                ))}
              </div>
            )}

            <div className="border-t border-[var(--color-border)]" />

            {/* Create new */}
            <button
              onClick={handleCreateNew}
              className="w-full flex items-center gap-2 px-3 py-2 text-sm text-[var(--color-primary)] hover:bg-[var(--color-surface-hover)] transition-colors"
            >
              <PlusIcon className="w-4 h-4" />
              <span>New Project</span>
            </button>
          </div>
        </>
      )}
    </div>
  );
}
