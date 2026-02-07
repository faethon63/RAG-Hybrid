import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Project, ProjectConfig, IndexResponse, ProjectFile, UploadFilesResponse } from '../types/api';
import { api } from '../api/client';

interface ProjectState {
  // Projects list
  projects: Project[];
  projectsLoading: boolean;

  // Current project
  currentProject: string | null;
  currentProjectConfig: ProjectConfig | null;

  // UI state
  showProjectForm: boolean;
  editingProject: string | null;

  // Recent projects (for quick access)
  recentProjects: string[];

  // Project KB files
  projectFiles: ProjectFile[];
  filesLoading: boolean;
  uploading: boolean;

  // Actions
  setCurrentProject: (name: string | null) => void;
  setShowProjectForm: (show: boolean) => void;
  setEditingProject: (name: string | null) => void;

  // API operations
  loadProjects: () => Promise<void>;
  loadProjectConfig: (name: string) => Promise<void>;
  createProject: (data: {
    name: string;
    description?: string;
    system_prompt?: string;
    instructions?: string;
    allowed_paths?: string[];
  }) => Promise<void>;
  updateProject: (name: string, config: Partial<ProjectConfig>) => Promise<void>;
  deleteProject: (name: string) => Promise<void>;
  indexProject: (name: string) => Promise<IndexResponse>;

  // Project KB file operations
  loadProjectFiles: (name: string) => Promise<void>;
  uploadFiles: (name: string, files: File[]) => Promise<UploadFilesResponse>;
  deleteFile: (name: string, filename: string) => Promise<void>;
}

export const useProjectStore = create<ProjectState>()(
  persist(
    (set, get) => ({
      projects: [],
      projectsLoading: false,
      currentProject: null,
      currentProjectConfig: null,
      showProjectForm: false,
      editingProject: null,
      recentProjects: [],
      projectFiles: [],
      filesLoading: false,
      uploading: false,

      setCurrentProject: (name) => {
        set({ currentProject: name, currentProjectConfig: null });
        if (name) {
          // Add to recent projects
          const recent = get().recentProjects.filter((p) => p !== name);
          set({ recentProjects: [name, ...recent].slice(0, 5) });
          // Load config
          get().loadProjectConfig(name);
        }
      },

      setShowProjectForm: (show) => set({ showProjectForm: show }),
      setEditingProject: (name) => set({ editingProject: name }),

      loadProjects: async () => {
        set({ projectsLoading: true });
        try {
          const response = await api.listProjects();
          set({ projects: response.projects, projectsLoading: false });
        } catch (err) {
          console.error('Failed to load projects:', err);
          set({ projectsLoading: false });
        }
      },

      loadProjectConfig: async (name) => {
        try {
          const response = await api.getProject(name);
          set({ currentProjectConfig: response.config });
        } catch (err) {
          console.error('Failed to load project config:', err);
        }
      },

      createProject: async (data) => {
        try {
          await api.createProject(data);
          await get().loadProjects();
          set({ showProjectForm: false });
        } catch (err) {
          console.error('Failed to create project:', err);
          throw err;
        }
      },

      updateProject: async (name, config) => {
        try {
          const response = await api.updateProject(name, config);
          set({ currentProjectConfig: response.config, editingProject: null });
          await get().loadProjects();
        } catch (err) {
          console.error('Failed to update project:', err);
          throw err;
        }
      },

      deleteProject: async (name) => {
        try {
          await api.deleteProject(name);
          // If we deleted the current project, clear selection
          if (get().currentProject === name) {
            set({ currentProject: null, currentProjectConfig: null });
          }
          // Remove from recent projects
          set({ recentProjects: get().recentProjects.filter((p) => p !== name) });
          await get().loadProjects();
        } catch (err) {
          console.error('Failed to delete project:', err);
          throw err;
        }
      },

      indexProject: async (name) => {
        try {
          const response = await api.indexProject(name);
          return response;
        } catch (err) {
          console.error('Failed to index project:', err);
          throw err;
        }
      },

      loadProjectFiles: async (name) => {
        set({ filesLoading: true });
        try {
          const response = await api.listProjectFiles(name);
          set({ projectFiles: response.files, filesLoading: false });
        } catch (err) {
          console.error('Failed to load project files:', err);
          set({ projectFiles: [], filesLoading: false });
        }
      },

      uploadFiles: async (name, files) => {
        set({ uploading: true });
        try {
          const response = await api.uploadProjectFiles(name, files);
          // Reload files list after upload
          await get().loadProjectFiles(name);
          set({ uploading: false });
          return response;
        } catch (err) {
          console.error('Failed to upload files:', err);
          set({ uploading: false });
          throw err;
        }
      },

      deleteFile: async (name, filename) => {
        try {
          await api.deleteProjectFile(name, filename);
          // Reload files list after delete
          await get().loadProjectFiles(name);
        } catch (err) {
          console.error('Failed to delete file:', err);
          throw err;
        }
      },
    }),
    {
      name: 'rag-projects',
      partialize: (state) => ({
        currentProject: state.currentProject,
        recentProjects: state.recentProjects,
      }),
    }
  )
);
