#!/usr/bin/env python3
"""
TogetherOS RAG System - Claude Code Automation
This script enables Claude Code to search your system, identify what's installed,
and automatically install/configure everything needed.

Usage:
    python claude_code_setup.py --mode check    # Check what's installed
    python claude_code_setup.py --mode install  # Install missing components
    python claude_code_setup.py --mode deploy   # Deploy to VPS
    python claude_code_setup.py --mode vibe     # Full auto mode (vibe coding)
"""

import os
import sys
import subprocess
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import platform

class ClaudeCodeRAGSetup:
    """Automated RAG system setup for Claude Code"""
    
    def __init__(self):
        self.os_type = platform.system()
        self.is_windows = self.os_type == "Windows"
        self.project_root = Path("G:/AI-Project") if self.is_windows else Path.home() / "AI-Project"
        self.rag_dir = self.project_root / "RAG-Hybrid"
        self.inventory = {}
        
    def search_drive(self, search_terms: List[str]) -> Dict[str, List[str]]:
        """Search G: drive for existing installations"""
        print(f"🔍 Searching drive for: {', '.join(search_terms)}")
        
        found = {}
        search_paths = [
            Path("G:/"),
            Path("G:/AI-Project"),
            Path("G:/Coopeverything"),
            Path("C:/Program Files"),
            Path.home()
        ]
        
        for term in search_terms:
            found[term] = []
            for search_path in search_paths:
                if not search_path.exists():
                    continue
                
                try:
                    # Search for directories matching term
                    matches = list(search_path.rglob(f"*{term}*"))[:10]  # Limit to 10 results
                    found[term].extend([str(m) for m in matches if m.is_dir()])
                except (PermissionError, OSError):
                    continue
        
        return found
    
    def check_python_packages(self) -> Tuple[List[str], List[str]]:
        """Check which Python packages are installed"""
        print("\n📚 Checking Python packages...")
        
        required = [
            'langchain',
            'chromadb',
            'streamlit',
            'fastapi',
            'sentence-transformers',
            'httpx',
            'uvicorn',
            'python-dotenv',
            'bcrypt',
            'pyjwt'
        ]
        
        installed = []
        missing = []
        
        for package in required:
            try:
                __import__(package.replace('-', '_'))
                installed.append(package)
                print(f"   ✓ {package}")
            except ImportError:
                missing.append(package)
                print(f"   ✗ {package}")
        
        return installed, missing
    
    def check_system_tools(self) -> Dict[str, bool]:
        """Check system tools (Ollama, Git, UV, Node)"""
        print("\n🔧 Checking system tools...")
        
        tools = {
            'ollama': ['ollama', '--version'],
            'git': ['git', '--version'],
            'uv': ['uv', '--version'],
            'node': ['node', '--version']
        }
        
        status = {}
        for tool, cmd in tools.items():
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=5)
                if result.returncode == 0:
                    status[tool] = True
                    version = result.stdout.decode().strip()
                    print(f"   ✓ {tool}: {version}")
                else:
                    status[tool] = False
                    print(f"   ✗ {tool}: not found")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                status[tool] = False
                print(f"   ✗ {tool}: not found")
        
        return status
    
    def generate_install_commands(self, missing_packages: List[str], missing_tools: List[str]) -> List[str]:
        """Generate installation commands"""
        print("\n📝 Generating installation commands...")
        
        commands = []
        
        # Python packages
        if missing_packages:
            if self.is_windows:
                uv_path = "G:\\AI-Project\\Python\\Scripts\\uv.exe"
                if Path(uv_path).exists():
                    cmd = f"{uv_path} pip install {' '.join(missing_packages)}"
                else:
                    cmd = f"pip install {' '.join(missing_packages)}"
            else:
                cmd = f"pip install {' '.join(missing_packages)}"
            
            commands.append(("python_packages", cmd))
        
        # Ollama
        if 'ollama' in missing_tools:
            if self.is_windows:
                commands.append(("ollama", "winget install Ollama.Ollama"))
            else:
                commands.append(("ollama", "curl -fsSL https://ollama.com/install.sh | sh"))
        
        # UV package manager
        if 'uv' in missing_tools:
            if self.is_windows:
                commands.append(("uv", "pip install uv"))
            else:
                commands.append(("uv", "curl -LsSf https://astral.sh/uv/install.sh | sh"))
        
        return commands
    
    def execute_vibe_mode(self):
        """Full automated setup - vibe coding mode"""
        print("\n" + "="*60)
        print("🎵 VIBE MODE ACTIVATED - Automated Setup")
        print("="*60)
        
        # Step 1: Search drive
        search_results = self.search_drive(['ollama', 'python', 'chromadb', 'langchain'])
        print(f"\n📂 Found on drive: {json.dumps(search_results, indent=2)}")
        
        # Step 2: Check what's installed
        installed_packages, missing_packages = self.check_python_packages()
        tool_status = self.check_system_tools()
        missing_tools = [tool for tool, installed in tool_status.items() if not installed]
        
        # Step 3: Generate install commands
        commands = self.generate_install_commands(missing_packages, missing_tools)
        
        if not commands:
            print("\n✨ Everything is already installed! Ready to go.")
            return True
        
        # Step 4: Execute installations
        print("\n🚀 Installing missing components...")
        
        for name, cmd in commands:
            print(f"\n▶️  Installing {name}...")
            print(f"   Command: {cmd}")
            
            # Ask for confirmation
            response = input("   Execute? (y/n/skip): ").lower()
            
            if response == 'y':
                try:
                    if self.is_windows and cmd.startswith("winget"):
                        # Use PowerShell for winget
                        result = subprocess.run(
                            ["powershell", "-Command", cmd],
                            capture_output=True,
                            text=True,
                            timeout=300
                        )
                    else:
                        result = subprocess.run(
                            cmd,
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=300
                        )
                    
                    if result.returncode == 0:
                        print(f"   ✅ {name} installed successfully")
                    else:
                        print(f"   ❌ {name} installation failed:")
                        print(f"   {result.stderr}")
                except subprocess.TimeoutExpired:
                    print(f"   ⏰ {name} installation timed out")
                except Exception as e:
                    print(f"   ❌ Error installing {name}: {e}")
            elif response == 'skip':
                print(f"   ⏭️  Skipped {name}")
            else:
                print(f"   🛑 Cancelled {name} installation")
        
        print("\n✅ Vibe mode complete!")
        return True
    
    def create_directory_structure(self):
        """Create RAG system directory structure"""
        print("\n📁 Creating directory structure...")
        
        dirs = [
            self.rag_dir,
            self.rag_dir / "backend",
            self.rag_dir / "frontend",
            self.rag_dir / "scripts",
            self.rag_dir / "config",
            self.rag_dir / "data" / "chromadb",
            self.rag_dir / "data" / "documents",
            self.rag_dir / "data" / "project-kb",
            self.rag_dir / "logs"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✓ {dir_path.relative_to(self.project_root)}")
        
        return True
    
    def copy_template_files(self):
        """Copy template files to RAG system directory"""
        print("\n📄 Setting up configuration files...")
        
        # This would copy the files we created in /tmp
        # In practice, Claude Code will have these files ready
        
        templates = {
            'config/.env': '/tmp/env.template',
            'backend/main.py': '/tmp/backend_main.py',
            'scripts/setup.py': '/tmp/rag-system-setup.py'
        }
        
        for dest, src in templates.items():
            dest_path = self.rag_dir / dest
            src_path = Path(src)
            
            if src_path.exists():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_text(src_path.read_text())
                print(f"   ✓ {dest}")
            else:
                print(f"   ⚠️  Template not found: {src}")
        
        return True
    
    def test_installation(self):
        """Test that everything is working"""
        print("\n🧪 Testing installation...")
        
        tests = [
            ("Python imports", lambda: __import__('langchain')),
            ("ChromaDB", lambda: __import__('chromadb')),
            ("Ollama", lambda: subprocess.run(['ollama', 'list'], capture_output=True, timeout=5)),
        ]
        
        passed = 0
        failed = 0
        
        for name, test_fn in tests:
            try:
                test_fn()
                print(f"   ✅ {name}")
                passed += 1
            except Exception as e:
                print(f"   ❌ {name}: {e}")
                failed += 1
        
        print(f"\n📊 Tests: {passed} passed, {failed} failed")
        return failed == 0
    
    def print_next_steps(self):
        """Print what to do next"""
        print("\n" + "="*60)
        print("🎯 NEXT STEPS")
        print("="*60)
        
        print("\n1. Configure API keys:")
        print(f"   Edit: {self.rag_dir}/config/.env")
        print("   Add: ANTHROPIC_API_KEY, PERPLEXITY_API_KEY")
        
        print("\n2. Pull Ollama model:")
        print("   ollama pull qwen3:8b")
        
        print("\n3. Index your documents:")
        print(f"   python {self.rag_dir}/scripts/index_documents.py")
        
        print("\n4. Start the backend:")
        print(f"   cd {self.rag_dir}/backend")
        print("   python main.py")
        
        print("\n5. Start the frontend:")
        print(f"   cd {self.rag_dir}/frontend")
        print("   streamlit run app.py")
        
        print("\n6. Access the UI:")
        print("   http://localhost:8501")
        
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="TogetherOS RAG System - Claude Code Setup")
    parser.add_argument(
        '--mode',
        choices=['check', 'install', 'deploy', 'vibe'],
        default='check',
        help='Setup mode'
    )
    
    args = parser.parse_args()
    
    setup = ClaudeCodeRAGSetup()
    
    if args.mode == 'check':
        # Just check what's installed
        setup.check_python_packages()
        setup.check_system_tools()
        
    elif args.mode == 'install':
        # Install missing components (with prompts)
        installed, missing = setup.check_python_packages()
        tools = setup.check_system_tools()
        commands = setup.generate_install_commands(missing, [k for k, v in tools.items() if not v])
        
        print("\n📋 Installation plan:")
        for name, cmd in commands:
            print(f"   {name}: {cmd}")
        
    elif args.mode == 'deploy':
        # Deploy to VPS
        print("🚀 VPS deployment coming soon...")
        
    elif args.mode == 'vibe':
        # Full automated setup
        setup.execute_vibe_mode()
        setup.create_directory_structure()
        setup.copy_template_files()
        setup.test_installation()
        setup.print_next_steps()


if __name__ == "__main__":
    main()
