#!/usr/bin/env python3
"""
TogetherOS RAG System - Master Setup Script
Checks existing installations and sets up missing components
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path

class RAGSystemSetup:
    def __init__(self):
        self.os_type = platform.system()
        self.is_windows = self.os_type == "Windows"
        self.python_version = sys.version_info
        self.project_root = Path("G:/AI-Project") if self.is_windows else Path.home() / "AI-Project"
        self.rag_dir = self.project_root / "RAG-Hybrid"
        self.inventory = {}
        
    def check_python(self):
        """Check Python version"""
        print("🐍 Checking Python...")
        version = f"{self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}"
        self.inventory['python'] = {
            'installed': True,
            'version': version,
            'path': sys.executable,
            'meets_requirement': self.python_version >= (3, 10)
        }
        print(f"   ✓ Python {version} found at {sys.executable}")
        if not self.inventory['python']['meets_requirement']:
            print("   ⚠️  Python 3.10+ recommended")
        return True
    
    def check_uv(self):
        """Check if UV package manager is installed"""
        print("\n📦 Checking UV package manager...")
        try:
            result = subprocess.run(['uv', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.inventory['uv'] = {
                    'installed': True,
                    'version': version,
                    'path': 'uv'
                }
                print(f"   ✓ UV found: {version}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        self.inventory['uv'] = {'installed': False}
        print("   ✗ UV not found")
        print("   💡 Install from: https://docs.astral.sh/uv/getting-started/installation/")
        return False
    
    def check_ollama(self):
        """Check if Ollama is installed"""
        print("\n🦙 Checking Ollama...")
        try:
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.inventory['ollama'] = {
                    'installed': True,
                    'version': version
                }
                print(f"   ✓ Ollama found: {version}")
                
                # Check installed models
                models_result = subprocess.run(['ollama', 'list'], 
                                             capture_output=True, text=True, timeout=10)
                if models_result.returncode == 0:
                    models = [line.split()[0] for line in models_result.stdout.strip().split('\n')[1:] 
                             if line.strip()]
                    self.inventory['ollama']['models'] = models
                    print(f"   📋 Installed models: {', '.join(models) if models else 'None'}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        self.inventory['ollama'] = {'installed': False}
        print("   ✗ Ollama not found")
        if self.is_windows:
            print("   💡 Install: winget install Ollama.Ollama")
        else:
            print("   💡 Install: curl -fsSL https://ollama.com/install.sh | sh")
        return False
    
    def check_git(self):
        """Check Git installation"""
        print("\n🌿 Checking Git...")
        try:
            result = subprocess.run(['git', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.inventory['git'] = {
                    'installed': True,
                    'version': version
                }
                print(f"   ✓ Git found: {version}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        self.inventory['git'] = {'installed': False}
        print("   ✗ Git not found")
        return False
    
    def check_node(self):
        """Check Node.js installation (for MCP servers)"""
        print("\n📗 Checking Node.js...")
        try:
            result = subprocess.run(['node', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.inventory['node'] = {
                    'installed': True,
                    'version': version
                }
                print(f"   ✓ Node.js found: {version}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        self.inventory['node'] = {'installed': False}
        print("   ✗ Node.js not found (optional for MCP servers)")
        return False
    
    def check_python_packages(self):
        """Check installed Python packages"""
        print("\n📚 Checking Python packages...")
        required_packages = {
            'langchain': 'langchain',
            'chromadb': 'chromadb',
            'streamlit': 'streamlit',
            'fastapi': 'fastapi',
            'sentence-transformers': 'sentence-transformers',
            'requests': 'requests',
            'python-dotenv': 'dotenv',
            'uvicorn': 'uvicorn',
            'httpx': 'httpx'
        }
        
        self.inventory['python_packages'] = {}
        missing_packages = []
        
        for package_name, import_name in required_packages.items():
            try:
                __import__(import_name)
                self.inventory['python_packages'][package_name] = {'installed': True}
                print(f"   ✓ {package_name}")
            except ImportError:
                self.inventory['python_packages'][package_name] = {'installed': False}
                print(f"   ✗ {package_name}")
                missing_packages.append(package_name)
        
        return len(missing_packages) == 0, missing_packages
    
    def check_directory_structure(self):
        """Check and create necessary directories"""
        print("\n📁 Checking directory structure...")
        
        dirs_to_check = [
            self.rag_dir,
            self.rag_dir / "data",
            self.rag_dir / "data" / "chromadb",
            self.rag_dir / "data" / "documents",
            self.rag_dir / "backend",
            self.rag_dir / "frontend",
            self.rag_dir / "scripts",
            self.rag_dir / "config",
            self.rag_dir / "logs"
        ]
        
        created = []
        existing = []
        
        for dir_path in dirs_to_check:
            if dir_path.exists():
                existing.append(dir_path.name)
            else:
                dir_path.mkdir(parents=True, exist_ok=True)
                created.append(dir_path.name)
        
        if existing:
            print(f"   ✓ Existing: {', '.join(existing)}")
        if created:
            print(f"   ➕ Created: {', '.join(created)}")
        
        self.inventory['directories'] = {
            'root': str(self.rag_dir),
            'existing': existing,
            'created': created
        }
        return True
    
    def generate_installation_script(self, missing_packages):
        """Generate installation commands for missing components"""
        print("\n🔧 Generating installation script...")
        
        install_script_path = self.rag_dir / "scripts" / "install_missing.sh"
        
        commands = [
            "#!/bin/bash",
            "# Auto-generated installation script",
            "echo '🚀 Installing missing components...'\n"
        ]
        
        # Ollama installation
        if not self.inventory.get('ollama', {}).get('installed'):
            if self.is_windows:
                commands.append("# Install Ollama on Windows")
                commands.append("winget install Ollama.Ollama")
            else:
                commands.append("# Install Ollama")
                commands.append("curl -fsSL https://ollama.com/install.sh | sh")
            commands.append("")
        
        # Python packages
        if missing_packages:
            commands.append("# Install Python packages")
            if self.inventory.get('uv', {}).get('installed'):
                commands.append(f"uv pip install {' '.join(missing_packages)}")
            else:
                commands.append(f"pip install {' '.join(missing_packages)}")
            commands.append("")
        
        # Ollama models
        if self.inventory.get('ollama', {}).get('installed'):
            models = self.inventory['ollama'].get('models', [])
            if 'qwen3:8b' not in models and 'mistral' not in models:
                commands.append("# Pull Ollama model")
                commands.append("ollama pull qwen3:8b")
                commands.append("")
        
        install_script_path.write_text('\n'.join(commands))
        print(f"   ✓ Installation script created: {install_script_path}")
        
        return install_script_path
    
    def save_inventory(self):
        """Save system inventory to JSON"""
        inventory_path = self.rag_dir / "config" / "system_inventory.json"
        with open(inventory_path, 'w') as f:
            json.dump(self.inventory, f, indent=2)
        print(f"\n💾 System inventory saved: {inventory_path}")
    
    def print_summary(self):
        """Print setup summary"""
        print("\n" + "="*60)
        print("📊 SYSTEM INVENTORY SUMMARY")
        print("="*60)
        
        ready_components = []
        missing_components = []
        
        for component, info in self.inventory.items():
            if isinstance(info, dict) and 'installed' in info:
                if info['installed']:
                    ready_components.append(component)
                else:
                    missing_components.append(component)
        
        print(f"\n✅ Ready: {', '.join(ready_components)}")
        if missing_components:
            print(f"❌ Missing: {', '.join(missing_components)}")
        
        print("\n" + "="*60)
        print("🎯 NEXT STEPS")
        print("="*60)
        
        if missing_components:
            print("\n1. Run the installation script:")
            print(f"   bash {self.rag_dir}/scripts/install_missing.sh")
            print("\n2. After installation, run this script again to verify")
        else:
            print("\n✨ All components ready!")
            print("\n1. Configure API keys in .env file")
            print("2. Run: python rag-system/scripts/initialize_system.py")
        
        print("\n" + "="*60)
    
    def run(self):
        """Run complete system check"""
        print("\n" + "="*60)
        print("🔍 TOGETHEROS RAG SYSTEM - SYSTEM INVENTORY")
        print("="*60)
        
        # Run all checks
        self.check_python()
        self.check_uv()
        self.check_ollama()
        self.check_git()
        self.check_node()
        packages_ok, missing_packages = self.check_python_packages()
        self.check_directory_structure()
        
        # Generate installation script if needed
        if not packages_ok or not self.inventory.get('ollama', {}).get('installed'):
            self.generate_installation_script(missing_packages)
        
        # Save inventory
        self.save_inventory()
        
        # Print summary
        self.print_summary()
        
        return self.inventory


if __name__ == "__main__":
    setup = RAGSystemSetup()
    inventory = setup.run()
