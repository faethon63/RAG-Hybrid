// PM2 Configuration for RAG-Hybrid Backend
// Deploy to: /opt/rag-hybrid/ecosystem.config.js
module.exports = {
  apps: [{
    name: 'rag-backend',
    cwd: '/opt/rag-hybrid/backend',
    script: 'main.py',
    interpreter: '/opt/rag-hybrid/venv/bin/python',
    env: {
      ENVIRONMENT: 'production',
    },
    error_file: '/opt/rag-hybrid/logs/backend-error.log',
    out_file: '/opt/rag-hybrid/logs/backend-out.log',
    max_restarts: 10,
    restart_delay: 5000,
  }]
};
