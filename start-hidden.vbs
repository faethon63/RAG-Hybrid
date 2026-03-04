' RAG-Hybrid Hidden Startup Wrapper
' This VBScript runs PowerShell completely hidden (no window at all)
' Used by Task Scheduler to start services at Windows login

Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -File ""G:\AI-Project\RAG-Hybrid\start-background.ps1""", 0, False
