# Run Streamlit playground from workspace root (Cursor Hackathon).
# Requires: pip install -e "mcp/blockchain_storage[dev-ui]"
Set-Location "$PSScriptRoot\mcp\blockchain_storage"
streamlit run dev/playground_blockchain_mcp/app.py
