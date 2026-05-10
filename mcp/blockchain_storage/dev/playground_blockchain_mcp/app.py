"""
Dev-only Streamlit UI for Blockchain MCP storage logic.

Uses ChainStore + the same tool runners as the MCP server (no stdio bridge).

Usage:
    cd mcp/blockchain_storage
    pip install -e ".[dev-ui]"
    streamlit run dev/playground_blockchain_mcp/app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from blockchain_mcp.resources.chain_uri import build_chain_payload
from blockchain_mcp.resources.current_uri import build_current_payload
from blockchain_mcp.store.chain_store import ChainStore
from blockchain_mcp.tools import (
    run_append_context,
    run_cancel_block,
    run_commit_block,
    run_edit_block,
    run_get_chain_stats,
    run_validate_block,
)

st.set_page_config(page_title="Blockchain MCP playground", layout="wide")

_test_root = Path(__file__).resolve().parent / "_testdata"
_test_root.mkdir(exist_ok=True)
default_dir = str(_test_root)
if os.environ.get("PLAYGROUND_DATA_DIR"):
    data_dir = Path(os.environ["PLAYGROUND_DATA_DIR"]).expanduser().resolve()
    os.environ["DATA_DIR"] = str(data_dir)
else:
    os.environ.setdefault("DATA_DIR", default_dir)
    data_dir = Path(os.environ["DATA_DIR"]).expanduser().resolve()


@st.cache_resource
def get_playground_store() -> ChainStore:
    return ChainStore(data_dir)


store = get_playground_store()

st.title("Blockchain MCP playground")
st.caption(f"DATA_DIR = `{data_dir}` (isolates test JSON under `_testdata/` by default)")

chain_id = st.text_input("chain_id", value="demo-chain")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("append_context")
    append_body = st.text_area("content", height=120, key="append_c")
    token_override = st.text_input(
        "token_count override (optional, blank = heuristic)",
        value="",
    )
    if st.button("Append"):
        try:
            toks = int(token_override) if token_override.strip() else None
            out = run_append_context(
                store, chain_id=chain_id, content=append_body, token_count=toks
            )
            st.success(json.dumps(out, indent=2))
        except ValueError as e:
            st.error(str(e))

with col_b:
    st.subheader("edit_block")
    e_bid = st.text_input("block_id (edit)")
    replace = st.text_area("edits: content (full replace)", height=100, key="ed_rep")
    prepend = st.text_input("edits: prepend")
    append_sfx = st.text_input("edits: append")
    if st.button("Edit"):
        edits = {}
        if replace.strip():
            edits["content"] = replace
        if prepend:
            edits["prepend"] = prepend
        if append_sfx:
            edits["append"] = append_sfx
        try:
            out = run_edit_block(store, block_id=e_bid, edits=edits)
            st.success(json.dumps(out, indent=2))
        except ValueError as e:
            st.error(str(e))

st.divider()

col_c, col_d, col_e = st.columns(3)

with col_c:
    st.subheader("validate_block")
    v_bid = st.text_input("block_id (validate)")
    if st.button("Validate"):
        try:
            out = run_validate_block(store, block_id=v_bid)
            st.json(out)
        except ValueError as e:
            st.error(str(e))

with col_d:
    st.subheader("commit_block")
    c_bid = st.text_input("block_id (commit)")
    if st.button("Commit"):
        try:
            out = run_commit_block(store, block_id=c_bid)
            st.json(out)
        except ValueError as e:
            st.error(str(e))

with col_e:
    st.subheader("cancel_block")
    x_bid = st.text_input("block_id (cancel)")
    reason = st.text_input("reason", value="playground cancel")
    if st.button("Cancel"):
        out = run_cancel_block(store, block_id=x_bid, reason=reason)
        st.json(out)

st.divider()

if st.button("get_chain_stats"):
    st.json(run_get_chain_stats(store, chain_id=chain_id))

if st.button("resource preview: chain"):
    try:
        st.json(build_chain_payload(store, chain_id))
    except Exception as e:
        st.error(str(e))

if st.button("resource preview: current open"):
    try:
        st.json(build_current_payload(store, chain_id))
    except Exception as e:
        st.error(str(e))

if st.button("Clear Streamlit cache & reload store"):
    st.cache_resource.clear()
    st.rerun()
