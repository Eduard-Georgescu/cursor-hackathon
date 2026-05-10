// ──────────────────────────────────────────────────────────────── state
const state = {
    sessionId: null,
    chainId: null,
    blocks: [],
    selectedBlockId: null,
    summary: null,
    snapshot: null,        // most recent backtrack result
    lastReclaim: null,     // most recent reclaim result
    health: null,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// ──────────────────────────────────────────────────────────── HTTP helpers

async function api(method, path, body) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body !== undefined) opts.body = JSON.stringify(body);
    const resp = await fetch(path, opts);
    if (!resp.ok) {
        let detail = '';
        try { detail = (await resp.json()).detail || ''; } catch { /* ignore */ }
        throw new Error(`${method} ${path} ${resp.status}: ${detail || resp.statusText}`);
    }
    return resp.json();
}

const post = (p, b) => api('POST', p, b);
const get = (p) => api('GET', p);

// ───────────────────────────────────────────────────────────────── toasts

function toast(tool, message, kind = 'ok') {
    const stack = $('#toast-stack');
    const el = document.createElement('div');
    el.className = `toast ${kind}`;
    el.innerHTML = `<div class="t-tool">${escapeHtml(tool)}</div>${escapeHtml(message)}`;
    stack.appendChild(el);
    setTimeout(() => el.remove(), 4200);
}

function escapeHtml(s) {
    return String(s ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// ───────────────────────────────────────────────────────────────── boot

async function boot() {
    try {
        state.health = await get('/api/health');
        const dot = $('#status-dot');
        dot.classList.add('ok');
        $('#status-text').textContent =
            `engine: ${state.health.engine} · embed: ${shortModel(state.health.embedding_model)}`;
    } catch (e) {
        $('#status-dot').classList.add('err');
        $('#status-text').textContent = 'API unreachable';
        toast('boot', String(e), 'err');
        return;
    }

    await refreshSessionsAndChains();
    await renderPrompts();

    bindUI();
}

function shortModel(name) {
    if (!name) return '?';
    const last = name.split('/').pop();
    return last || name;
}

async function refreshSessionsAndChains() {
    const [sessions, chains] = await Promise.all([get('/api/sessions'), get('/api/chains')]);
    const sSel = $('#session-select');
    const cSel = $('#chain-select');
    sSel.innerHTML = '';
    cSel.innerHTML = '';
    for (const s of sessions) {
        const opt = document.createElement('option');
        opt.value = s.session_id;
        opt.textContent = `${s.session_id}${s.active_chain_id ? ` → ${s.active_chain_id}` : ''}`;
        sSel.appendChild(opt);
    }
    for (const c of chains) {
        const opt = document.createElement('option');
        opt.value = c.chain_id;
        opt.textContent = `${c.chain_id}${c.title ? ` · ${c.title}` : ''}`;
        cSel.appendChild(opt);
    }
    if (sessions.length === 0) {
        toast('boot', 'No sessions found. Run `python scripts/seed_demo.py` first.', 'warn');
        return;
    }
    state.sessionId = sessions[0].session_id;
    state.chainId = sessions[0].active_chain_id || chains[0]?.chain_id;
    sSel.value = state.sessionId;
    cSel.value = state.chainId;
    await loadChainBlocks();
    await loadLatestHistory();
}

async function loadChainBlocks() {
    if (!state.chainId) return;
    state.blocks = await get(`/api/chain/${encodeURIComponent(state.chainId)}/blocks`);
    renderBlockList();
    $('#chain-meta').textContent = `${state.blocks.length} blocks`;
}

async function loadLatestHistory() {
    const h = await get(`/api/history/${encodeURIComponent(state.sessionId)}`);
    if (h.summary === null) {
        state.summary = null;
        renderHistory();
        return;
    }
    state.summary = h;
    renderHistory();
}

// ─────────────────────────────────────────────────────────────── rendering

function renderBlockList() {
    const root = $('#block-list');
    root.innerHTML = '';
    for (const b of state.blocks) {
        const el = document.createElement('div');
        el.className = 'block-item' + (b.block_id === state.selectedBlockId ? ' selected' : '');
        el.dataset.blockId = b.block_id;
        el.innerHTML = `
            <div class="meta">
                <span class="role-badge ${b.role}">${b.role}</span>
                <span>seq ${b.sequence}</span>
                <span>${b.block_id}</span>
            </div>
            <div class="content">${escapeHtml(b.content)}</div>`;
        el.addEventListener('click', () => {
            state.selectedBlockId = b.block_id;
            renderBlockList();
            scanAdjacent(b.block_id);
        });
        el.addEventListener('dblclick', () => previewBacktrack({ block_id: b.block_id }));
        root.appendChild(el);
    }
}

function renderResults(qr) {
    const root = $('#results');
    root.innerHTML = '';
    $('#query-meta').textContent =
        `${qr.matches.length} / ${qr.total_candidates} · mode=${qr.mode} · lookback=${qr.lookback_limit}`;

    if (qr.matches.length === 0) {
        root.innerHTML = `<p class="placeholder">No matches. Try a different prompt or widen the lookback.</p>`;
        return;
    }
    for (const m of qr.matches) {
        const el = document.createElement('div');
        el.className = 'match';
        const br = m.score_breakdown;
        const bars = br
            ? `<div class="match-bars">
                  <div class="match-bar sem"><span style="width:${pct(br.semantic)}"></span></div>
                  <div class="match-bar lex"><span style="width:${pct(br.lexical)}"></span></div>
                  <div class="match-bar rec"><span style="width:${pct(br.recency)}"></span></div>
               </div>
               <div class="match-legend">
                  <span class="sem"><i></i>sem ${br.semantic.toFixed(2)}</span>
                  <span class="lex"><i></i>lex ${br.lexical.toFixed(2)}</span>
                  <span class="rec"><i></i>rec ${br.recency.toFixed(2)}</span>
               </div>`
            : '';
        const hint = m.context_hint
            ? `<div class="match-hint">${escapeHtml(m.context_hint)}</div>`
            : '';
        el.innerHTML = `
            <div class="match-head">
                <span><span class="role-badge ${m.role}">${m.role}</span> seq ${m.sequence} · ${m.block_id}</span>
                <span class="score">${m.relevance.toFixed(3)}</span>
            </div>
            <div class="match-preview">${escapeHtml(m.preview)}</div>
            ${hint}
            ${bars}
        `;
        el.addEventListener('click', () => {
            state.selectedBlockId = m.block_id;
            renderBlockList();
            scanAdjacent(m.block_id);
            $$('.match').forEach((n) => n.classList.remove('selected'));
            el.classList.add('selected');
        });
        el.addEventListener('dblclick', () => previewBacktrack({ block_id: m.block_id }));
        root.appendChild(el);
    }
}

function pct(v) { return `${Math.max(0, Math.min(1, v)) * 100}%`; }

function renderAdjacent(adj) {
    const root = $('#adjacent');
    root.innerHTML = '';
    $('#adjacent-meta').textContent =
        `${adj.blocks.length} block(s) · ${adj.total_tokens} tok · chain len ${adj.total_chain_length}${adj.truncated ? ' · truncated' : ''}`;
    for (const b of adj.blocks) {
        const el = document.createElement('div');
        el.className = 'adjacent-block' + (b.block_id === adj.block_id ? ' anchor' : '');
        el.innerHTML = `
            <div class="meta">
                <span class="role-badge ${b.role}">${b.role}</span>
                <span>seq ${b.sequence}</span>
                <span>${b.block_id}</span>
            </div>
            <div>${escapeHtml(b.content)}</div>`;
        root.appendChild(el);
    }
}

function renderHistory() {
    const root = $('#history');
    root.innerHTML = '';
    if (!state.summary) {
        root.innerHTML = `<p class="placeholder">Click <strong>Generate</strong> to build a Claude / extractive history with clickable checkpoints.</p>`;
        return;
    }
    const s = state.summary;
    const isClaude = s.engine && s.engine.startsWith('claude');
    const enginePill = `<span class="pill ${isClaude ? 'claude' : 'extractive'}">${escapeHtml(s.engine || 'extractive')}</span>`;
    const cachedPill = s.cached ? `<span class="pill cached">cached</span>` : '';
    const detailPill = `<span class="pill">${escapeHtml(s.detail_level)}</span>`;
    const blockPill  = `<span class="pill">${s.block_count} blocks</span>`;

    const themesHtml = (s.themes && s.themes.length)
        ? `<div class="themes">
              <span class="section-label">themes</span>
              <div class="themes-row">
                ${s.themes.map((t) => `<span class="theme-chip">${escapeHtml(t)}</span>`).join('')}
              </div>
           </div>`
        : '';

    const decHtml = (s.decisions && s.decisions.length)
        ? `<div class="decisions">
              <span class="section-label">decisions</span>
              <ul>${s.decisions.map((d) => `<li>${escapeHtml(d.summary)}${d.block_id ? ` <span class="decision-link" data-block="${escapeHtml(d.block_id)}">[${escapeHtml(d.block_id)}]</span>` : ''}</li>`).join('')}</ul>
           </div>`
        : '';

    const qHtml = (s.open_questions && s.open_questions.length)
        ? `<div class="open-questions">
              <span class="section-label">open questions</span>
              <ul>${s.open_questions.map((q) => `<li>${escapeHtml(q)}</li>`).join('')}</ul>
           </div>`
        : '';

    root.innerHTML = `
        <div class="summary-meta">${enginePill}${cachedPill}${detailPill}${blockPill}</div>
        <div class="summary-text">${escapeHtml(s.plain_text)}</div>
        ${themesHtml}
        ${decHtml}
        ${qHtml}
        <div class="checkpoints">
            <span class="section-label">checkpoints · click to backtrack</span>
            ${(s.checkpoints || []).map((cp) => renderCheckpointCard(cp)).join('')}
        </div>
    `;

    $$('#history .checkpoint').forEach((el) => {
        el.addEventListener('click', () => {
            const cpId = el.dataset.checkpoint;
            previewBacktrack({ checkpoint_id: cpId });
        });
    });
    $$('#history .decision-link').forEach((el) => {
        el.addEventListener('click', (e) => {
            e.stopPropagation();
            const bid = el.dataset.block;
            state.selectedBlockId = bid;
            renderBlockList();
            scanAdjacent(bid);
        });
    });
}

function renderCheckpointCard(cp) {
    const cat = cp.category || 'transition';
    const conf = cp.confidence != null ? `· ${(cp.confidence * 100).toFixed(0)}%` : '';
    return `
        <div class="checkpoint" data-checkpoint="${escapeHtml(cp.checkpoint_id)}">
            <div class="cp-head">
                <span class="cp-label">${escapeHtml(cp.label)}</span>
                <span class="cp-meta">
                    <span class="cat-chip ${cat}">${escapeHtml(cat)}</span>
                    <span>seq ${escapeHtml(cp.block_id)} ${conf}</span>
                </span>
            </div>
            ${cp.description ? `<div class="cp-desc">${escapeHtml(cp.description)}</div>` : ''}
            ${cp.reason ? `<div class="cp-reason">${escapeHtml(cp.reason)}</div>` : ''}
        </div>`;
}

function renderSnapshot() {
    const root = $('#snapshot');
    if (!state.snapshot) {
        root.innerHTML = `<p class="placeholder">Click a checkpoint above (or a block on the left) to preview a backtrack.</p>`;
        $('#snapshot-meta').textContent = '';
        return;
    }
    const s = state.snapshot;
    $('#snapshot-meta').textContent = s.dry_run ? 'preview · not persisted' : `snapshot_id=${shortId(s.snapshot_id)}`;
    const rolePills = Object.entries(s.role_counts || {})
        .map(([r, n]) => `<span class="role-badge ${r}">${r}·${n}</span>`)
        .join(' ');
    let reclaimHtml = '';
    if (state.lastReclaim && state.lastReclaim.snapshot_id === s.snapshot_id) {
        const r = state.lastReclaim;
        reclaimHtml = `
            <div class="snapshot-reclaim">
                <div><strong>reclaim_context</strong> — ${r.blocks_injected} injected · ${r.blocks_skipped} skipped · ${r.tokens_injected} tokens · ${r.compressed ? 'compressed' : 'no compression'}</div>
                ${r.compression_summary ? `<div class="compression">${escapeHtml(r.compression_summary)}</div>` : ''}
            </div>`;
    }
    root.innerHTML = `
        <div class="snapshot-card">
            <div class="manifest">
                <span>anchor</span><strong>${escapeHtml(s.anchor_block_id)} (seq ${s.anchor_sequence})</strong>
                <span>blocks</span><strong>${s.block_count}</strong>
                <span>est. tokens</span><strong>${s.estimated_tokens}</strong>
                <span>roles</span><strong>${rolePills}</strong>
            </div>
            <div class="snapshot-actions">
                <label>token budget <input type="number" id="reclaim-budget" min="1" placeholder="optional" /></label>
                <label class="checkbox"><input type="checkbox" id="reclaim-compress" checked /> compress</label>
                <button class="btn btn-primary" id="reclaim-btn" ${s.dry_run ? 'disabled title="confirm the backtrack first"' : ''}>reclaim_context</button>
            </div>
            ${s.restored_summary ? `<div class="cp-reason" style="margin-top:8px;">${escapeHtml(s.restored_summary)}</div>` : ''}
        </div>
        ${reclaimHtml}
    `;
    const btn = $('#reclaim-btn');
    if (btn) btn.addEventListener('click', () => reclaimSnapshot(s.snapshot_id));
}

function shortId(id) { return id ? id.slice(0, 8) : ''; }

async function renderPrompts() {
    const root = $('#prompts');
    const prompts = await get('/api/prompts');
    root.innerHTML = '';
    for (const p of prompts) {
        const card = document.createElement('div');
        card.className = 'prompt-card';
        let actionsHtml = '';
        if (p.name === 'recap_history') {
            actionsHtml = `
                <select data-role="detail">
                    <option value="skim">skim</option>
                    <option value="standard" selected>standard</option>
                    <option value="deep">deep</option>
                </select>
                <button class="btn btn-primary" data-action="recap">Run</button>`;
        } else if (p.name === 'find_related_context') {
            actionsHtml = `
                <input type="text" placeholder="topic" data-role="topic" />
                <button class="btn btn-primary" data-action="find">Run</button>`;
        } else if (p.name === 'restore_to_decision') {
            actionsHtml = `
                <input type="text" placeholder="decision query" data-role="decision" />
                <button class="btn btn-primary" data-action="restore">Run</button>`;
        }
        card.innerHTML = `
            <h4>${escapeHtml(p.title)}</h4>
            <p>${escapeHtml(p.description)}</p>
            <div class="prompt-actions">${actionsHtml}</div>
        `;
        root.appendChild(card);
    }
    $$('#prompts [data-action="recap"]').forEach((b) => b.addEventListener('click', async () => {
        const detail = b.parentElement.querySelector('[data-role="detail"]').value;
        $('#detail-select').value = detail;
        await summarize(detail, false);
        toast('prompt:recap_history', `Generated ${detail} history`);
    }));
    $$('#prompts [data-action="find"]').forEach((b) => b.addEventListener('click', async () => {
        const topic = b.parentElement.querySelector('[data-role="topic"]').value.trim();
        if (!topic) { toast('prompt:find_related_context', 'enter a topic', 'warn'); return; }
        $('#search-input').value = topic;
        $('#diversify-toggle').checked = true;
        $('#explain-toggle').checked = true;
        await runSearch();
        toast('prompt:find_related_context', `Searched for "${topic}"`);
    }));
    $$('#prompts [data-action="restore"]').forEach((b) => b.addEventListener('click', async () => {
        const q = b.parentElement.querySelector('[data-role="decision"]').value.trim();
        if (!q) { toast('prompt:restore_to_decision', 'enter a decision query', 'warn'); return; }
        await summarize('deep', true);
        // Pick the checkpoint whose label/reason best matches q (case-insensitive substring).
        const cps = state.summary?.checkpoints || [];
        const lower = q.toLowerCase();
        let best = cps.find(c => (c.label + ' ' + (c.reason || '') + ' ' + (c.description || '')).toLowerCase().includes(lower));
        if (!best) best = cps[Math.floor(cps.length / 2)];
        if (!best) { toast('prompt:restore_to_decision', 'no checkpoint found', 'warn'); return; }
        previewBacktrack({ checkpoint_id: best.checkpoint_id });
    }));
}

// ───────────────────────────────────────────────────────────── tool calls

async function runSearch() {
    const prompt = $('#search-input').value.trim();
    if (!prompt) return;
    const body = {
        chain_id: state.chainId,
        prompt,
        session_id: state.sessionId,
        top_k: +$('#topk-input').value || 6,
        mode: $('#mode-select').value,
        diversify: $('#diversify-toggle').checked,
        explain: $('#explain-toggle').checked,
        min_relevance: +$('#minrel-input').value || 0,
        roles: $$('#role-filters input:checked').map((i) => i.value),
    };
    if (body.roles.length === 0) delete body.roles;
    try {
        const res = await post('/api/query', body);
        renderResults(res);
        toast('query_chain', `${res.matches.length} matches · ${res.mode}`);
    } catch (e) {
        toast('query_chain', String(e), 'err');
    }
}

async function scanAdjacent(blockId) {
    try {
        const res = await post('/api/adjacent', { block_id: blockId, window: 2 });
        renderAdjacent(res);
    } catch (e) {
        toast('scan_adjacent', String(e), 'err');
    }
}

async function saveLookback() {
    const limit = +$('#lookback-slider').value;
    try {
        const res = await post('/api/lookback', {
            session_id: state.sessionId,
            limit,
            scope: 'session',
        });
        toast('set_lookback', `session lookback: ${res.previous_limit} → ${res.updated_limit}`);
    } catch (e) {
        toast('set_lookback', String(e), 'err');
    }
}

async function summarize(detail, force) {
    try {
        const res = await post('/api/summarize', {
            chain_id: state.chainId,
            session_id: state.sessionId,
            detail_level: detail || $('#detail-select').value,
            force_refresh: !!force,
        });
        state.summary = res;
        renderHistory();
        toast('summarize_history', `${res.engine}${res.cached ? ' · cached' : ''} · ${res.checkpoints.length} checkpoints`);
    } catch (e) {
        toast('summarize_history', String(e), 'err');
    }
}

function previewBacktrack(target) {
    openModal(target);
}

async function reclaimSnapshot(snapshot_id) {
    const budget = +$('#reclaim-budget').value || null;
    const compress = $('#reclaim-compress').checked;
    try {
        const res = await post('/api/reclaim', {
            session_id: state.sessionId,
            snapshot_id,
            token_budget: budget || undefined,
            compress_if_over: compress,
        });
        state.lastReclaim = res;
        renderSnapshot();
        toast('reclaim_context',
            `${res.blocks_injected} injected · ${res.blocks_skipped} skipped · ${res.tokens_injected} tokens${res.compressed ? ' · compressed' : ''}`);
    } catch (e) {
        toast('reclaim_context', String(e), 'err');
    }
}

// ─────────────────────────────────────────────────────────────── modal

let pendingBacktrack = null;

async function openModal(target) {
    pendingBacktrack = target;
    try {
        // dry-run first so the user can see what's about to be frozen
        const preview = await post('/api/backtrack', {
            session_id: state.sessionId,
            ...target,
            dry_run: true,
        });
        const rolePills = Object.entries(preview.role_counts || {})
            .map(([r, n]) => `<span class="role-badge ${r}">${r}·${n}</span>`)
            .join(' ');
        $('#modal-body').innerHTML = `
            <p>${escapeHtml(preview.restored_summary)}</p>
            <div class="manifest" style="display:grid;grid-template-columns:max-content 1fr;gap:6px 12px;font-family:var(--mono);font-size:12px;">
                <span>anchor</span><strong>${escapeHtml(preview.anchor_block_id)} (seq ${preview.anchor_sequence})</strong>
                <span>blocks</span><strong>${preview.block_count}</strong>
                <span>est. tokens</span><strong>${preview.estimated_tokens}</strong>
                <span>roles</span><strong>${rolePills}</strong>
            </div>
            <p style="margin-top:12px;">Confirm to write the snapshot and re-inject it into the session.</p>
        `;
        state.snapshot = preview;
        renderSnapshot();
        $('#modal-backdrop').hidden = false;
    } catch (e) {
        toast('backtrack', String(e), 'err');
    }
}

function closeModal() {
    $('#modal-backdrop').hidden = true;
    pendingBacktrack = null;
}

async function confirmBacktrack() {
    if (!pendingBacktrack) return;
    try {
        const res = await post('/api/backtrack', {
            session_id: state.sessionId,
            ...pendingBacktrack,
            dry_run: false,
            name: 'demo-ui',
            reason: 'restored from web demo',
        });
        state.snapshot = res;
        state.lastReclaim = null;
        renderSnapshot();
        toast('backtrack', `Snapshot ${shortId(res.snapshot_id)} created · ${res.block_count} blocks`);
        // Auto-reclaim with no budget so the user sees the full path.
        await reclaimSnapshot(res.snapshot_id);
        closeModal();
    } catch (e) {
        toast('backtrack', String(e), 'err');
    }
}

// ───────────────────────────────────────────────────────────────── wiring

function bindUI() {
    $('#session-select').addEventListener('change', async (e) => {
        state.sessionId = e.target.value;
        state.summary = null;
        state.snapshot = null;
        state.lastReclaim = null;
        renderSnapshot();
        await loadLatestHistory();
        toast('session', `→ ${state.sessionId}`);
    });
    $('#chain-select').addEventListener('change', async (e) => {
        state.chainId = e.target.value;
        await loadChainBlocks();
        $('#results').innerHTML = `<p class="placeholder">Run a search to see hybrid matches with score breakdowns.</p>`;
        $('#adjacent').innerHTML = `<p class="placeholder">Click a search result to load its neighbors.</p>`;
        toast('chain', `→ ${state.chainId}`);
    });
    $('#lookback-slider').addEventListener('input', (e) => {
        $('#lookback-value').value = e.target.value;
    });
    $('#lookback-save').addEventListener('click', saveLookback);
    $('#search-btn').addEventListener('click', runSearch);
    $('#search-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') runSearch();
    });
    $('#summarize-btn').addEventListener('click', () => summarize());
    $('#refresh-btn').addEventListener('click', () => summarize(undefined, true));
    $('#modal-close').addEventListener('click', closeModal);
    $('#modal-cancel').addEventListener('click', closeModal);
    $('#modal-confirm').addEventListener('click', confirmBacktrack);
    $('#modal-backdrop').addEventListener('click', (e) => {
        if (e.target.id === 'modal-backdrop') closeModal();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });
}

// ─────────────────────────────────────────────────────────────────── go

boot();
