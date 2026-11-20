import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'
import { LocalAgentAdapter } from '../src/host/local-agents/adapter.js'
import { AcpEngine, stdioBridgePath } from '../src/host/local-agents/acp.js'
import { CodexEngine } from '../src/host/local-agents/codex.js'
import { JsonRpcPeer, wireProcess } from '../src/host/local-agents/jsonrpc.js'
import { ClaudeEngine } from '../src/host/local-agents/claude.js'
import { LoopbackMcp, flattenSchema } from '../src/host/local-agents/mcp.js'
import { cleanEnv } from '../src/host/local-agents/discover.js'
import { ENGINES, canSteerEngine, validateCustom, isLocalAgentRoute, cliMode, acpMode } from '../src/host/local-agents/engines.js'
import { AgySteerServer, ensureAgySteerPlugin, postAgySteer, readAgyTranscriptWakeup, readAgyTranscriptWatermark } from '../src/host/local-agents/agy-steer.js'
import { install as localAgentsInstall } from '../src/host/local-agents/index.js'
import { AGENT_BADGES, agentAvatar, badgeOf, isLocalRoute } from '../src/client/agent-routes.js'
import { stripHeader } from '../src/client/text.js'
import { INKS } from '../src/client/inks.js'
import { install as partsInstall } from '../src/host/parts.js'
import { install as modelsInstall } from '../src/host/models.js'
import { install as turnsInstall } from '../src/host/turns.js'
import { fakeContext } from './fake-ctx.js'

const FIXTURE = name => fileURLToPath(new URL(`./fixtures/${name}`, import.meta.url))
const FIXTURES = { acp: FIXTURE('fake-acp.mjs'), codex: FIXTURE('fake-codex.mjs'), claude: FIXTURE('fake-claude.mjs') }

const user = (id, text) => ({ id, role: 'user', content: [{ type: 'text', text }], source: { kind: 'user' } })
const system = (id, text) => ({ id, role: 'system', content: [{ type: 'text', text }], source: { kind: 'system-prompt' } })
const toolResult = (id, callId, text, isError = false) => ({ id, role: 'tool', content: [{ type: 'text', text }], toolCallId: callId, isError, source: { kind: 'tool', callId } })

// An adapter wired to the fake CLIs: every engine route spawns `node fixture`.
function makeAdapter({ records = {}, preset = 'workspace-write', acp = FIXTURES.acp, acpArgs = [], codex = FIXTURES.codex, codexArgs = [], claude = FIXTURES.claude, claudeArgs = [], cwdFor, attachments, fs, rt, onBackgroundWake, onBackgroundSettled, onBackgroundText, bgIdleSettleMs, bgActiveSettleMs, peekInboxSteers, hasInboxSteers, onStreamTouch, onCliWork } = {}) {
  const files = { acp, codex, claude }
  const spawned = []
  const agyHomeDir = mkdtempSync(join(tmpdir(), 'dsb-agy-home-'))
  const adapter = new LocalAgentAdapter({
    rt: rt ?? { warn: () => {} },
    attachments,
    fs,
    customAgents: () => [{ id: 'fake', name: 'Fake ACP', command: 'x', args: [] }, { id: 'fake2', name: 'Fake ACP 2', command: 'x', args: [] }],
    findExecutable: path => path,
    spawnEngine: (kind, route, spec, hooks = {}) => {
      const engine = kind === 'acp' ? new AcpEngine(route, { ...spec, command: process.execPath, args: [files.acp, ...acpArgs] }, { findExecutable: path => path, agyHomeDir, ...hooks })
        : kind === 'codex' ? new CodexEngine(route, { ...spec, command: process.execPath, args: [files.codex, ...codexArgs] }, { findExecutable: path => path })
          : new ClaudeEngine(route, { ...spec, command: process.execPath, args: [files.claude, ...claudeArgs] }, { findExecutable: path => path })
      spawned.push(engine)
      return engine
    },
    record: sessionId => records[sessionId],
    saveRecord: (sessionId, record) => { records[sessionId] = record },
    presetFor: typeof preset === 'function' ? preset : () => preset,
    cwdFor: cwdFor ?? (async () => mkdtempSync(join(tmpdir(), 'dsb-cwd-'))),
    probeCwd: () => mkdtempSync(join(tmpdir(), 'dsb-probe-')),
    toolsFor: () => [{ name: 'message_bot', description: 'Message another Bot', parameters: { type: 'object', properties: { bot: { type: 'string' }, text: { type: 'string' } } } }],
    peekInboxSteers,
    hasInboxSteers,
    onStreamTouch,
    onCliWork,
    onBackgroundWake,
    onBackgroundSettled,
    onBackgroundText,
    bgIdleSettleMs,
    bgActiveSettleMs,
  })
  adapter._spawned = spawned
  adapter._agyHomeDir = agyHomeDir
  return adapter
}

// Wait for the agent process behind a live session to exit.
async function waitForDead(adapter, sessionId = 's1') {
  for (let attempt = 0; attempt < 300; attempt += 1) {
    if (adapter.live.get(sessionId)?.engineSession?.dead === true) return
    await new Promise(resolve => setTimeout(resolve, 10))
  }
  throw new Error('the agent process did not exit')
}

// Collect a stream into {text, thinking, calls, finish} for assertions.
async function collect(stream) {
  const out = { text: '', thinking: '', calls: [], finish: undefined, usage: undefined }
  for await (const chunk of stream) {
    if (chunk.type === 'text-delta') out.text += chunk.text
    else if (chunk.type === 'reasoning-delta') out.thinking += chunk.text
    else if (chunk.type === 'block-end' && chunk.block?.type === 'tool-call') out.calls.push(chunk.block)
    else if (chunk.type === 'usage') out.usage = chunk.usage
    else if (chunk.type === 'finish') out.finish = chunk.reason
  }
  return out
}

const options = (sessionId, messages, extra = {}) => ({ provider: 'local-acp-fake', model: 'default', sessionId, messages, tools: [{ name: 'message_bot' }], signal: AbortSignal.timeout(60_000), ...extra })

test('session-title and compaction stay deterministic, no CLI is started', async () => {
  const adapter = makeAdapter()
  const title = await collect(adapter.stream(options('s1', [user('m1', 'Please review the auth changes')], { purpose: 'session-title' })))
  assert.equal(title.text, 'Please review the auth changes')
  assert.equal(title.finish.kind, 'stop')
  const digest = await collect(adapter.stream(options('s1', [user('m1', 'hello')], { purpose: 'compaction' })))
  assert.match(digest.text, /condensed/)
  assert.equal(adapter.engines.size, 0)
  await adapter.dispose()
})

test('a first turn replays the conversation, a later one sends only the delta', async (t) => {
  const records = {}
  const adapter = makeAdapter({ records })
  t.after(() => adapter.dispose())
  const first = await collect(adapter.stream(options('s1', [system('sys', 'You are Kai, a coder.'), user('m1', 'hello')])))
  assert.match(first.text, /^ok: /)
  assert.match(first.thinking, /thought about/)
  assert.equal(first.finish.kind, 'stop')
  // The record remembers the CLI session and how far it was handed.
  assert.ok(records.s1.cliSessionId?.startsWith('acp-'))
  assert.equal(records.s1.handedId, 'm1')
  const second = await collect(adapter.stream(options('s1', [system('sys', 'You are Kai, a coder.'), user('m1', 'hello'), user('m2', 'again')])))
  assert.match(second.text, /ok: again/)
  assert.equal(records.s1.handedId, 'm2')
})

test('each DS Bot Session maps to its own CLI session', async (t) => {
  const records = {}
  const adapter = makeAdapter({ records })
  t.after(() => adapter.dispose())
  await collect(adapter.stream(options('own', [user('m1', 'one')])))
  await collect(adapter.stream(options('group', [user('m1', 'two')])))
  assert.notEqual(records.own.cliSessionId, records.group.cliSessionId)
})

test('the MCP bridge suspends the turn and the next stream answers it', async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  const first = await collect(adapter.stream(options('s1', [user('m1', 'please call-tool now')])))
  assert.equal(first.finish.kind, 'tool-calls')
  assert.equal(first.calls.length, 1)
  assert.equal(first.calls[0].name, 'message_bot')
  const step2Context = { id: 'rc-step2', role: 'user', content: [{ type: 'text', text: 'Time sampled while preparing turn 1, step 2: 2026-11-16T13:40:00+08:00' }], source: { kind: 'runtime-context' } }
  const second = await collect(adapter.stream(options('s1', [user('m1', 'please call-tool now'), toolResult('m2', first.calls[0].id, 'Kai says hi back'), step2Context])))
  assert.equal(second.finish.kind, 'stop')
  assert.match(second.text, /tool said: Kai says hi back/)
  assert.doesNotMatch(second.text, /ok: Time sampled/, 'step-2 runtime-context must not queue a second ACP prompt')
})

test('queued calls finish later streams', async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  const first = await collect(adapter.stream(options('s1', [user('m1', 'two-calls')])))
  assert.equal(first.finish.kind, 'tool-calls')
  assert.equal(first.calls.length, 1)
  const seen = [user('m1', 'two-calls'), toolResult('m2', first.calls[0].id, 'one back')]
  const second = await collect(adapter.stream(options('s1', seen)))
  // The second queued call is emitted (or the agent already got both answered).
  assert.ok(second.finish.kind === 'tool-calls' || /tool said/.test(second.text), JSON.stringify(second))
})

test('a permission prompt bridges to approve_agent_action and fails closed', async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  const ask = await collect(adapter.stream(options('s1', [user('m1', 'ask-permission')])))
  assert.equal(ask.finish.kind, 'tool-calls')
  assert.equal(ask.calls[0].name, 'approve_agent_action')
  // The request named the call only by id; the card still says what it runs.
  const asked = JSON.parse(ask.calls[0].arguments ?? JSON.stringify(ask.calls[0].args))
  assert.equal(asked.action, 'Run `rm -rf /tmp/x`')
  assert.equal(asked.detail ?? '', '')
  // An allowed answer picks the agent's allow option: the tool returns its
  // structured decision, and only {"decision":"allow"} counts.
  const allowed = await collect(adapter.stream(options('s1', [user('m1', 'ask-permission'), toolResult('m2', ask.calls[0].id, JSON.stringify({ decision: 'allow' }))])))
  assert.match(allowed.text, /allowed via allow-once/)
})
test('a call to the dsbot server is allowed once without asking; other servers still ask', async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  const team = await collect(adapter.stream(options('s1', [user('m1', 'team-permission')])))
  // The MCP call itself reaches DS Bot's tools; no approval card in between.
  assert.equal(team.calls[0]?.name, 'message_bot', JSON.stringify(team))
  const done = await collect(adapter.stream(options('s1', [user('m1', 'team-permission'), toolResult('m2', team.calls[0].id, 'River: pong')])))
  assert.match(done.text, /allowed via allow-once; tool said: River: pong/)
  const other = await collect(adapter.stream(options('s2', [user('m1', 'other-permission')])))
  assert.equal(other.calls[0]?.name, 'approve_agent_action')
})
test('a permission answer that is not the structured allow fails closed', async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  const ask = await collect(adapter.stream(options('s1', [user('m1', 'ask-permission')])))
  // Free text that reads like an approval is not an approval.
  const loose = await collect(adapter.stream(options('s1', [user('m1', 'ask-permission'), toolResult('m2', ask.calls[0].id, 'yes, allow it')])))
  assert.match(loose.text, /denied/)
})
test('a denied answer rejects the agent\'s action', async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  const ask = await collect(adapter.stream(options('s1', [user('m1', 'ask-permission')])))
  const denied = await collect(adapter.stream(options('s1', [user('m1', 'ask-permission'), toolResult('m2', ask.calls[0].id, 'denied', true)])))
  assert.match(denied.text, /denied/)
})

test('abort cancels the turn and the pending calls', async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  const control = new AbortController()
  const stream = collect(adapter.stream(options('s1', [user('m1', 'hang')], { signal: control.signal })))
  setTimeout(() => control.abort(), 300)
  const out = await stream
  assert.equal(out.finish.kind, 'aborted')
})

test('env sanitization drops agent and key variables', () => {
  process.env.ACP_BACKEND = 'windsurf'
  process.env.OPENAI_API_KEY = 'sk-test'
  process.env.DS_BOT_HOME = '/x'
  const env = cleanEnv({ EXTRA: '1' })
  assert.equal(env.ACP_BACKEND, undefined)
  assert.equal(env.OPENAI_API_KEY, undefined)
  assert.equal(env.DS_BOT_HOME, undefined)
  assert.equal(env.EXTRA, '1')
  assert.ok(env.PATH.startsWith(process.env.PATH))
})

// A Dock-launched app has PATH=/usr/bin:/bin:…; a CLI found in ~/.local/bin or
// Homebrew must also find its `#!/usr/bin/env node` there.
test('agent env PATH includes the folders CLIs are found in', { skip: process.platform === 'win32' }, () => {
  const saved = { HOME: process.env.HOME, PATH: process.env.PATH }
  const home = mkdtempSync(join(tmpdir(), 'dsbot-home-'))
  mkdirSync(join(home, '.local', 'bin'), { recursive: true })
  process.env.HOME = home
  process.env.PATH = '/usr/bin:/bin'
  try {
    const dirs = cleanEnv().PATH.split(':')
    assert.deepEqual(dirs.slice(0, 2), ['/usr/bin', '/bin'])
    assert.ok(dirs.includes(join(home, '.local', 'bin')))
    assert.equal(cleanEnv({ PATH: '/only' }).PATH, '/only')
  } finally {
    Object.assign(process.env, saved)
  }
})

test('custom agent validation', () => {
  assert.deepEqual(validateCustom({ name: 'Mine', command: 'my-agent', args: ['acp', '--fast'] }), { name: 'Mine', command: 'my-agent', args: ['acp', '--fast'] })
  assert.throws(() => validateCustom({ name: 'X', command: 'a b' }))
  assert.throws(() => validateCustom({ name: 'X', command: 'agent; rm -rf /' }))
  assert.throws(() => validateCustom({ name: '', command: 'a' }))
})

test('routes and modes', () => {
  assert.equal(isLocalAgentRoute('local-devin-cli'), true)
  assert.equal(isLocalAgentRoute('local-acp-custom'), true)
  assert.equal(isLocalAgentRoute('deepseek'), false)
  assert.equal(acpMode('read-only', [{ id: 'ask' }, { id: 'bypass' }]), 'ask')
  assert.equal(acpMode('danger-full-access', [{ id: 'ask' }, { id: 'bypass' }]), 'bypass')
  // Config options name a mode by `value`; an agent without a match keeps its own.
  assert.equal(acpMode('read-only', [{ value: 'agent' }, { value: 'plan' }, { value: 'ask' }]), 'ask')
  assert.equal(acpMode('workspace-write', [{ value: 'agent' }, { value: 'plan' }, { value: 'ask' }]), undefined)
  assert.equal(cliMode('claude', 'workspace-write'), 'acceptEdits')
  assert.equal(cliMode('acp', 'workspace-write'), undefined)
})

test('acp: a resumed session that lists no modes keeps the ones its CLI listed', async (t) => {
  const records = { s2: { route: 'local-acp-fake', cliSessionId: 'acp-old', model: 'default' } }
  const adapter = makeAdapter({ records, preset: 'workspace-write', acpArgs: ['--cursor-modes'] })
  t.after(() => adapter.dispose())
  await collect(adapter.stream(options('s1', [user('m1', 'show-mode')])))
  // session/load lists nothing; without what session/new listed, the guess
  // accept-edits would be sent and refused twice.
  const resumed = await collect(adapter.stream(options('s2', [user('m1', 'show-mode')])))
  assert.match(resumed.text, /mode: agent; refused: 0/)
})

test('acp: full access allows and read-only refuses a change without a card', async (t) => {
  const full = makeAdapter({ preset: 'danger-full-access', acpArgs: ['--cursor-modes'] })
  const readOnly = makeAdapter({ preset: 'read-only', acpArgs: ['--cursor-modes'] })
  t.after(() => Promise.all([full, readOnly].map(adapter => adapter.dispose())))
  // Cursor has no bypass mode: the preset is kept by DS Bot's own answer.
  const allowed = await collect(full.stream(options('s1', [user('m1', 'ask-permission')])))
  assert.equal(allowed.calls.length, 0)
  assert.match(allowed.text, /allowed via allow-once/)
  const refused = await collect(readOnly.stream(options('s1', [user('m1', 'ask-permission')])))
  assert.equal(refused.calls.length, 0)
  assert.match(refused.text, /denied/)
})

test('devin: a call to the dsbot server is allowed once without asking; other servers still ask', async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  const team = await collect(adapter.stream(options('s1', [user('m1', 'devin-team')])))
  assert.equal(team.calls[0]?.name, 'message_bot', JSON.stringify(team))
  const done = await collect(adapter.stream(options('s1', [user('m1', 'devin-team'), toolResult('m2', team.calls[0].id, 'River: pong')])))
  assert.match(done.text, /allowed via allow_once; tool said: River: pong/)
  const other = await collect(adapter.stream(options('s2', [user('m1', 'devin-other')])))
  assert.equal(other.calls[0]?.name, 'approve_agent_action')
})

test('acp: a dsbot call with no one-time allow still asks', async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  const asked = await collect(adapter.stream(options('s1', [user('m1', 'team-permission always-only')])))
  assert.equal(asked.calls[0]?.name, 'approve_agent_action')
})

test('acp: the preset picks among the modes the session lists', async (t) => {
  const plain = makeAdapter({ preset: 'workspace-write' })
  const readOnly = makeAdapter({ preset: 'read-only', acpArgs: ['--cursor-modes'] })
  const write = makeAdapter({ preset: 'workspace-write', acpArgs: ['--cursor-modes'] })
  t.after(() => Promise.all([plain, readOnly, write].map(adapter => adapter.dispose())))
  assert.match((await collect(plain.stream(options('s1', [user('m1', 'show-mode')])))).text, /mode: accept-edits; refused: 0/)
  assert.match((await collect(readOnly.stream(options('s1', [user('m1', 'show-mode')])))).text, /mode: ask; refused: 0/)
  // Cursor has no accept-edits: it stays in its own agent mode and asks before acting.
  assert.match((await collect(write.stream(options('s1', [user('m1', 'show-mode')])))).text, /mode: agent; refused: 0/)
})

// #1503: a live CLI keeps the permission mode it opened with. Changing the
// Bot's preset must reopen the CLI for the next turn, not wait for a restart.
test('a changed permission preset reopens the local agent for the next turn', async () => {
  let preset = 'workspace-write'
  const adapter = makeAdapter({ preset: () => preset })
  try {
    const first = await collect(adapter.stream(options('s1', [user('m1', 'show-mode')])))
    assert.match(first.text, /mode: accept-edits/)
    preset = 'danger-full-access'
    const second = await collect(adapter.stream(options('s1', [user('m2', 'show-mode')])))
    assert.match(second.text, /mode: bypass/)
  } finally {
    await adapter.dispose()
  }
})

test('codex: a changed permission preset resumes the thread with the new sandbox and approval', async () => {
  let preset = 'workspace-write'
  const record = join(mkdtempSync(join(tmpdir(), 'dsb-codex-preset-')), 'params.jsonl')
  const records = {}
  const adapter = makeAdapter({ preset: () => preset, records, codexArgs: ['--record', record] })
  try {
    const request = (id, text) => options('s1', [user(id, text)], { provider: 'local-codex-cli', model: 'gpt-5.1-codex' })
    await collect(adapter.stream(request('m1', 'hello')))
    preset = 'danger-full-access'
    await collect(adapter.stream(request('m2', 'again')))
  } finally {
    await adapter.dispose()
  }
  const calls = readFileSync(record, 'utf8').trim().split('\n').map(line => JSON.parse(line))
  const resumed = calls.filter(call => call.method === 'thread/resume')
  assert.equal(resumed.length, 1)
  assert.equal(resumed[0].params.sandbox, 'danger-full-access')
  assert.equal(resumed[0].params.approvalPolicy, 'never')
})

test('probeModels reports the probe session\'s model options plus CLI default', async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  const models = await adapter.probeModels('local-acp-fake')
  assert.deepEqual(models.map(model => model.id), ['default', 'fake-large', 'fake-small'])
  assert.equal(models[0].name, 'CLI default')
})

test('codex app-server: thread per session, models from model/list', async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  const models = await adapter.probeModels('local-codex-cli')
  assert.deepEqual(models.map(model => model.id), ['default', 'gpt-5.1-codex'])
  const out = await collect(adapter.stream(options('s1', [user('m1', 'fix it')], { provider: 'local-codex-cli' })))
  assert.match(out.text, /fix it/)
  assert.match(out.thinking, /▸/)
})

// #1144: Codex 0.166 reads ThreadStartParams at the top level and the MCP servers from
// `config`; keys it does not know are dropped silently. With the old snake_case shape
// the team tools never registered and the permission preset never applied.
test('codex app-server: thread start and resume send the fields Codex reads', async () => {
  for (const [preset, sandbox, approvalPolicy] of [
    ['read-only', 'read-only', 'on-request'],
    ['workspace-write', 'workspace-write', 'on-request'],
    ['danger-full-access', 'danger-full-access', 'never'],
  ]) {
    const record = join(mkdtempSync(join(tmpdir(), 'dsb-codex-params-')), 'params.jsonl')
    const records = {}
    const messages = [system('sys', 'You are Kai, a coder.'), user('m1', 'hello')]
    const first = makeAdapter({ preset, records, codexArgs: ['--record', record] })
    try {
      await collect(first.stream(options('s1', messages, { provider: 'local-codex-cli', model: 'gpt-5.1-codex' })))
    } finally {
      await first.dispose()
    }
    // A new host resumes the thread the first one saved.
    const second = makeAdapter({ preset, records, codexArgs: ['--record', record] })
    try {
      await collect(second.stream(options('s1', [...messages, user('m2', 'again')], { provider: 'local-codex-cli', model: 'gpt-5.1-codex' })))
    } finally {
      await second.dispose()
    }
    const calls = readFileSync(record, 'utf8').trim().split('\n').map(line => JSON.parse(line))
    const start = calls.find(call => call.method === 'thread/start')?.params
    const resume = calls.find(call => call.method === 'thread/resume')?.params
    assert.ok(start !== undefined, `${preset}: a thread starts`)
    assert.ok(resume !== undefined, `${preset}: the saved thread resumes`)
    // The team instructions travel on the new thread; a resumed one keeps its own.
    assert.equal(start.developerInstructions, 'You are Kai, a coder.')
    assert.equal(resume.developerInstructions, undefined, `${preset} resume: keeps the instructions it started with`)
    assert.equal(resume.threadId, records.s1.cliSessionId, `${preset} resume: resumes the saved thread`)
    assert.equal('threadId' in start, false, `${preset} start: no thread id`)
    for (const [phase, params] of [['start', start], ['resume', resume]]) {
      assert.equal(params.sandbox, sandbox, `${preset} ${phase}: sandbox`)
      assert.equal(params.approvalPolicy, approvalPolicy, `${preset} ${phase}: approvalPolicy`)
      assert.equal(params.cwd, records.s1.cwd, `${preset} ${phase}: cwd`)
      assert.equal(params.model, 'gpt-5.1-codex', `${preset} ${phase}: model`)
      const server = params.config?.mcp_servers?.dsbot
      assert.match(server?.url ?? '', /^http:\/\/127\.0\.0\.1:\d+\//, `${preset} ${phase}: MCP url`)
      assert.match(server?.http_headers?.Authorization ?? '', /^Bearer \S+$/, `${preset} ${phase}: MCP auth`)
      for (const wrong of ['sandbox_mode', 'approval_policy', 'developer_instructions', 'mcp_servers']) {
        assert.equal(wrong in params, false, `${preset} ${phase}: ${wrong} must not be sent`)
      }
    }
  }
})

// Codex 0.166 takes camelCase fields on thread/start and drops any other one
// without a word: the Bot's permission preset, the team instructions and DS
// Bot's tool server must arrive in the fields it reads.
test('codex: a thread opens with the permission preset, instructions and tool server it reads', async (t) => {
  const adapter = makeAdapter({ preset: 'read-only' })
  t.after(() => adapter.dispose())
  const out = await collect(adapter.stream(options('s1', [system('p1', 'Team rule: be brief.'), user('m1', 'show-thread')], { provider: 'local-codex-cli' })))
  const params = JSON.parse(out.text)
  assert.equal(params.sandbox, 'read-only')
  assert.equal(params.approvalPolicy, 'on-request')
  assert.match(params.developerInstructions, /be brief/)
  const server = params.config.mcp_servers.dsbot
  assert.match(server.url, /^http:\/\/127\.0\.0\.1:/)
  assert.match(server.http_headers.Authorization, /^Bearer /)
})

// The level goes on the request that starts the turn, so a Bot's choice travels
// with every turn and a Bot without one keeps the CLI's own default.
test('codex: the chosen thinking level reaches the turn', async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  const chosen = await collect(adapter.stream(options('s1', [user('m1', 'fix it')], { provider: 'local-codex-cli', reasoningEffort: 'high' })))
  assert.match(chosen.text, /effort=high/)
  const none = await collect(adapter.stream(options('s2', [user('m1', 'fix it')], { provider: 'local-codex-cli' })))
  assert.match(none.text, /effort=none/)
})

// What the model catalog shows for a codex ref is what model/list advertises.
test('the catalog carries the thinking levels a codex model serves', async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  // The menu lists a CLI's models once it has been asked for them.
  await adapter.probeModels('local-codex-cli')
  const rt = {
    ctx: {
      llm: {
        listProviders: () => [{ id: 'local-codex-cli' }],
        listModels: async () => adapter.listModels('local-codex-cli'),
        resolveModelInfo: (provider, model) => adapter.resolveModel(provider, model),
      },
    },
    state: { bots: { b1: { model: 'local-codex-cli/gpt-5.1-codex' } } },
  }
  modelsInstall(rt)
  const entry = (await rt.catalog()).find(candidate => candidate.ref === 'local-codex-cli/gpt-5.1-codex')
  assert.deepEqual(entry.reasoning.efforts.map(effort => effort.id), ['low', 'medium', 'high'])
  // The CLI's own default is not advertised: a Bot without a level keeps the
  // one the user set in their Codex config.
  assert.equal(entry.reasoning.defaultEffort, undefined)
})

// An engine that advertises no level keeps today's behaviour: none, no error.
test('an engine without levels reports none', async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  assert.equal((await adapter.resolveModel('local-claude-code', 'default')).reasoning, undefined)
  assert.equal((await adapter.resolveModel('local-acp-fake', 'default')).reasoning, undefined)
})

// Codex 0.153+ asks before every MCP tool call under on-request (#1089).
test('codex app-server: approves tool calls on DS Bot\'s own MCP server, refuses others', async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  const answer = async (sessionId, text) => JSON.parse((await collect(adapter.stream(options(sessionId, [user('m1', text)], { provider: 'local-codex-cli' })))).text)
  assert.deepEqual(await answer('s1', 'elicit dsbot mcp_tool_call'), { action: 'accept', content: null, _meta: null })
  assert.deepEqual(await answer('s2', 'elicit github mcp_tool_call'), { error: -32601 })
  assert.deepEqual(await answer('s3', 'elicit dsbot form'), { error: -32601 })
})

test('claude stream-json: thinking, activity line, usage', async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  const out = await collect(adapter.stream(options('s1', [user('m1', 'fix it')], { provider: 'local-claude-code' })))
  assert.match(out.text, /fix it/)
  assert.match(out.thinking, /thought about/)
  assert.match(out.thinking, /▸ Read/)
  assert.ok(out.usage.inputTokens === 10)
})

test('a write to a CLI that has exited does not throw out of DS Bot', async () => {
  const unheard = []
  const hear = error => unheard.push(error)
  process.on('uncaughtException', hear)
  try {
    const child = spawn(process.execPath, ['-e', 'setTimeout(() => {}, 5000)'], { stdio: ['pipe', 'pipe', 'inherit'] })
    const peer = new JsonRpcPeer(line => child.stdin.write(`${JSON.stringify(line)}\n`))
    wireProcess(child, peer)
    // A cancel that goes out as the process is let go of, as closing a Session does.
    child.kill('SIGTERM')
    for (let i = 0; i < 20; i += 1) peer.notify('session/cancel', { sessionId: 'x'.repeat(65536) })
    await new Promise(resolve => child.on('exit', resolve))
    const claude = new ClaudeEngine('local-claude-code', { name: 'Claude Code', command: process.execPath, args: [FIXTURES.claude] }, { findExecutable: path => path })
    const session = await claude.open('s1', { cwd: tmpdir() })
    const cli = session.child
    cli.kill('SIGTERM')
    for (let i = 0; i < 20; i += 1) session.write({ type: 'user', message: { role: 'user', content: [{ type: 'text', text: 'x'.repeat(65536) }] } })
    await new Promise(resolve => cli.on('exit', resolve))
    await new Promise(resolve => setTimeout(resolve, 200))
  } finally {
    process.off('uncaughtException', hear)
  }
  assert.deepEqual(unheard.map(error => error.code), [])
})

// A CLI that exists but cannot run (a broken install, a missing interpreter)
// fails spawn with 'error', not 'exit'; unheard it took the whole host down.
test('a CLI that cannot start fails its probe as a typed error, not the host', async () => {
  const broken = join(mkdtempSync(join(tmpdir(), 'dsb-broken-')), 'broken-cli')
  writeFileSync(broken, '#!/missing/interpreter/never\nexit 1\n', { mode: 0o755 })
  const unheard = []
  const hear = error => unheard.push(error)
  process.on('uncaughtException', hear)
  try {
    for (const Engine of [AcpEngine, CodexEngine]) {
      const engine = new Engine('local-broken', { name: 'Broken CLI', command: broken, args: [] }, { findExecutable: () => broken })
      await assert.rejects(() => engine.ensure(), error => error.code === 'LOCAL_AGENT_EXIT' && /could not start/.test(error.message))
      // The failed process is let go of: the next ensure() tries a fresh one.
      assert.equal(engine.peer, undefined)
      assert.equal(engine.child, undefined)
    }
  } finally {
    process.off('uncaughtException', hear)
  }
  assert.deepEqual(unheard, [])
})

test('claude: another model restarts only that conversation\'s process, with its new tool server', async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  const claude = (id, messages, extra = {}) => adapter.stream(options(id, messages, { provider: 'local-claude-code', ...extra }))
  await collect(claude('s1', [user('m1', 'one')]))
  await collect(claude('s2', [user('m1', 'two')]))
  const engine = adapter.engines.get('local-claude-code')
  const [first, other] = [engine.sessions.get('s1').child, engine.sessions.get('s2').child]
  const out = await collect(claude('s1', [user('m1', 'one'), user('m2', 'again')], { model: 'sonnet' }))
  assert.match(out.text, /again/)
  const args = engine.sessions.get('s1').child.spawnargs
  assert.notEqual(engine.sessions.get('s1').child, first)
  assert.equal(args[args.indexOf('--model') + 1], 'sonnet')
  assert.match(args[args.indexOf('--mcp-config') + 1], new RegExp(adapter.live.get('s1').mcp.url))
  assert.equal(engine.sessions.get('s2').child, other)
  assert.equal(other.killed, false)
})

// Claude Code 2.1.302 sends `system/init` only once the first message arrives,
// so waiting for it before the first prompt would deadlock every turn.
test('claude: a CLI that sends init only after the first prompt still starts', async (t) => {
  const engine = new ClaudeEngine('local-claude-code', { name: 'Claude Code', kind: 'claude', command: process.execPath, args: [FIXTURE('fake-claude-lazy-init.mjs')] }, { findExecutable: path => path })
  t.after(() => engine.close())
  const session = await Promise.race([
    engine.open('s-lazy', { cwd: mkdtempSync(join(tmpdir(), 'dsb-cwd-')), mode: 'acceptEdits' }),
    new Promise((_, reject) => { setTimeout(() => reject(new Error('start() waited for system/init')), 10_000).unref() }),
  ])
  await session.prompt([{ type: 'text', text: 'hello' }])
  const seen = []
  for (;;) {
    const event = await session.next()
    seen.push(event.type)
    if (event.type === 'done') break
  }
  assert.deepEqual(seen, ['text', 'done'])
  assert.match(session.cliSessionId, /^claude-/)
})

// The id arrives with init, after the adapter has already saved its record: the
// saved record is what a restarted host resumes from.
test('claude: the session id from init is saved and the next host resumes it', async (t) => {
  const records = {}
  const lazy = FIXTURE('fake-claude-lazy-init.mjs')
  const first = makeAdapter({ records, claude: lazy })
  t.after(() => first.dispose())
  const out = await collect(first.stream(options('s1', [user('m1', 'hello')], { provider: 'local-claude-code' })))
  assert.match(out.text, /hello/)
  assert.match(records.s1.cliSessionId, /^claude-/)
  // A new adapter is a host restart: the saved record is all it has.
  const second = makeAdapter({ records, claude: lazy })
  t.after(() => second.dispose())
  const resumed = await collect(second.stream(options('s1', [user('m2', 'again')], { provider: 'local-claude-code' })))
  assert.match(resumed.text, new RegExp(`resume=${records.s1.cliSessionId}`))
})

// A write to a CLI that dies while it starts fails asynchronously on stdin; an
// unhandled stream error takes the whole host down instead of failing one turn.
test('claude: a prompt written as the CLI dies becomes a typed error', async (t) => {
  const engine = new ClaudeEngine('local-claude-code', { name: 'Claude Code', kind: 'claude', command: process.execPath, args: [FIXTURE('fake-claude-dead.mjs')] }, { findExecutable: path => path })
  t.after(() => engine.close())
  const session = await engine.open('s-dead', { cwd: mkdtempSync(join(tmpdir(), 'dsb-cwd-')), mode: 'acceptEdits' })
  // Larger than the pipe buffer and written while the CLI is still starting, so
  // the write is pending when the process goes away.
  await session.prompt([{ type: 'text', text: 'x'.repeat(200_000) }])
  const failure = await Promise.race([
    (async () => {
      for (;;) {
        const event = await session.next()
        if (event.type === 'error' && /stopped reading input/.test(event.error.message)) return event
      }
    })(),
    new Promise((_, reject) => { setTimeout(() => reject(new Error('the stdin failure never reached the queue')), 10_000).unref() }),
  ])
  assert.equal(failure.error.code, 'LOCAL_AGENT_EXIT')
})

// A resume that dies before init must not erase the id the record already has:
// the next host would have nothing to resume and would send only the delta.
test('claude: a failed resume keeps the saved session id', async (t) => {
  const runs = join(mkdtempSync(join(tmpdir(), 'dsb-claude-')), 'runs')
  const records = {}
  const claude = FIXTURE('fake-claude-flaky.mjs')
  const host = () => {
    const adapter = makeAdapter({ records, claude, claudeArgs: [runs, '2'] })
    t.after(() => adapter.dispose())
    return adapter
  }
  const first = await collect(host().stream(options('s1', [user('m1', 'hello')], { provider: 'local-claude-code' })))
  assert.match(first.text, /hello/)
  const saved = records.s1.cliSessionId
  assert.match(saved, /^claude-/)
  // The second host resumes, and this CLI run dies before `system/init`.
  await assert.rejects(collect(host().stream(options('s1', [user('m2', 'again')], { provider: 'local-claude-code' }))), error => error.code === 'LOCAL_AGENT_EXIT')
  assert.equal(records.s1.cliSessionId, saved)
  // The third host still resumes the conversation the first one started.
  const third = await collect(host().stream(options('s1', [user('m3', 'third')], { provider: 'local-claude-code' })))
  assert.match(third.text, new RegExp(`resume=${saved}`))
})

// Claude Code deletes old conversations. Resuming one that is gone fails at
// once; the turn must start a new conversation with the history, not fail on
// the same id every time.
test('claude: a saved conversation the CLI no longer has starts over with the history', async (t) => {
  const runs = join(mkdtempSync(join(tmpdir(), 'dsb-claude-')), 'runs')
  const records = { s1: { route: 'local-claude-code', cliSessionId: 'claude-gone', cwd: tmpdir(), handedId: 'm1' } }
  const adapter = makeAdapter({ records, claude: FIXTURE('fake-claude-flaky.mjs'), claudeArgs: [runs, '', 'claude-gone'] })
  t.after(() => adapter.dispose())
  const history = [user('m1', 'fix the login'), user('m2', 'and the logout')]
  const out = await collect(adapter.stream(options('s1', history, { provider: 'local-claude-code', signal: AbortSignal.timeout(15_000) })))
  assert.equal(out.finish.kind, 'stop')
  assert.match(out.text, /resume=none/)
  assert.equal(readFileSync(runs, 'utf8'), '2')
  const prompts = readFileSync(`${runs}.prompts`, 'utf8')
  assert.match(prompts, /fix the login/, 'the new conversation gets the history')
  assert.match(prompts, /and the logout/)
  for (let i = 0; i < 40 && records.s1.cliSessionId === 'claude-gone'; i += 1) await new Promise(resolve => setTimeout(resolve, 25))
  assert.match(records.s1.cliSessionId, /^claude-(?!gone)/)
})

// A CLI that exits before init for another reason keeps its id (above); only a
// conversation the CLI reports missing is dropped.
test('claude: print mode asks DS Bot for permission and lets team tools through', async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  await collect(adapter.stream(options('s1', [user('m1', 'hello')], { provider: 'local-claude-code' })))
  const args = adapter.engines.get('local-claude-code').sessions.get('s1').child.spawnargs
  assert.equal(args[args.indexOf('--permission-prompt-tool') + 1], 'stdio')
  assert.equal(args[args.indexOf('--allowedTools') + 1], 'mcp__dsbot')
})

// A CLI that exits before init leaves a dead bridge: the retry must start a
// new process instead of writing into the one that is gone.
test('claude: a retry after a pre-init exit starts a new CLI', async (t) => {
  const runs = join(mkdtempSync(join(tmpdir(), 'dsb-claude-')), 'runs')
  const adapter = makeAdapter({ claude: FIXTURE('fake-claude-flaky.mjs'), claudeArgs: [runs, '1'] })
  t.after(() => adapter.dispose())
  await assert.rejects(collect(adapter.stream(options('s1', [user('m1', 'hello')], { provider: 'local-claude-code' }))), error => error.code === 'LOCAL_AGENT_EXIT')
  const out = await collect(adapter.stream(options('s1', [user('m2', 'again')], { provider: 'local-claude-code', signal: AbortSignal.timeout(15_000) })))
  assert.equal(out.finish.kind, 'stop')
  assert.match(out.text, /again/)
  assert.equal(readFileSync(runs, 'utf8'), '2')
})

// The failed prompt was handed but never read: the retry must send it again,
// not only what came after it.
test('claude: the retry replays the input the dead CLI never read', async (t) => {
  const runs = join(mkdtempSync(join(tmpdir(), 'dsb-claude-')), 'runs')
  const adapter = makeAdapter({ claude: FIXTURE('fake-claude-flaky.mjs'), claudeArgs: [runs, '1'] })
  t.after(() => adapter.dispose())
  const history = [user('m1', 'fix the login')]
  await assert.rejects(collect(adapter.stream(options('s1', history, { provider: 'local-claude-code' }))), error => error.code === 'LOCAL_AGENT_EXIT')
  // The same full history comes back, as DS Bot resends the conversation.
  const out = await collect(adapter.stream(options('s1', history, { provider: 'local-claude-code', signal: AbortSignal.timeout(15_000) })))
  assert.equal(out.finish.kind, 'stop')
  assert.match(readFileSync(`${runs}.prompts`, 'utf8'), /fix the login/)
})

// The repair card's Retry adds a `continue` message; the new CLI must still see
// the user message that never got an answer.
test('claude: the retry flow replays the unanswered message', async (t) => {
  const runs = join(mkdtempSync(join(tmpdir(), 'dsb-claude-')), 'runs')
  const adapter = makeAdapter({ claude: FIXTURE('fake-claude-flaky.mjs'), claudeArgs: [runs, '1'] })
  t.after(() => adapter.dispose())
  const asked = user('m1', 'fix the login')
  await assert.rejects(collect(adapter.stream(options('s1', [asked], { provider: 'local-claude-code' }))), error => error.code === 'LOCAL_AGENT_EXIT')
  const retry = {
    id: 'm2',
    role: 'user',
    content: [{ type: 'text', text: "[Retry after a failed turn]\nYour last turn failed (LOCAL_AGENT_EXIT) and the user has fixed it. Answer the user's last unanswered message." }],
    source: { kind: 'bot', form: 'relay', role: 'continue' },
  }
  const out = await collect(adapter.stream(options('s1', [asked, retry], { provider: 'local-claude-code', signal: AbortSignal.timeout(15_000) })))
  assert.equal(out.finish.kind, 'stop')
  const prompts = readFileSync(`${runs}.prompts`, 'utf8')
  assert.match(prompts, /fix the login/)
  assert.match(prompts, /last unanswered message/)
})

// The rollback is saved, not only kept in memory: a host that restarts after a
// failed start rebuilds its live state from the record.
test('claude: a host restart after a failed start still replays the unread input', async (t) => {
  const runs = join(mkdtempSync(join(tmpdir(), 'dsb-claude-')), 'runs')
  const records = {}
  const claude = FIXTURE('fake-claude-flaky.mjs')
  const host = () => {
    const adapter = makeAdapter({ records, claude, claudeArgs: [runs, '1'] })
    t.after(() => adapter.dispose())
    return adapter
  }
  const history = [user('m1', 'fix the login')]
  await assert.rejects(collect(host().stream(options('s1', history, { provider: 'local-claude-code' }))), error => error.code === 'LOCAL_AGENT_EXIT')
  // The first host is gone: the saved record is all the new one has.
  const out = await collect(host().stream(options('s1', history, { provider: 'local-claude-code', signal: AbortSignal.timeout(15_000) })))
  assert.equal(out.finish.kind, 'stop')
  assert.match(readFileSync(`${runs}.prompts`, 'utf8'), /fix the login/)
})

// ACP and Codex create their session before the prompt, so a process that exits
// after a turn did read it: reopening sends only the new input, not a replay.
test('acp: a reopen after the process exits sends only the new input', async (t) => {
  const adapter = makeAdapter({ acpArgs: ['--exit-after-turn'] })
  t.after(() => adapter.dispose())
  const first = await collect(adapter.stream(options('s1', [user('m1', 'alpha-one')])))
  assert.match(first.text, /alpha-one/)
  await waitForDead(adapter)
  const second = await collect(adapter.stream(options('s1', [user('m1', 'alpha-one'), user('m2', 'beta-two')])))
  assert.match(second.text, /beta-two/)
  assert.doesNotMatch(second.text, /alpha-one/)
})

// A reopened Codex thread needs its own queue: a pump still waiting on the dead
// one takes the first event of the new turn, and the reply arrives empty.
test('codex: a resume after the process exits still delivers the reply text', async (t) => {
  const adapter = makeAdapter({ codexArgs: ['--exit-after-turn', '--text-only'] })
  t.after(() => adapter.dispose())
  const first = await collect(adapter.stream(options('s1', [user('m1', 'alpha-one')], { provider: 'local-codex-cli' })))
  assert.match(first.text, /alpha-one/)
  await waitForDead(adapter)
  const second = await collect(adapter.stream(options('s1', [user('m1', 'alpha-one'), user('m2', 'beta-two')], { provider: 'local-codex-cli' })))
  assert.match(second.text, /beta-two/)
  assert.doesNotMatch(second.text, /alpha-one/)
})

test('MCP: bad token gets 401, tools/list flattens schemas', async () => {
  const mcp = new LoopbackMcp(() => [{ name: 'message_bot', description: 'd', parameters: { type: 'object', properties: { x: { anyOf: [{ type: 'string' }, { type: 'number' }] } }, required: ['x'] } }], async () => ({ text: 'done' }))
  await mcp.listen()
  const url = mcp.url
  const bad = await fetch(url, { method: 'POST', headers: { 'content-type': 'application/json', authorization: 'Bearer wrong' }, body: '{}' })
  assert.equal(bad.status, 401)
  const good = await fetch(url, { method: 'POST', headers: { 'content-type': 'application/json', authorization: `Bearer ${mcp.token}` }, body: JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'tools/list', params: {} }) })
  const list = (await good.json()).result
  assert.equal(list.tools[0].name, 'message_bot')
  assert.equal(list.tools[0].inputSchema.properties.x.type, 'string')
  mcp.close()
})

test('the adapter\'s MCP server lists the current turn\'s tools (#1061)', async () => {
  const adapter = makeAdapter()
  const live = { pending: new Map(), push: () => {} }
  const list = async (mcp) => {
    const response = await fetch(mcp.url, { method: 'POST', headers: { 'content-type': 'application/json', authorization: `Bearer ${mcp.token}` }, body: JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'tools/list', params: {} }) })
    assert.equal(response.status, 200)
    return (await response.json()).result.tools.map(tool => tool.name)
  }
  const mcp = await adapter.ensureMcp(live, () => [{ name: 'message_bot', parameters: { type: 'object', properties: {} } }])
  try {
    assert.deepEqual(await list(mcp), ['message_bot'])
    const again = await adapter.ensureMcp(live, () => [{ name: 'post_to_group', parameters: { type: 'object', properties: {} } }])
    assert.equal(again, mcp, 'one server per live session')
    assert.deepEqual(await list(mcp), ['post_to_group'])
  } finally {
    mcp.close()
    // fetch keeps the socket alive; on Windows a forced exit with it still
    // closing trips a libuv assertion.
    mcp.server.closeAllConnections()
  }
})

test('flattenSchema removes unions, refs and const', () => {
  const flat = flattenSchema({ type: 'object', $defs: { a: { type: 'string' } }, properties: { p: { $ref: '#/$defs/a' }, q: { const: 'x' }, r: { oneOf: [{ type: 'null' }, { type: 'number' }] } } })
  assert.equal(flat.$defs, undefined)
  assert.equal(flat.properties.p.$ref, undefined)
  assert.equal(flat.properties.q.const, undefined)
  assert.equal(flat.properties.r.type, 'number')
})

// The steward must never touch an abot's Session: the pre-step listener returns
// early for a local-agent Bot, and checkContext leaves too.
// A CLI condenses on its own inside a running turn: its part is checked when the turn
// ends (test/local-agent-context.test.js), never before a step.
test('a running turn of a local-agent part is not checked before its steps', async () => {
  const ctx = fakeContext()
  const bot = { id: 'b1', name: 'Kai', sessionId: 's1', model: 'local-acp-fake/default' }
  const rt = {
    ctx,
    config: {},
    state: { bots: { b1: bot }, rooms: {}, prefs: {}, exchanges: [] },
    warnings: [],
    warn() {},
    async save() {},
    async load() {},
    readOnly: () => undefined,
    botOf: id => rt.state.bots[id],
    roomOf: () => undefined,
    selfOf: () => 'b1',
    chatOf: () => 's1',
    isBotSession: id => id === 's1',
    isLocalAgentRef: ref => typeof ref === 'string' && ref.startsWith('local-acp-'),
    isLocalAgentRoute: route => route.startsWith('local-acp-'),
    localAgentContext: () => ({ used: 999_999, size: 200_000, compactions: 0 }),
    chatOwners: new Map([['s1', 'b1']]),
    groupOwners: new Map(),
    pastOwners: new Map(),
    roomOwners: new Map(),
    partsOf: () => [],
    resolveAgent: async id => ({ id, session: { id, deriveMessages: () => [] }, status: 'idle' }),
    teamTools: [],
  }
  partsInstall(rt)
  const listener = ctx.listeners.get('agent/pre-step')?.[0]
  assert.ok(listener !== undefined)
  const agent = { id: 's1', session: { id: 's1', deriveMessages: () => [], requestHeader: () => ({ config: { provider: 'local-acp-fake', model: 'default' } }) }, status: 'idle' }
  const next = async () => 'next-called'
  const result = await listener({ agent, messages: [] }, next)
  assert.equal(result, 'next-called')
})

// A manual new part gives the deterministic handoff: verbatim tail + anchors +
// the read_own_chat line, with no model call.
test('a fresh part carries the deterministic handoff', async () => {
  const ctx = fakeContext()
  ctx.workspaceRegistry = { archiveSession: async () => {}, unarchiveSession: async () => {} }
  ctx.sessions = { flush: async () => {} }
  const messages = Array.from({ length: 14 }, (_, i) => ({ seq: i + 1, who: i % 2 ? 'you' : 'user', text: `line ${i + 1}` }))
  const bot = { id: 'b1', name: 'Kai', sessionId: 's1', model: 'local-acp-fake/default', segments: [] }
  const followed = []
  const delivered = []
  const rt = {
    ctx,
    config: {},
    state: { bots: { b1: bot }, rooms: {}, prefs: {}, exchanges: [], capabilities: { items: {} }, switching: {}, questions: {}, views: {} },
    warn: (...args) => rt.warnings.push(args.join(' ')),
    warnings: [],
    async save() {},
    async load() {},
    readOnly: () => undefined,
    botOf: id => rt.state.bots[id],
    roomOf: () => undefined,
    selfOf: () => 'b1',
    isMain: () => false,
    chatOf: () => 's1',
    sessionOfRoom: () => undefined,
    isBotSession: id => ['s1', 's2'].includes(id),
    isLocalAgentRef: ref => typeof ref === 'string' && ref.startsWith('local-acp-'),
    chatOwners: new Map([['s1', 'b1']]),
    groupOwners: new Map(),
    pastOwners: new Map(),
    roomOwners: new Map(),
    teamTools: [],
    live: new Map(),
    groupSessionsOf: () => [],
    unreadOf: () => [],
    clock: () => 'now',
    partsOf: () => (bot.segments ?? []).map(segment => segment.sessionId),
    partLines: async function* () { for (const line of messages) yield line },
    eventsBackwards: async function* () {
      for (const line of [...messages].reverse()) {
        yield line.who === 'user'
          ? { type: 'user/message', time: line.seq, seq: line.seq, data: { content: [{ type: 'text', text: line.text }], source: { kind: 'user' } } }
          : { type: 'assistant/message', time: line.seq, seq: line.seq, data: { message: { content: [{ type: 'text', text: line.text }] } } }
      }
    },
    historyEntry: () => undefined,
    displayItems: async () => [],
    earlierPage: async () => ({}),
    chatPage: async () => ({}),
    noteText: () => '',
    followFolder: async () => {},
    heldBack: () => [],
    resolveAgent: async id => ({ id, session: { id, deriveMessages: () => messages.map(line => ({ role: line.who === 'user' ? 'user' : 'assistant', content: [{ type: 'text', text: line.text }] })) }, status: 'idle', followup: (message) => followed.push(message), whenIdle: async () => {}, runMaintenance: async fn => fn(AbortSignal.timeout(5_000)), inbox: { nextStep: [], nextTurn: [] }, send: async () => {} }),
    createSession: async () => 's2',
    desiredModel: async bot => bot.model,
    desiredEffort: async () => undefined,
    selectSessionModel: async () => {},
    applyModel: async () => {},
    applyPermission: async () => {},
    ensureCapabilities: async () => {},
    renewMemory: () => {},
    moveSchedules: async () => {},
    sessionTitle: { rename: async () => {} },
    deliver: async (message) => delivered.push(message),
    teamFolder: () => '/tmp/team',
    workDirOf: () => '/tmp/team',
    isFolder: async () => true,
    uiLanguage: () => 'en',
    roomProgress: new Map(),
    resolveModelInfo: async () => ({}),
    llmRequests: new Map(),
  }
  partsInstall(rt)
  assert.equal(typeof rt.newPart, 'function')
  await rt.newPart('s1')
  // The switch runs as a job; the Bot lands on s2 with s1 kept as an earlier part.
  for (let i = 0; i < 50 && bot.sessionId !== 's2'; i += 1) await new Promise(resolve => setTimeout(resolve, 20))
  assert.equal(bot.sessionId, 's2', rt.warnings.join(' | '))
  assert.equal(bot.segments.length, 1)
  assert.equal(bot.segments[0].sessionId, 's1')
  const text = [...followed, ...delivered].map(message => message?.content?.map?.(block => block.text).join('') ?? message?.text ?? String(message)).join('\n')
  assert.match(text, /fresh conversation was started|previous part|read_own_chat/i)
})

test('an open-ended request outlives the default deadline', async () => {
  const peer = new JsonRpcPeer(() => {})
  const open = peer.request('session/prompt', {}, { timeout: 0 })
  const deadline = peer.request('session/load', {}, { timeout: 40 })
  const settled = deadline.then(() => 'ok', () => 'timeout')
  await new Promise(resolve => setTimeout(resolve, 90))
  assert.equal(await settled, 'timeout', 'a positive timeout still fires')
  peer.handleLine(JSON.stringify({ jsonrpc: '2.0', id: 1, result: { stopReason: 'end_turn' } }))
  assert.deepEqual(await open, { stopReason: 'end_turn' }, 'a 0 timeout never expires')
})

// #1513: a card that suspends the stream can outlive the CLI's turn. The answer must
// start a turn of its own, and the finished turn's leftover events must not answer it.
test('codex app-server: an answer whose turn already ended starts a new turn', async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  const messages = [user('m1', 'late-ask')]
  const first = await collect(adapter.stream(options('s1', messages, { provider: 'local-codex-cli' })))
  const call = first.calls.find(entry => entry.name === 'approve_agent_action')
  assert.ok(call, 'the approval came as a card')
  // The CLI's turn completed while the card was open: its `done` is still queued.
  await new Promise(resolve => setTimeout(resolve, 100))
  const second = await collect(adapter.stream(options('s1', [...messages, toolResult('t1', call.id, '{"decision":"allow"}', false)], { provider: 'local-codex-cli' })))
  // The answer opened a turn of its own, and the stream says what that turn said.
  assert.match(second.text, /ok: /, JSON.stringify(second))
  assert.equal(second.finish?.kind, 'stop')
})

test('codex app-server: a steer refused while the turn ends still gets a turn of its own', async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  const messages = [user('m1', 'hold-ask')]
  const first = await collect(adapter.stream(options('s1', messages, { provider: 'local-codex-cli' })))
  assert.ok(first.calls.some(entry => entry.name === 'approve_agent_action'), 'the card came through')
  // The CLI refuses the steer and completes the turn in the same breath.
  const second = await collect(adapter.stream(options('s1', [...messages, user('m2', 'race')], { provider: 'local-codex-cli' })))
  assert.match(second.text, /ok: race/, JSON.stringify(second))
})

test('codex app-server: text answering a bridged call after the turn ended starts one', async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  const messages = [user('m1', 'late-ask')]
  const first = await collect(adapter.stream(options('s1', messages, { provider: 'local-codex-cli' })))
  assert.ok(first.calls.some(entry => entry.name === 'approve_agent_action'), 'the card came through')
  // A loopback call is still open — browser_handoff and friends look like this — and
  // the user answers it with plain text after the CLI's turn already ended.
  const live = adapter.live.get('s1')
  const answered = []
  live.pending.set('bridged-1', { name: 'browser_handoff', resolve: result => answered.push(result) })
  const second = await collect(adapter.stream(options('s1', [...messages, user('m2', 'yes, go ahead')], { provider: 'local-codex-cli' })))
  assert.deepEqual(answered, [{ text: 'yes, go ahead' }], 'the answer reached the bridged call')
  assert.match(second.text, /ok: The calls that were waiting on the user are settled/, JSON.stringify(second))
})

test('codex app-server: a late notice for an earlier turn keeps the running turn steerable', async (t) => {
  const recordPath = join(mkdtempSync(join(tmpdir(), 'dsb-late-')), 'record.jsonl')
  const engine = new CodexEngine('local-codex-cli', { name: 'Codex', command: process.execPath, args: [FIXTURES.codex, '--record', recordPath] }, { findExecutable: path => path })
  t.after(() => { engine.close?.() })
  const thread = await engine.open('s1', { cwd: mkdtempSync(join(tmpdir(), 'dsb-late-cwd-')), mode: 'workspace-write' })
  const first = await engine.prompt(thread, [{ type: 'text', text: 'hold' }], {})
  const second = await engine.prompt(thread, [{ type: 'text', text: 'hold' }], {})
  assert.notEqual(first, second)
  // The turn a steer replaced finishes late: its notice must not clear the live turn.
  await engine.steer(thread, `emit-stale ${first}`)
  assert.equal(engine.canSteer(thread), true, 'the running turn stays steerable')
  await engine.steer(thread, 'next input')
  const lines = readFileSync(recordPath, 'utf8').trim().split('\n').filter(line => line !== '').map(line => JSON.parse(line))
  const starts = lines.filter(entry => entry.method === 'turn/start')
  assert.equal(starts.length, 2, 'no turn was started to replace the live one')
  const steer = lines.filter(entry => entry.method === 'turn/steer').at(-1)
  assert.equal(steer.params.expectedTurnId, second, 'the next input steers the running turn')
})

test('codex app-server: the approval answer rides along with runtime context in the new turn', async (t) => {
  const recordPath = join(mkdtempSync(join(tmpdir(), 'dsb-settled-')), 'record.jsonl')
  const adapter = makeAdapter({ codexArgs: ['--record', recordPath] })
  t.after(() => adapter.dispose())
  const messages = [user('m1', 'late-ask')]
  const first = await collect(adapter.stream(options('s1', messages, { provider: 'local-codex-cli' })))
  const call = first.calls.find(entry => entry.name === 'approve_agent_action')
  assert.ok(call, 'the approval came as a card')
  // The CLI's turn is over. The answer arrives in the same call as runtime context, which
  // used to become the whole prompt and leave the decision behind.
  const context = { id: 'c1', role: 'user', content: [{ type: 'text', text: 'workspace note: the folder is /team' }], source: { kind: 'runtime-context' } }
  await collect(adapter.stream(options('s1', [...messages, toolResult('t1', call.id, '{"decision":"allow"}', false), context], { provider: 'local-codex-cli' })))
  const lines = readFileSync(recordPath, 'utf8').trim().split('\n').filter(line => line !== '').map(line => JSON.parse(line))
  const starts = lines.filter(entry => entry.method === 'turn/start')
  const last = starts.at(-1)
  const text = (last?.params?.input ?? []).map(item => item.text ?? '').join('\n')
  assert.match(text, /\{"decision":"allow"\}/, `the decision reached the turn: ${text}`)
  assert.match(text, /workspace note: the folder is \/team/, `the context reached the turn: ${text}`)
})

test('codex app-server: a steer names the turn it steers, and a finished turn cannot be steered', async (t) => {
  const recordPath = join(mkdtempSync(join(tmpdir(), 'dsb-steer-')), 'record.jsonl')
  const engine = new CodexEngine('local-codex-cli', { name: 'Codex', command: process.execPath, args: [FIXTURES.codex, '--record', recordPath] }, { findExecutable: path => path })
  t.after(() => { engine.close?.() })
  const thread = await engine.open('s1', { cwd: mkdtempSync(join(tmpdir(), 'dsb-steer-cwd-')), mode: 'workspace-write' })
  const turnId = await engine.prompt(thread, [{ type: 'text', text: 'hold' }], {})
  assert.equal(engine.canSteer(thread), true, 'the turn is live')
  await engine.steer(thread, 'steer me')
  const lines = readFileSync(recordPath, 'utf8').trim().split('\n').filter(line => line !== '').map(line => JSON.parse(line))
  const steer = lines.find(entry => entry.method === 'turn/steer')
  assert.ok(steer, 'a steer was sent')
  assert.equal(steer.params.expectedTurnId, turnId, 'the steer names the turn it steers')
  assert.equal(steer.params.input[0].text, 'steer me')
  // A turn the CLI ends has nothing to steer: it says so instead of dropping the input.
  await engine.prompt(thread, [{ type: 'text', text: 'bye' }], {})
  // The CLI's own turn/completed is the signal: read until it arrives, not for a clock.
  const deadline = Date.now() + 30_000
  for (;;) {
    const event = await Promise.race([
      thread.next(),
      new Promise((_, reject) => setTimeout(() => reject(new Error('no turn/completed')), 30_000)),
    ])
    if (event.type === 'done') break
    if (Date.now() > deadline) throw new Error(`no turn/completed: ${JSON.stringify(event)}`)
  }
  // The engine clears its turn off the same notice; poll that, never wait for a second
  // event that will not come.
  for (let attempt = 0; attempt < 300 && engine.canSteer(thread); attempt += 1) {
    await new Promise(resolve => setTimeout(resolve, 10))
  }
  assert.equal(engine.canSteer(thread), false, 'turn/completed cleared the live turn')
  await assert.rejects(() => engine.steer(thread, 'too late'), /already ended/)
})


test('Codex completion in the same JSON-RPC batch cannot resurrect a live turn', async () => {
  const engine = new CodexEngine('local-codex-cli', {}, { findExecutable: value => value })
  const thread = engine.threadFor('s1')
  thread.threadId = 'thread'
  const peer = new JsonRpcPeer(message => {
    peer.handleLine(JSON.stringify({ jsonrpc: '2.0', id: message.id, result: { turn: { id: 'completed' } } }))
    peer.handleLine(JSON.stringify({ jsonrpc: '2.0', method: 'turn/completed', params: { threadId: 'thread', turn: { id: 'completed', status: 'completed' } } }))
  })
  peer.notificationHandler = (method, params) => engine.onNotice(method, params)
  engine.ensure = async () => peer
  assert.equal(await engine.prompt(thread, [{ type: 'text', text: 'go' }], {}), 'completed')
  assert.equal(engine.canSteer(thread), false)
})

test('a late ACP decision discards old output and drains a continuation that asks again', { timeout: 5000 }, async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  const messages = [user('m1', 'late-permission')]
  const first = await collect(adapter.stream(options('s1', messages)))
  await new Promise(resolve => setTimeout(resolve, 50))
  messages.push(toolResult('m2', first.calls[0].id, '{"decision":"allow"}'))
  const second = await collect(adapter.stream(options('s1', messages)))
  assert.equal(second.finish.kind, 'tool-calls')
  assert.equal(second.calls[0].name, 'approve_agent_action')
  assert.doesNotMatch(second.text, /old output/)
  messages.push(toolResult('m3', second.calls[0].id, '{"decision":"allow"}'))
  const third = await collect(adapter.stream(options('s1', messages)))
  assert.match(third.text, /allowed via allow-once/)
})

test('replacing a completed Codex turn retains another queued approval', { timeout: 5000 }, async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  const messages = [user('m1', 'late-double')]
  const first = await collect(adapter.stream(options('s1', messages, { provider: 'local-codex-cli' })))
  await new Promise(resolve => setTimeout(resolve, 50))
  messages.push(toolResult('m2', first.calls[0].id, '{"decision":"allow"}'))
  const second = await collect(adapter.stream(options('s1', messages, { provider: 'local-codex-cli' })))
  assert.equal(second.calls[0].name, 'approve_agent_action')
  assert.notEqual(second.calls[0].id, first.calls[0].id)
})

test('antigravity route, badge, avatar, modes, sign-in hint, and env stripping', async (t) => {
  const spec = ENGINES['local-antigravity']
  assert.equal(spec.command, 'agy')
  assert.deepEqual(spec.args, ['acp'])
  assert.equal(spec.signIn, 'agy acp --login')
  assert.equal(spec.cwdPerProcess, true)
  const listedModes = ['default', 'accept-edits', 'auto', 'plan'].map(id => ({ id }))
  assert.equal(acpMode('read-only', listedModes, spec.modes), 'default')
  assert.equal(acpMode('workspace-write', listedModes, spec.modes), 'accept-edits')
  assert.equal(acpMode('danger-full-access', listedModes, spec.modes), 'auto')
  assert.equal(isLocalRoute('local-antigravity/gemini-4-carbon'), true)
  assert.deepEqual(badgeOf('local-antigravity'), ['AG', 'sky'])
  assert.match(agentAvatar('local-antigravity').image, /^data:image\/png;base64,/)

  const prevAlias = process.env.ANTIGRAVITY_CLI_ALIAS
  const prevAcc = process.env.AGY_ACCOUNT
  process.env.ANTIGRAVITY_CLI_ALIAS = 'custom'
  process.env.AGY_ACCOUNT = 'secret'
  t.after(() => {
    if (prevAlias === undefined) delete process.env.ANTIGRAVITY_CLI_ALIAS; else process.env.ANTIGRAVITY_CLI_ALIAS = prevAlias
    if (prevAcc === undefined) delete process.env.AGY_ACCOUNT; else process.env.AGY_ACCOUNT = prevAcc
  })
  const cleaned = cleanEnv()
  assert.equal(cleaned.ANTIGRAVITY_CLI_ALIAS, undefined)
  assert.equal(cleaned.AGY_ACCOUNT, undefined)

  const adapter = makeAdapter({ acpArgs: ['--omit-mcp-caps', '--auth-required'] })
  t.after(() => adapter.dispose())
  await assert.rejects(
    collect(adapter.stream(options('s1', [user('m1', 'hi')], { provider: 'local-antigravity' }))),
    err => err.code === 'LOCAL_AGENT_SIGNIN' && /agy acp --login/.test(err.message),
  )
})

test('acp tries HTTP MCP when mcpCapabilities is omitted (#1073), bridges stdio MCP when HTTP is unsupported/rejected, and falls back to plain on total rejection (#1463)', async (t) => {
  const omit = makeAdapter({ acpArgs: ['--omit-mcp-caps'] })
  t.after(() => omit.dispose())
  const call = await collect(omit.stream(options('s1', [user('m1', 'call-tool')])))
  assert.equal(call.calls[0]?.name, 'message_bot')
  const withTool = await collect(omit.stream(options('s1', [user('m1', 'call-tool'), toolResult('m2', call.calls[0].id, 'delivered to Chief')])))
  assert.match(withTool.text, /tool said: delivered to Chief/)

  const stdioOnly = makeAdapter({ acpArgs: ['--no-http-mcp'] })
  t.after(() => stdioOnly.dispose())
  const stdio1 = await collect(stdioOnly.stream(options('s-stdio', [user('m1', 'two-calls')])))
  assert.equal(stdio1.calls.length, 1)
  assert.equal(stdio1.calls[0]?.name, 'message_bot')
  const msg1 = JSON.parse(stdio1.calls[0].arguments).message
  const reply1 = msg1 === 'hi 0' ? 'zero' : 'one'
  const stdio2 = await collect(stdioOnly.stream(options('s-stdio', [
    user('m1', 'two-calls'),
    toolResult('m2', stdio1.calls[0].id, reply1),
  ])))
  assert.equal(stdio2.calls.length, 1)
  const msg2 = JSON.parse(stdio2.calls[0].arguments).message
  assert.notEqual(msg1, msg2)
  const reply2 = msg2 === 'hi 0' ? 'zero' : 'one'
  const stdioDone = await collect(stdioOnly.stream(options('s-stdio', [
    user('m1', 'two-calls'),
    toolResult('m2', stdio1.calls[0].id, reply1),
    toolResult('m3', stdio2.calls[0].id, reply2),
  ])))
  assert.match(stdioDone.text, /tool said: zero \| one/)

  const records = {}
  const httpRejected1 = makeAdapter({ records, acpArgs: ['--omit-mcp-caps', '--reject-http-mcp'] })
  t.after(() => httpRejected1.dispose())
  const fbCall = await collect(httpRejected1.stream(options('s-fb', [user('m1', 'call-tool')])))
  assert.equal(fbCall.calls[0]?.name, 'message_bot')
  const fbDone = await collect(httpRejected1.stream(options('s-fb', [user('m1', 'call-tool'), toolResult('m2', fbCall.calls[0].id, 'via stdio fallback')])))
  assert.match(fbDone.text, /tool said: via stdio fallback/)
  await httpRejected1.dispose()

  const httpRejected2 = makeAdapter({ records, acpArgs: ['--omit-mcp-caps', '--reject-http-mcp'] })
  t.after(() => httpRejected2.dispose())
  const resumedCall = await collect(httpRejected2.stream(options('s-fb', [
    user('m1', 'call-tool'),
    toolResult('m2', fbCall.calls[0].id, 'via stdio fallback'),
    user('m3', 'call-tool'),
  ])))
  assert.equal(resumedCall.calls[0]?.name, 'message_bot')
  const resumedDone = await collect(httpRejected2.stream(options('s-fb', [
    user('m1', 'call-tool'),
    toolResult('m2', fbCall.calls[0].id, 'via stdio fallback'),
    user('m3', 'call-tool'),
    toolResult('m4', resumedCall.calls[0].id, 'resumed stdio'),
  ])))
  assert.match(resumedDone.text, /tool said: resumed stdio/)

  const rejected = makeAdapter({ acpArgs: ['--omit-mcp-caps', '--reject-mcp'] })
  t.after(() => rejected.dispose())
  const fallback = await collect(rejected.stream(options('s2', [user('m1', 'hello')])))
  assert.match(fallback.text, /^ok: [\s\S]*hello/)
})

test('LoopbackMcp answers JSON-RPC notifications with 202 and an empty body', async (t) => {
  const mcp = new LoopbackMcp(() => [], async () => ({ text: 'ok' }))
  await mcp.listen()
  t.after(() => mcp.close())
  const headers = { 'content-type': 'application/json', authorization: `Bearer ${mcp.token}` }
  for (const payload of [
    { jsonrpc: '2.0', method: 'notifications/initialized' },
    { jsonrpc: '2.0', method: 'notifications/roots/list_changed' },
    [{ jsonrpc: '2.0', method: 'notifications/initialized' }],
  ]) {
    const res = await fetch(mcp.url, { method: 'POST', headers, body: JSON.stringify(payload) })
    assert.equal(res.status, 202)
    assert.equal(await res.text(), '')
  }
})

test('acp exposes per-model thought_level efforts, maps Antigravity presets, and switches/restores them per turn', async (t) => {
  let currentPreset = 'workspace-write'
  const adapter = makeAdapter({ acpArgs: ['--agy-modes', '--thought-levels'], preset: () => currentPreset })
  t.after(() => adapter.dispose())
  const catalog = await adapter.probeModels('local-antigravity')
  const large = catalog.find(m => m.id === 'fake-large')
  const small = catalog.find(m => m.id === 'fake-small')
  assert.deepEqual(large.reasoning.efforts.map(e => e.id), ['low', 'medium', 'high', 'xhigh'])
  assert.equal(large.reasoning.defaultEffort, undefined)
  assert.equal(small.reasoning, undefined)

  const modeWrite = await collect(adapter.stream(options('m0', [user('m1', 'show-mode')], { provider: 'local-antigravity', model: 'fake-large' })))
  assert.match(modeWrite.text, /mode: accept-edits; refused: 0/)
  currentPreset = 'read-only'
  const modeRead = await collect(adapter.stream(options('m0', [user('m1', 'show-mode'), user('m2', 'show-mode')], { provider: 'local-antigravity', model: 'fake-large' })))
  assert.match(modeRead.text, /mode: default; refused: 0/)
  currentPreset = 'danger-full-access'
  const modeAuto = await collect(adapter.stream(options('m0', [user('m1', 'show-mode'), user('m2', 'show-mode'), user('m3', 'show-mode')], { provider: 'local-antigravity', model: 'fake-large' })))
  assert.match(modeAuto.text, /mode: auto; refused: 0/)

  const high = await collect(adapter.stream(options('s1', [user('m1', 'show-state')], {
    provider: 'local-antigravity',
    model: 'fake-large',
    reasoningEffort: 'high',
  })))
  assert.match(high.text, /effort: high/)

  const restored = await collect(adapter.stream(options('s1', [user('m1', 'show-state'), user('m2', 'show-state')], {
    provider: 'local-antigravity',
    model: 'fake-large',
    reasoningEffort: undefined,
  })))
  assert.match(restored.text, /effort: medium/)
})

test('cwdPerProcess isolates ACP child cwd per folder, shares within a folder, and closes probe workers', async (t) => {
  const dirA = realpathSync(mkdtempSync(join(tmpdir(), 'dsb-agy-a-')))
  const dirB = realpathSync(mkdtempSync(join(tmpdir(), 'dsb-agy-b-')))
  const cwds = { sA1: dirA, sA2: dirA, sB: dirB }
  const spawned = []
  const adapter = new LocalAgentAdapter({
    rt: { warn: () => {} },
    customAgents: () => [],
    findExecutable: path => path,
    spawnEngine: (_kind, route, spec) => {
      const engine = new AcpEngine(route, { ...spec, command: process.execPath, args: [FIXTURES.acp, '--agy-modes', '--thought-levels'] }, { findExecutable: path => path })
      spawned.push(engine)
      return engine
    },
    record: () => undefined,
    saveRecord: () => {},
    presetFor: () => 'workspace-write',
    cwdFor: async sessionId => cwds[sessionId] ?? dirA,
    probeCwd: () => dirA,
    toolsFor: () => [],
  })
  t.after(() => adapter.dispose())

  // Probe creates a temporary worker and closes it once models are read.
  const models = await adapter.probeModels('local-antigravity')
  assert.ok(models.length >= 2)
  assert.equal(spawned[0].workers.size, 0)

  const outA1 = await collect(adapter.stream(options('sA1', [user('m1', 'show-state')], { provider: 'local-antigravity', model: 'fake-large' })))
  const outB = await collect(adapter.stream(options('sB', [user('m1', 'show-state')], { provider: 'local-antigravity', model: 'fake-large' })))
  const outA2 = await collect(adapter.stream(options('sA2', [user('m1', 'show-state')], { provider: 'local-antigravity', model: 'fake-large' })))
  assert.match(outA1.text, new RegExp(`cwd: ${dirA.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`))
  assert.match(outB.text, new RegExp(`cwd: ${dirB.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`))
  assert.match(outA2.text, new RegExp(`cwd: ${dirA.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`))
  assert.equal(spawned[0].workers.size, 2)
})

test('every built-in CLI in ENGINES has valid metadata, badge, valid ink, avatar, and mode mapping', () => {
  const routes = Object.keys(ENGINES)
  assert.ok(routes.length >= 24)
  for (const route of routes) {
    const spec = ENGINES[route]
    const cli = route.replace(/^local-/, '')
    assert.ok(['acp', 'claude', 'codex'].includes(spec.kind), `${route} kind`)
    assert.ok(typeof spec.name === 'string' && spec.name.length > 0, `${route} name`)
    assert.ok(typeof spec.command === 'string' && spec.command.length > 0, `${route} command`)
    assert.ok(typeof spec.signIn === 'string' && spec.signIn.length > 0, `${route} signIn`)
    const badge = AGENT_BADGES[cli]
    assert.ok(Array.isArray(badge) && badge.length === 2, `${route} AGENT_BADGES`)
    assert.ok(INKS[badge[1]] !== undefined, `${route} ink ${badge[1]}`)
    assert.ok(isLocalRoute(route), `${route} isLocalRoute`)
    assert.deepEqual(badgeOf(route), badge, `${route} badgeOf`)
    const avatar = agentAvatar(route)
    assert.ok(avatar?.image?.startsWith('data:image/'), `${route} agentAvatar`)
    if (spec.kind === 'acp') {
      assert.equal(typeof acpMode('workspace-write', spec.modes?.['workspace-write'] ?? ['accept-edits'], spec.modes), 'string')
    } else {
      assert.equal(typeof cliMode(spec.kind, 'workspace-write'), 'string')
    }
  }
})

test('AcpEngine.ensure deduplicates concurrent startups and cleans up after failed initialize', async (t) => {
  let findCalls = 0
  const engine = new AcpEngine('local-test', {
    name: 'Test ACP',
    kind: 'acp',
    command: process.execPath,
    args: [FIXTURES.acp, '--fail-first-init'],
  }, {
    findExecutable: (name) => { findCalls += 1; return name },
  })
  t.after(() => engine.close())

  await assert.rejects(() => Promise.all([engine.ensure(), engine.ensure()]), /init boom/)
  assert.equal(findCalls, 1)
  assert.equal(engine.child, undefined)
  assert.equal(engine.peer, undefined)
  assert.equal(engine.starting, undefined)

  engine.engine = { ...engine.engine, args: [FIXTURES.acp] }
  const [peerA, peerB] = await Promise.all([engine.ensure(), engine.ensure()])
  assert.equal(peerA, peerB)
  assert.equal(findCalls, 2)
})

test('ownToolCall auto-approves generic mcp__dsbot__ and dsbot: calls without leaking to unrelated names', async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())

  for (const trigger of ['generic-mcp-team', 'generic-mcp-team-colon']) {
    const step1 = await collect(adapter.stream(options(`team-${trigger}`, [user('m1', trigger)])))
    assert.equal(step1.calls[0]?.name, 'message_bot', JSON.stringify(step1))
    const step2 = await collect(adapter.stream(options(`team-${trigger}`, [
      user('m1', trigger),
      toolResult('m2', step1.calls[0].id, 'delivered'),
    ])))
    assert.match(step2.text, /allowed via allow_once; tool said: delivered/)
  }

  for (const trigger of ['generic-mcp-other-github', 'generic-mcp-other-prefix']) {
    const asked = await collect(adapter.stream(options(`other-${trigger}`, [user('m1', trigger)])))
    assert.equal(asked.calls[0]?.name, 'approve_agent_action', JSON.stringify(asked))
  }
})

test('ClaudeEngine can_use_tool deny sends required message string (#1539)', async (t) => {
  const adapter = makeAdapter()
  t.after(() => adapter.dispose())
  const ask = await collect(adapter.stream(options('c1', [user('m1', 'ask-claude-perm')], { provider: 'local-claude-code' })))
  assert.equal(ask.calls[0]?.name, 'approve_agent_action')
  const denied = await collect(adapter.stream(options('c1', [
    user('m1', 'ask-claude-perm'),
    toolResult('m2', ask.calls[0].id, JSON.stringify({ decision: 'deny' })),
  ], { provider: 'local-claude-code' })))
  assert.equal(denied.text, 'perm:{"behavior":"deny","message":"The user denied this action."}')
})


test('stdioBridgePath extracts .asar bridges into a private directory and leaves unpacked files alone', (t) => {
  const dir = mkdtempSync(join(tmpdir(), 'dsb-asar-'))
  t.after(() => rmSync(dir, { recursive: true, force: true }))
  const asarDir = join(dir, 'app.asar')
  const unpackedDir = join(dir, 'app.asar.unpacked')
  mkdirSync(asarDir, { recursive: true })
  mkdirSync(unpackedDir, { recursive: true })
  const asarSource = join(asarDir, 'mcp-stdio.js')
  const unpackedSource = join(unpackedDir, 'mcp-stdio.js')
  writeFileSync(asarSource, '// bridge\n')
  writeFileSync(unpackedSource, '// unpacked\n')

  const extracted = stdioBridgePath(asarSource)
  t.after(() => rmSync(dirname(extracted), { recursive: true, force: true }))
  assert.notEqual(extracted, asarSource)
  assert.equal(stdioBridgePath(asarSource), extracted)
  assert.equal(readFileSync(extracted, 'utf8'), '// bridge\n')
  if (process.platform !== 'win32') {
    assert.equal(statSync(dirname(extracted)).mode & 0o777, 0o700)
    assert.equal(statSync(extracted).mode & 0o777, 0o600)
  }
  assert.equal(stdioBridgePath(unpackedSource), unpackedSource)
})

test('ACP session/load falls back to stdio or plain sessions without dropping cliSessionId', async (t) => {
  for (const flag of ['--reject-http-mcp', '--reject-mcp']) {
    const records = {}
    const first = makeAdapter({ records, acpArgs: [flag] })
    const second = makeAdapter({ records, acpArgs: [flag] })
    t.after(async () => {
      await first.dispose()
      await second.dispose()
    })

    const turn1 = await collect(first.stream(options('s1', [user('m1', 'first')])))
    assert.equal(turn1.finish.kind, 'stop')
    const cliId = records.s1?.cliSessionId
    assert.ok(cliId)
    await first.dispose()

    const turn2 = await collect(second.stream(options('s1', [user('m1', 'first'), user('m2', 'second')])))
    assert.equal(turn2.finish.kind, 'stop')
    assert.equal(records.s1?.cliSessionId, cliId)
  }
})

test('clearing reasoningEffort restores ACP thought_level after session/load and Host restart', async (t) => {
  const dir = mkdtempSync(join(tmpdir(), 'dsb-effort-'))
  t.after(() => rmSync(dir, { recursive: true, force: true }))

  for (const [tag, extra] of [['bare', []], ['with-options', ['--load-thought-level']]]) {
    const records = {}
    const stateFile = join(dir, `${tag}.json`)
    const args = ['--thought-levels', `--state-file=${stateFile}`, ...extra]
    const first = makeAdapter({ records, acpArgs: args })
    const second = makeAdapter({ records, acpArgs: args })
    t.after(async () => {
      await first.dispose()
      await second.dispose()
    })

    const t1 = await collect(first.stream(options('s1', [user('m1', 'show-state')], { reasoningEffort: 'high' })))
    assert.match(t1.text, /effort: high/)
    assert.equal(records.s1?.defaultEffort, 'medium')
    assert.equal(records.s1?.effort, 'high')
    await first.dispose()

    const t2 = await collect(second.stream(options('s1', [user('m1', 'show-state'), user('m2', 'show-state')])))
    assert.match(t2.text, /effort: medium/)
    assert.equal(records.s1?.defaultEffort, 'medium')
    assert.equal(records.s1?.effort, undefined)

    const t3 = await collect(second.stream(options('s1', [user('m1', 'show-state'), user('m2', 'show-state'), user('m3', 'show-state')], { reasoningEffort: 'low' })))
    assert.match(t3.text, /effort: low/)
    await second.closeSession('s1')

    const t4 = await collect(second.stream(options('s1', [user('m1', 'show-state'), user('m2', 'show-state'), user('m3', 'show-state'), user('m4', 'show-state')])))
    assert.match(t4.text, /effort: medium/)
    assert.equal(records.s1?.effort, undefined)
  }
})

test('legacy ACP agents switch models via session/set_model and fail on unknown models', async (t) => {
  const records = {}
  const adapter = makeAdapter({ records, acpArgs: ['--legacy-models'] })
  t.after(() => adapter.dispose())

  const probed = await adapter.probeModels('local-acp-fake')
  assert.ok(probed.some(model => model.id === 'legacy-fast'))

  const turn1 = await collect(adapter.stream(options('s1', [user('m1', 'show-state')], { model: 'legacy-fast' })))
  assert.match(turn1.text, /model: legacy-fast/)

  await adapter.closeSession('s1')
  const turn2 = await collect(adapter.stream(options('s1', [user('m1', 'show-state'), user('m2', 'show-state')], { model: 'legacy-fast' })))
  assert.match(turn2.text, /model: legacy-fast/)

  await assert.rejects(
    collect(adapter.stream(options('s2', [user('m1', 'show-state')], { model: 'legacy-missing' }))),
    err => err?.code === 'LOCAL_AGENT_MODEL',
  )
})

test('switching ACP models resets effort tracking cleanly and ignores unapplied efforts', async (t) => {
  const dir = mkdtempSync(join(tmpdir(), 'dsb-model-effort-'))
  t.after(() => rmSync(dir, { recursive: true, force: true }))
  const records = {}
  const stateFile = join(dir, 'state.json')
  const adapter = makeAdapter({ records, acpArgs: ['--thought-levels', '--load-thought-level', '--model-resets-thought', `--state-file=${stateFile}`] })
  t.after(() => adapter.dispose())

  const t1 = await collect(adapter.stream(options('s1', [user('m1', 'show-state')], { model: 'fake-large', reasoningEffort: 'high' })))
  assert.match(t1.text, /effort: high/)
  assert.equal(records.s1?.defaultEffort, 'medium')
  assert.equal(records.s1?.effort, 'high')

  const t2 = await collect(adapter.stream(options('s1', [user('m1', 'show-state'), user('m2', 'show-state')], { model: 'fake-small', reasoningEffort: 'high' })))
  assert.match(t2.text, /effort: high/)
  assert.equal(records.s1?.defaultEffort, 'low')
  assert.equal(records.s1?.effort, 'high')

  const t3 = await collect(adapter.stream(options('s1', [user('m1', 'show-state'), user('m2', 'show-state'), user('m3', 'show-state')], { model: 'fake-small' })))
  assert.match(t3.text, /effort: low/)
  assert.equal(records.s1?.defaultEffort, 'low')
  assert.equal(records.s1?.effort, undefined)

  const t4 = await collect(adapter.stream(options('s1', [user('m1', 'show-state'), user('m2', 'show-state'), user('m3', 'show-state'), user('m4', 'show-state')], { model: 'fake-small', reasoningEffort: 'bogus' })))
  assert.match(t4.text, /effort: low/)
  assert.equal(records.s1?.effort, undefined)
})

test('ACP compaction clears persisted effort so Host restart reapplies reasoningEffort cleanly', async (t) => {
  const dir = mkdtempSync(join(tmpdir(), 'dsb-compact-effort-'))
  t.after(() => rmSync(dir, { recursive: true, force: true }))
  const records = {}
  const stateFile = join(dir, 'state.json')
  const args = ['--thought-levels', `--state-file=${stateFile}`]
  const first = makeAdapter({ records, acpArgs: args })
  const second = makeAdapter({ records, acpArgs: args })
  t.after(async () => {
    await first.dispose()
    await second.dispose()
  })

  const t1 = await collect(first.stream(options('s1', [user('m1', 'show-state')], { reasoningEffort: 'high' })))
  assert.match(t1.text, /effort: high/)
  assert.equal(records.s1?.effort, 'high')

  const compacted = await first.compact('s1')
  assert.equal(compacted, 'done')
  assert.equal(records.s1?.effort, undefined)
  await first.dispose()

  const t2 = await collect(second.stream(options('s1', [user('m1', 'show-state'), user('m2', 'show-state')], { reasoningEffort: 'high' })))
  assert.match(t2.text, /effort: high/)
  assert.equal(records.s1?.effort, 'high')
})

test('resuming an ACP session keeps its saved defaultEffort over another workspace default for the same model', async (t) => {
  const dir = mkdtempSync(join(tmpdir(), 'dsb-ws-effort-'))
  t.after(() => rmSync(dir, { recursive: true, force: true }))
  const lowWs = join(dir, 'low-ws')
  const normalWs = join(dir, 'normal-ws')
  mkdirSync(lowWs, { recursive: true })
  mkdirSync(normalWs, { recursive: true })
  const records = {}
  const stateFile = join(dir, 'state.json')
  const args = ['--thought-levels', '--load-thought-level', '--preserve-thought-on-model', `--state-file=${stateFile}`]
  const cwdFor = async id => (id === 'sA' ? lowWs : normalWs)
  const first = makeAdapter({ records, acpArgs: args, cwdFor })
  const second = makeAdapter({ records, acpArgs: args, cwdFor })
  t.after(async () => {
    await first.dispose()
    await second.dispose()
  })

  const a1 = await collect(first.stream(options('sA', [user('m1', 'show-state')], { model: 'fake-large', reasoningEffort: 'high' })))
  assert.match(a1.text, /effort: high/)
  assert.equal(records.sA?.defaultEffort, 'low')
  assert.equal(records.sA?.effort, 'high')
  await first.dispose()

  const b1 = await collect(second.stream(options('sB', [user('m1', 'show-state')], { model: 'fake-large' })))
  assert.match(b1.text, /effort: medium/)
  assert.equal(records.sB?.defaultEffort, 'medium')

  const a2 = await collect(second.stream(options('sA', [user('m1', 'show-state'), user('m2', 'show-state')], { model: 'fake-large' })))
  assert.match(a2.text, /effort: low/)
  assert.equal(records.sA?.defaultEffort, 'low')
  assert.equal(records.sA?.effort, undefined)
})

function makeMemoryAttachments(entries = {}) {
  return {
    async readImage(ref) {
      const entry = entries[ref.attachmentId]
      if (entry === undefined) throw new Error(`missing image ${ref.attachmentId}`)
      if (entry instanceof Error) throw entry
      return { ref: { ...ref, mediaType: entry.mediaType ?? ref.mediaType }, data: entry.data }
    },
    imageHostPath(ref) {
      return entries[ref.attachmentId]?.hostPath
    },
  }
}

const imageMsg = (id, text, ref, { offloaded = false } = {}) => ({
  id,
  role: 'user',
  content: [
    ...(text === undefined ? [] : [{ type: 'text', text }]),
    { type: 'image', attachment: ref, ...(offloaded ? { offloaded: true } : {}) },
  ],
  source: { kind: 'user' },
})

test('codex receives image blocks on first replay turn, delta turn, and image-only turns', async (t) => {
  const pngBytes = Buffer.from([137, 80, 78, 71, 1, 2, 3, 4])
  const jpgBytes = Buffer.from([255, 216, 255, 5, 6, 7])
  const pngRef = { attachmentId: 'att-png', mediaType: 'image/png', bytes: pngBytes.length, width: 1, height: 1, name: 'photo.png' }
  const jpgRef = { attachmentId: 'att-jpg', mediaType: 'image/jpeg', bytes: jpgBytes.length, width: 2, height: 2 }
  const adapter = makeAdapter({
    attachments: makeMemoryAttachments({
      'att-png': { mediaType: 'image/png', data: pngBytes },
      'att-jpg': { mediaType: 'image/jpeg', data: jpgBytes },
    }),
  })
  t.after(() => adapter.dispose())

  const first = await collect(adapter.stream(options('s1', [imageMsg('m1', 'describe this', pngRef)], { provider: 'local-codex-cli', model: 'gpt-5.1-codex' })))
  assert.equal(first.finish?.kind, 'stop')
  assert.match(first.text, new RegExp(`data:image/png;base64,${pngBytes.toString('base64')}`))

  const second = await collect(adapter.stream(options('s1', [imageMsg('m1', 'describe this', pngRef), imageMsg('m2', undefined, jpgRef)], { provider: 'local-codex-cli', model: 'gpt-5.1-codex' })))
  assert.equal(second.finish?.kind, 'stop')
  assert.match(second.text, new RegExp(`data:image/jpeg;base64,${jpgBytes.toString('base64')}`))
})

test('claude receives Anthropic base64 image blocks for text+image and image-only turns', async (t) => {
  const pngBytes = Buffer.from([137, 80, 78, 71, 9, 8, 7, 6])
  const pngRef = { attachmentId: 'att-c1', mediaType: 'image/png', bytes: pngBytes.length, width: 1, height: 1 }
  const adapter = makeAdapter({
    attachments: makeMemoryAttachments({ 'att-c1': { mediaType: 'image/png', data: pngBytes } }),
  })
  t.after(() => adapter.dispose())

  const first = await collect(adapter.stream(options('s1', [imageMsg('m1', 'check screenshot', pngRef)], { provider: 'local-claude-code', model: 'sonnet' })))
  assert.equal(first.finish?.kind, 'stop')
  assert.match(first.text, new RegExp(`"mediaType":"image/png".*"data":"${pngBytes.toString('base64')}"`))

  const second = await collect(adapter.stream(options('s1', [imageMsg('m1', 'check screenshot', pngRef), imageMsg('m2', undefined, pngRef)], { provider: 'local-claude-code', model: 'sonnet' })))
  assert.equal(second.finish?.kind, 'stop')
  assert.match(second.text, new RegExp(`"data":"${pngBytes.toString('base64')}"`))
})

test('acp receives {type:"image", mimeType, data} when promptCapabilities.image is true', async (t) => {
  const webpBytes = Buffer.from([82, 73, 70, 70, 1, 2, 3])
  const ref = { attachmentId: 'att-a1', mediaType: 'image/webp', bytes: webpBytes.length, width: 1, height: 1 }
  const adapter = makeAdapter({
    attachments: makeMemoryAttachments({ 'att-a1': { mediaType: 'image/webp', data: webpBytes } }),
  })
  t.after(() => adapter.dispose())

  const first = await collect(adapter.stream(options('s1', [imageMsg('m1', 'look', ref)])))
  assert.equal(first.finish?.kind, 'stop')
  assert.match(first.text, new RegExp(`"mimeType":"image/webp","data":"${webpBytes.toString('base64')}"`))

  const second = await collect(adapter.stream(options('s1', [imageMsg('m1', 'look', ref), imageMsg('m2', undefined, ref)])))
  assert.equal(second.finish?.kind, 'stop')
  assert.match(second.text, new RegExp(`"mimeType":"image/webp","data":"${webpBytes.toString('base64')}"`))
})

test('acp without image capability degrades image to text and updates resolveModel inputModalities', async (t) => {
  const pngBytes = Buffer.from([137, 80, 78, 71, 4, 5, 6])
  const ref = { attachmentId: 'att-noimg', mediaType: 'image/png', bytes: pngBytes.length, width: 10, height: 20, name: 'shot.png' }
  const adapter = makeAdapter({
    acpArgs: ['--no-image'],
    attachments: makeMemoryAttachments({
      'att-noimg': { mediaType: 'image/png', data: pngBytes, hostPath: '/tmp/host-shot.png' },
    }),
    fs: { processPathFromHostPath: p => p.replace('/tmp/', '/guest/') },
  })
  t.after(() => adapter.dispose())

  // Before any CLI starts, resolveModel must stay fast/lazy and optimistic.
  const before = await adapter.resolveModel('local-acp-fake', 'default')
  assert.deepEqual(before.inputModalities, ['text', 'image'])
  assert.equal(adapter._spawned.length, 0)

  // First turn sees promptCapabilities.image === false and must degrade instead of sending an invalid image block (-32602).
  const first = await collect(adapter.stream(options('s1', [imageMsg('m1', 'what is this?', ref)])))
  assert.equal(first.finish?.kind, 'stop')
  assert.doesNotMatch(first.text, /"mimeType"/)
  assert.match(first.text, /what is this\?/)

  // Once known, resolveModel excludes 'image' so host OCR takes over on subsequent turns.
  const after = await adapter.resolveModel('local-acp-fake', 'default')
  assert.deepEqual(after.inputModalities, ['text'])
})

test('offloaded or unreadable images degrade cleanly without calling readImage on offloaded blocks', async (t) => {
  let reads = 0
  const ref1 = { attachmentId: 'att-off', mediaType: 'image/png', bytes: 12, width: 2, height: 2, name: 'old.png' }
  const ref2 = { attachmentId: 'att-err', mediaType: 'image/png', bytes: 12, width: 2, height: 2, name: 'broken.png' }
  const adapter = makeAdapter({
    attachments: {
      async readImage(ref) {
        reads += 1
        throw new Error(`read failed for ${ref.attachmentId}`)
      },
    },
  })
  t.after(() => adapter.dispose())

  const out = await collect(adapter.stream(options('s1', [
    imageMsg('m1', 'check both', ref1, { offloaded: true }),
    imageMsg('m2', 'and this', ref2),
  ], { provider: 'local-codex-cli', model: 'gpt-5.1-codex' })))
  assert.equal(out.finish?.kind, 'stop')
  assert.equal(reads, 1)
  assert.doesNotMatch(out.text, /images=/)
})

test('replay drops images outside the replay window without reading their attachments', async (t) => {
  const readIds = []
  const pngBytes = Buffer.from([137, 80, 78, 71, 1, 2, 3])
  const oldRef = { attachmentId: 'att-too-old', mediaType: 'image/png', bytes: pngBytes.length, width: 1, height: 1 }
  const recentRef = { attachmentId: 'att-recent', mediaType: 'image/png', bytes: pngBytes.length, width: 1, height: 1 }
  const adapter = makeAdapter({
    attachments: {
      async readImage(ref) {
        readIds.push(ref.attachmentId)
        return { ref, data: pngBytes }
      },
    },
  })
  t.after(() => adapter.dispose())

  const msgs = [
    imageMsg('m0', 'ancient image', oldRef),
    ...Array.from({ length: 13 }, (_, i) => user(`m${i + 1}`, `turn ${i + 1}`)),
    imageMsg('m14', 'latest image', recentRef),
  ]
  const out = await collect(adapter.stream(options('s1', msgs, { provider: 'local-codex-cli', model: 'gpt-5.1-codex' })))
  assert.equal(out.finish?.kind, 'stop')
  assert.deepEqual(readIds, ['att-recent'])
})

test('acp background wakeup drives a real DSH turn, supports bridged MCP calls, and leaves no leaked waiters', async (t) => {
  let wakeResolve
  const woke = new Promise(resolve => { wakeResolve = resolve })
  const settled = []
  const wakes = []
  const adapter = makeAdapter({
    bgIdleSettleMs: 60,
    bgActiveSettleMs: 300,
    onBackgroundWake: async (sessionId) => {
      wakes.push(sessionId)
      wakeResolve(sessionId)
      return true
    },
    onBackgroundSettled: async (sessionId) => {
      settled.push(sessionId)
    },
  })
  t.after(() => adapter.dispose())

  // 1. Initial user turn starts background work and finishes immediately.
  const first = await collect(adapter.stream(options('s1', [user('m1', 'please run bg-wakeup-mcp')])))
  assert.equal(first.text, 'started in background')
  assert.equal(first.finish.kind, 'stop')

  // 2. Spontaneous ACP updates wake the session once; DSH opens a background turn
  //    (with runtime-context + [Background update]) without sending a new prompt.
  assert.equal(await woke, 's1')
  assert.deepEqual(wakes, ['s1'])
  const rtCtx = { id: 'rc1', role: 'user', content: [{ type: 'text', text: 'Time sampled: 2026-11-16T13:30:00+08:00' }], source: { kind: 'runtime-context' } }
  const bgWake = { id: 'bg1', role: 'user', content: [{ type: 'text', text: '[Background update]' }], source: { kind: 'bot', form: 'relay', role: 'background' } }
  const step1 = await collect(adapter.stream(options('s1', [user('m1', 'please run bg-wakeup-mcp'), rtCtx, bgWake])))
  assert.equal(step1.finish.kind, 'tool-calls')
  assert.equal(step1.calls[0]?.name, 'message_bot')
  assert.match(step1.thinking, /checking background output/)
  assert.match(step1.thinking, /▸ Run `cat out\.log`/)

  // 3. Tool result resumes the background turn and settles cleanly into the chat stream.
  const step2 = await collect(adapter.stream(options('s1', [
    user('m1', 'please run bg-wakeup-mcp'),
    rtCtx,
    bgWake,
    toolResult('tr1', step1.calls[0].id, 'Chief: ack'),
  ])))
  assert.equal(step2.finish.kind, 'stop')
  assert.equal(step2.text, 'background finished: Chief: ack')
  assert.deepEqual(settled, ['s1'])

  // 4. Subsequent normal user turn receives its full first chunk (no leaked waiter from bg-settle).
  const next = await collect(adapter.stream(options('s1', [
    user('m1', 'please run bg-wakeup-mcp'),
    rtCtx,
    bgWake,
    toolResult('tr1', step1.calls[0].id, 'Chief: ack'),
    user('m2', 'follow up question'),
  ])))
  assert.equal(next.finish.kind, 'stop')
  assert.match(next.text, /^ok: follow up question/)
})

test('local-agents install wires background wakeups as valid DSH user/messages and hides synthetic headers', async () => {
  assert.equal(stripHeader('[Background update]'), '')
  assert.equal(stripHeader('[Background update]\nextra'), 'extra')

  let registeredAdapter
  const ctx = fakeContext()
  ctx.llm = {
    registerAdapter: (_routes, adapter) => {
      registeredAdapter = adapter
      return Object.assign(() => {}, { replace: () => {} })
    },
  }
  const flushed = []
  ctx.sessions = { flush: async (session) => { flushed.push(session.id) } }
  const nextTurn = []
  const agent = {
    id: 's1',
    session: { id: 's1', header: { cwd: '/tmp' } },
    inbox: {
      nextTurn,
      remove: (id) => {
        const at = nextTurn.findIndex(item => item.id === id)
        if (at !== -1) nextTurn.splice(at, 1)
      },
    },
    followup: (msg) => { nextTurn.push(msg) },
  }
  const rt = {
    ctx,
    state: { bots: { b1: { id: 'b1', model: 'local-antigravity/default' } }, localAgents: { custom: [], sessions: {} } },
    async save() {},
    async load() {},
    botOf: id => rt.state.bots[id],
    selfOf: () => 'b1',
    roomOf: () => undefined,
    groupOwners: new Map(),
    teamTools: [],
    resolveAgent: async () => agent,
  }
  localAgentsInstall(rt)
  assert.ok(registeredAdapter !== undefined)

  const woke = await registeredAdapter.deps.onBackgroundWake('s1')
  assert.equal(woke, true)
  assert.equal(nextTurn.length, 1)
  assert.equal(typeof nextTurn[0].id, 'string')
  assert.ok(nextTurn[0].id.length > 0)
  assert.deepEqual(nextTurn[0].source, { kind: 'bot', form: 'relay', role: 'background' })
  assert.equal(flushed.length, 1)

  // Duplicate wake while already queued does not push a second trigger.
  await registeredAdapter.deps.onBackgroundWake('s1')
  assert.equal(nextTurn.length, 1)

  // Settling cleans up any unconsumed queued trigger.
  await registeredAdapter.deps.onBackgroundSettled('s1')
  assert.equal(nextTurn.length, 0)
})

test('agy-steer: Unix socket + marker gate no-ops in pure /bin/sh without a steer and injects PreInvocation / PostInvocation / Stop steers when pending', async (t) => {
  assert.equal(canSteerEngine(ENGINES['local-antigravity']), true)
  assert.equal(canSteerEngine(ENGINES['local-codex-cli']), true)
  assert.equal(canSteerEngine(ENGINES['local-claude-code']), true)
  assert.equal(canSteerEngine(ENGINES['local-cursor-cli']), false)

  const homeDir = mkdtempSync(join(tmpdir(), 'dsb-agy-plugin-'))
  const dir = ensureAgySteerPlugin({ homeDir })
  assert.ok(dir !== undefined)
  const hooksCfg = JSON.parse(readFileSync(join(dir, 'hooks.json'), 'utf8'))
  const stepCmd = hooksCfg['dsbot-steer'].PostInvocation[0].command
  const stopCmd = hooksCfg['dsbot-steer'].Stop[0].command

  const runShellHook = (command, stdinObj, envOver = {}) => new Promise((resolve, reject) => {
    const child = process.platform === 'win32'
      ? spawn(process.execPath, [join(dir, 'steer-hook.mjs')], { cwd: dir, env: { ...process.env, ...envOver }, stdio: ['pipe', 'pipe', 'pipe'] })
      : spawn('/bin/sh', ['-c', command], { cwd: dir, env: { ...process.env, ...envOver }, stdio: ['pipe', 'pipe', 'pipe'] })
    let out = ''
    child.stdout.setEncoding('utf8')
    child.stdout.on('data', chunk => { out += chunk })
    child.on('error', reject)
    child.on('exit', code => (code === 0 ? resolve(JSON.parse(out)) : reject(new Error(`exit ${code}`))))
    child.stdin.end(JSON.stringify(stdinObj))
  })

  // 1. Standalone terminal agy (no DSBOT_STEER_DIR): immediate {} in pure /bin/sh.
  const standalone = await runShellHook(stepCmd, { conversationId: 'c1', hookEventName: 'PostInvocation' }, { DSBOT_STEER_DIR: '', DSBOT_STEER_SOCK: '', DSBOT_STEER_TOKEN: '' })
  assert.deepEqual(standalone, {})

  // 2. Active AgySteerServer over private Unix socket (on POSIX) + marker gate.
  let hits = 0
  const queues = new Map([['c1', ['first steer', 'second steer']], ['c2', ['late steer']]])
  const server = new AgySteerServer({
    drain: (id) => { hits += 1; return (queues.get(id) ?? []).splice(0) },
    hasPending: id => (queues.get(id)?.length ?? 0) > 0,
  })
  const steerEnv = await server.listen()
  t.after(() => server.close())
  if (process.platform !== 'win32') {
    assert.ok(typeof server.socketPath === 'string' && existsSync(server.socketPath), 'listens on a Unix domain socket instead of a random TCP port')
    assert.equal(server.url, undefined)
  }

  // Inside DS Bot WITH ANTIGRAVITY_CONVERSATION_ID='c1' but BEFORE setPending('c1', true):
  // pure /bin/sh marker gate exits immediately with {} without touching the socket!
  if (process.platform !== 'win32') {
    const gated = await runShellHook(stepCmd, { conversationId: 'c1', hookEventName: 'PostInvocation' }, { ...steerEnv, ANTIGRAVITY_CONVERSATION_ID: 'c1' })
    assert.deepEqual(gated, {})
    assert.equal(hits, 0, 'marker gate skips spawning node when no .steer file exists')
  }

  // Mark c1 pending -> marker file exists -> shell invokes steer-hook.mjs -> drains & unlinks marker!
  server.setPending('c1', true)
  assert.equal(existsSync(join(server.dir, 'c1.steer')), true)

  // Wrong token fails open with {} so agy is never blocked.
  const badToken = await runShellHook(stepCmd, { conversationId: 'c1', hookEventName: 'PostInvocation' }, { ...steerEnv, DSBOT_STEER_TOKEN: 'wrong', ANTIGRAVITY_CONVERSATION_ID: 'c1' })
  assert.deepEqual(badToken, {})
  assert.equal(queues.get('c1').length, 2, 'unauthorized call does not drain')

  const postOut = await runShellHook(stepCmd, { conversationId: 'c1', hookEventName: 'PostInvocation' }, { ...steerEnv, ANTIGRAVITY_CONVERSATION_ID: 'c1' })
  assert.deepEqual(postOut, {
    injectSteps: [{ userMessage: 'first steer' }, { userMessage: 'second steer' }],
    terminationBehavior: 'force_continue',
  })
  assert.equal(existsSync(join(server.dir, 'c1.steer')), false, 'marker file is removed automatically once drained')

  // Stop with terminal USER_CANCELED never continues; Stop with NO_TOOL_CALL peeks without popping, sets marker, then PreInvocation drains.
  assert.deepEqual(await runShellHook(stopCmd, { conversationId: 'c2', hookEventName: 'Stop', terminationReason: 'USER_CANCELED' }, { ...steerEnv, ANTIGRAVITY_CONVERSATION_ID: 'c2' }), {})
  const stopOut = await runShellHook(stopCmd, { conversationId: 'c2', hookEventName: 'Stop', terminationReason: 'NO_TOOL_CALL' }, { ...steerEnv, ANTIGRAVITY_CONVERSATION_ID: 'c2' })
  assert.equal(stopOut.decision, 'continue')
  assert.equal(queues.get('c2').length, 1, 'Stop peeks without popping so PreInvocation injects userMessage')
  assert.equal(existsSync(join(server.dir, 'c2.steer')), true, 'Stop arms c2.steer so the subsequent PreInvocation passes the shell gate')
  const preOut = await runShellHook(stepCmd, { conversationId: 'c2', hookEventName: 'PreInvocation' }, { ...steerEnv, ANTIGRAVITY_CONVERSATION_ID: 'c2' })
  assert.deepEqual(preOut, { injectSteps: [{ userMessage: 'late steer' }] })
})

test('acp (antigravity): unconsumed steerQueue items at turn end fall back to next prompt without loss', async (t) => {
  const adapter = makeAdapter({ acpArgs: ['--agy'] })
  t.after(() => adapter.dispose())
  await collect(adapter.stream(options('s1', [user('m1', 'hello')], { provider: 'local-antigravity' })))
  const live = adapter.live.get('s1')
  assert.ok(live !== undefined)
  live.steerQueue.push('arrived right before done')
  const second = await collect(adapter.stream(options('s1', [user('m1', 'hello'), user('m2', 'next')], { provider: 'local-antigravity' })))
  assert.match(second.text, /ok: next/)
  assert.match(second.text, /ok: arrived right before done/)
})

test('group CLI turns touch the room waiter, switch to working on internal tools, and flush step bubbles without duplication at turn/end', async (t) => {
  const ctx = fakeContext()
  const stepPosts = []
  const latePosts = []
  let touches = 0
  let works = 0
  const rt = {
    ctx,
    state: { bots: { b1: { id: 'b1', name: 'DSHDev', model: 'local-antigravity/default' } }, rooms: { r1: { id: 'r1', members: ['b1'] } }, exchangeTurns: {}, questions: {}, views: {} },
    async save() {},
    async load() {},
    warn() {},
    noteFailure() {},
    isBotSession: id => id === 'gs1',
    roomOf: id => rt.state.rooms[id],
    botOf: id => rt.state.bots[id],
    selfOf: id => (id === 'gs1' ? 'b1' : undefined),
    chatOf: () => 'dm1',
    chatOwners: new Map(),
    groupOwners: new Map([['gs1', { roomId: 'r1', botId: 'b1' }]]),
    switchJobs: () => [],
    sessionFolder: () => '/tmp',
    touchRoomTurn: () => { touches += 1 },
    workRoomTurn: () => { works += 1 },
    roomStepText: async (_id, _info, text, first) => { stepPosts.push({ text, first }) },
    lateRoomReply: async (_id, _info, reply, meta) => { latePosts.push({ reply, ...meta }) },
  }
  turnsInstall(rt)

  const adapter = makeAdapter({
    acpArgs: ['--agy'],
    onStreamTouch: id => rt.touchRoomTurn(id),
    onCliWork: (id, texts) => rt.noteCliWork(id, texts),
  })
  t.after(() => adapter.dispose())

  const emit = async (type, data) => {
    for (const fn of ctx.listeners.get('session/event') ?? []) {
      if (fn.length >= 2) await fn({ id: 'gs1' }, { type, data })
    }
  }

  await emit('turn/start', { turn: 1 })
  await emit('user/message', {
    content: [{ type: 'text', text: '[Group chat]\ndo work' }],
    source: { kind: 'bot', role: 'room', roomId: 'r1', exchangeId: 'ex-1' },
  })

  const blocks = []
  for await (const chunk of adapter.stream(options('gs1', [user('m1', 'run group-progress')], { provider: 'local-antigravity' }))) {
    if (chunk.type === 'block-end') blocks.push(chunk.block)
  }
  assert.equal(touches >= 1, true, 'stream heartbeat touches the room waiter')
  assert.equal(works, 2, 'each internal CLI tool marks the room turn as working')
  assert.deepEqual(stepPosts, [
    { text: 'Opening note before tools.', first: true },
    { text: 'Milestone halfway done.', first: false },
  ])

  await emit('assistant/message', { message: { role: 'assistant', content: blocks } })
  await emit('turn/end', { turn: 1, reason: { kind: 'completed' } })
  assert.equal(latePosts.length, 1)
  assert.equal(latePosts[0].reply, 'Final result delivered.')
  assert.equal(latePosts[0].shown, true)
  assert.equal(latePosts[0].toolCount, 2)
})

test('acp session/load drops historical replay notifications arriving both before and after the load RPC reply', async (t) => {
  const records = { s1: { route: 'local-antigravity', cliSessionId: 'acp-resumed-1', handedId: 'm1', model: 'default' } }
  const adapter = makeAdapter({ records, acpArgs: ['--agy', '--replay-on-load'] })
  t.after(() => adapter.dispose())

  const out = await collect(adapter.stream(options('s1', [user('m1', 'old'), user('m2', 'fresh command')], { provider: 'local-antigravity' })))
  assert.equal(out.finish.kind, 'stop')
  assert.match(out.text, /^ok: fresh command/)
  assert.doesNotMatch(out.text, /STALE_/)
  assert.doesNotMatch(out.thinking, /STALE_/)
})

test('Stop hook fullyIdle:false protects long in-process agy subagents from 10-minute idle session GC', async (t) => {
  const adapter = makeAdapter({ acpArgs: ['--agy'] })
  t.after(() => adapter.dispose())

  await collect(adapter.stream(options('s1', [user('m1', 'start subagent')], { provider: 'local-antigravity' })))
  const live = adapter.live.get('s1')
  const engine = adapter.engines.get('local-antigravity')
  assert.ok(live !== undefined && engine?.steerServer !== undefined)

  const reportStop = (fullyIdle) => postAgySteer(engine.steerServer, {
    conversationId: live.cliSessionId,
    hookEventName: 'Stop',
    terminationReason: 'NO_TOOL_CALL',
    fullyIdle,
  })

  await reportStop(false)
  assert.equal(live.cliFullyIdle, false)
  await adapter.closeLive(live, { idle: true })
  assert.equal(live.dead, false, 'live session stays registered while in-process subagent runs')
  assert.equal(adapter.live.has('s1'), true)

  await reportStop(true)
  assert.equal(live.cliFullyIdle, true)
  await adapter.closeLive(live, { idle: true })
  assert.equal(live.dead, true)
  assert.equal(adapter.live.has('s1'), false)
})

test('resolveModel auto-probes named models on a cold cache and reuses persisted models across restarts without spawning', async (t) => {
  const savedModels = {}
  const makePersistedAdapter = () => {
    const spawned = []
    const adapter = new LocalAgentAdapter({
      rt: { warn: () => {} },
      customAgents: () => [],
      findExecutable: path => path,
      spawnEngine: (_kind, route, spec, hooks = {}) => {
        const engine = new AcpEngine(route, { ...spec, command: process.execPath, args: [FIXTURES.acp, '--agy', '--thought-levels'] }, { findExecutable: path => path, agyHomeDir: false, ...hooks })
        spawned.push(engine)
        return engine
      },
      cachedModels: route => savedModels[route],
      saveModels: (route, models) => { savedModels[route] = models },
      dropModels: (route) => { if (route === undefined) for (const k of Object.keys(savedModels)) delete savedModels[k]; else delete savedModels[route] },
      record: () => undefined,
      saveRecord: () => {},
      presetFor: () => 'workspace-write',
      cwdFor: async () => mkdtempSync(join(tmpdir(), 'dsb-cwd-')),
      probeCwd: () => mkdtempSync(join(tmpdir(), 'dsb-probe-')),
      toolsFor: () => [],
    })
    adapter._spawned = spawned
    return adapter
  }

  // 1. Cold boot with empty savedModels: resolveModel for 'fake-large' auto-probes and persists efforts.
  const first = makePersistedAdapter()
  t.after(() => first.dispose())
  const info1 = await first.resolveModel('local-antigravity', 'fake-large')
  assert.deepEqual(info1.reasoning?.efforts?.map(e => e.id), ['low', 'medium', 'high', 'xhigh'])
  assert.ok(Array.isArray(savedModels['local-antigravity']) && savedModels['local-antigravity'].length > 0)

  // 2. Simulated restart: second adapter reads savedModels immediately with 0 spawned processes!
  const second = makePersistedAdapter()
  t.after(() => second.dispose())
  const info2 = await second.resolveModel('local-antigravity', 'fake-large')
  assert.deepEqual(info2.reasoning?.efforts?.map(e => e.id), ['low', 'medium', 'high', 'xhigh'])
  assert.equal(second._spawned.length, 0, 'persisted model cache resolves reasoning efforts across restarts without spawning')
})

test('transcript watermark and hydration handle >64KiB partial records, >2MiB full records, delayed full flushes, consecutive wakes, mixed stdout+transcript, and USER_CANCELED', async (t) => {
  const dir = mkdtempSync(join(tmpdir(), 'dsb-transcript-edges-'))
  t.after(() => rmSync(dir, { recursive: true, force: true }))
  const compactPath = join(dir, 'transcript.jsonl')
  const fullPath = join(dir, 'transcript_full.jsonl')

  // 1. >64 KiB partial JSONL line must never anchor mid-record.
  const step0 = `${JSON.stringify({ step_index: 0, type: 'PLANNER_RESPONSE', status: 'DONE', content: 'initial ok' })}\n`
  const longText = 'A'.repeat(70_000)
  const step1Full = JSON.stringify({ step_index: 1, type: 'PLANNER_RESPONSE', status: 'DONE', content: longText })
  writeFileSync(compactPath, step0 + step1Full.slice(0, 70_000), 'utf8')
  const mark0 = readAgyTranscriptWatermark(compactPath)
  assert.equal(mark0.maxStepIndex, 0)
  assert.equal(mark0.byteOffset, Buffer.byteLength(step0), 'watermark stays at exact boundary before 70KB partial record')

  writeFileSync(compactPath, `${step0}${step1Full}\n`, 'utf8')
  const wake1 = readAgyTranscriptWakeup(compactPath, mark0)
  assert.equal(wake1.events.length, 1)
  assert.equal(wake1.events[0].text, longText)

  // 2. Delayed hydration + >2 MiB full record in transcript_full.jsonl.
  const step2Compact = `${JSON.stringify({ step_index: 2, type: 'PLANNER_RESPONSE', status: 'DONE', content: 'TRUNCATED', truncated_fields: ['content'] })}\n`
  writeFileSync(compactPath, `${step0}${step1Full}\n${step2Compact}`, 'utf8')
  const pendingWake = readAgyTranscriptWakeup(compactPath, wake1.watermark)
  assert.equal(pendingWake.unresolvedTruncated, true, 'retains unresolved truncated step when transcript_full.jsonl is not flushed yet')
  assert.equal(pendingWake.events.length, 0)
  assert.equal(pendingWake.watermark.maxStepIndex, 1)

  const overTwoMiB = 'B'.repeat(2_200_000)
  writeFileSync(fullPath, `${JSON.stringify({ step_index: 2, type: 'PLANNER_RESPONSE', status: 'DONE', content: overTwoMiB })}\n`, 'utf8')
  const hydratedWake = readAgyTranscriptWakeup(compactPath, pendingWake.watermark)
  assert.equal(hydratedWake.unresolvedTruncated, false)
  assert.equal(hydratedWake.events.length, 1)
  assert.equal(hydratedWake.events[0].text.length, 2_200_000)

  // 3. Consecutive background Stop hooks coalesce without stranding batches, mixed stdout+transcript reconciles, and USER_CANCELED is suppressed.
  let wokeCount = 0
  const adapter = makeAdapter({
    acpArgs: ['--agy'],
    onBackgroundWake: async () => { wokeCount += 1; return true },
  })
  t.after(() => adapter.dispose())

  await collect(adapter.stream(options('s-edge', [user('m1', 'hello')], { provider: 'local-antigravity' })))
  const live = adapter.live.get('s-edge')
  const sessionLog = join(dir, 'live-transcript.jsonl')
  const turn1Line = `${JSON.stringify({ step_index: 0, type: 'PLANNER_RESPONSE', status: 'DONE', content: 'ok: hello' })}\n`
  writeFileSync(sessionLog, turn1Line, 'utf8')
  live.transcriptWatermark = readAgyTranscriptWatermark(sessionLog)

  // Mixed stdout activity + transcript-only final text across two consecutive Stop hooks before draining!
  live.bgStdoutActivities = 1
  live.push({ type: 'activity', text: '▸ run_command' })
  const wakeBatch1 = `${JSON.stringify({ step_index: 1, type: 'PLANNER_RESPONSE', status: 'DONE', content: 'first subagent finished' })}\n`
  writeFileSync(sessionLog, turn1Line + wakeBatch1, 'utf8')
  await adapter.noteHookStopState('local-antigravity', live.cliSessionId, false, sessionLog, 'NO_TOOL_CALL')

  const wakeBatch2 = `${JSON.stringify({ step_index: 2, type: 'PLANNER_RESPONSE', status: 'DONE', content: 'second subagent finished' })}\n`
  writeFileSync(sessionLog, turn1Line + wakeBatch1 + wakeBatch2, 'utf8')
  await adapter.noteHookStopState('local-antigravity', live.cliSessionId, true, sessionLog, 'NO_TOOL_CALL')

  const bgMsg = { id: 'bg-1', role: 'user', content: [{ type: 'text', text: '[Background update]' }], source: { kind: 'bot', form: 'relay', role: 'background' } }
  const coalesced = await collect(adapter.stream(options('s-edge', [user('m1', 'hello'), bgMsg], { provider: 'local-antigravity' })))
  assert.equal(coalesced.text, 'first subagent finished\n\nsecond subagent finished')
  assert.equal(live.background, false)
  assert.equal(live.events.length, 0, 'no stranded events left behind for subsequent questions')

  // USER_CANCELED Stop hook advances watermark without queuing a bogus background wake.
  const wakeCountBeforeCancel = wokeCount
  const cancelledLine = `${JSON.stringify({ step_index: 3, type: 'PLANNER_RESPONSE', status: 'DONE', content: 'cancelled partial' })}\n`
  writeFileSync(sessionLog, turn1Line + wakeBatch1 + wakeBatch2 + cancelledLine, 'utf8')
  await adapter.noteHookStopState('local-antigravity', live.cliSessionId, true, sessionLog, 'USER_CANCELED')
  assert.equal(wokeCount, wakeCountBeforeCancel)
  assert.equal(live.background, false)
  assert.equal(live.transcriptWatermark?.maxStepIndex, 3)
})

test('side session reopens cleanly with full snapshot replay when main cwd or permission preset changes', async (t) => {
  let currentCwd = realpathSync(mkdtempSync(join(tmpdir(), 'dsb-side-cwd1-')))
  const nextCwd = realpathSync(mkdtempSync(join(tmpdir(), 'dsb-side-cwd2-')))
  let currentPreset = 'read-only'
  const sideRecords = new Map()
  const adapter = new LocalAgentAdapter({
    rt: { warn: () => {} },
    customAgents: () => [],
    findExecutable: path => path,
    spawnEngine: (_kind, route, spec, hooks = {}) => new AcpEngine(route, { ...spec, command: process.execPath, args: [FIXTURES.acp, '--agy'] }, { findExecutable: path => path, agyHomeDir: false, ...hooks }),
    record: id => sideRecords.get(id),
    saveRecord: (id, rec) => sideRecords.set(id, rec),
    dropRecord: id => sideRecords.delete(id),
    presetFor: () => currentPreset,
    cwdFor: async () => currentCwd,
    probeCwd: () => currentCwd,
    toolsFor: () => [],
  })
  t.after(async () => {
    await adapter.dispose()
    rmSync(currentCwd, { recursive: true, force: true })
    rmSync(nextCwd, { recursive: true, force: true })
  })

  const mainMsg = user('main-1', 'Main context anchor line')
  const q1 = user('side-q-1', 'side-inspect initial')
  const r1 = await collect(adapter.stream({
    provider: 'local-antigravity',
    model: 'default',
    sessionId: 'side:c1',
    sideMainCount: 1,
    messages: [mainMsg, q1],
  }))
  assert.match(r1.text, new RegExp(`cwd=${currentCwd.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`))
  assert.match(r1.text, /readOk=true execOk=false/)

  // 1. Permission preset upgrades to danger-full-access -> reopens side session AND preserves main snapshot!
  currentPreset = 'danger-full-access'
  const a1 = { id: 'side-a-1', role: 'assistant', content: [{ type: 'text', text: 'First side answer' }] }
  const q2 = user('side-q-2', 'side-inspect after-preset-change')
  const r2 = await collect(adapter.stream({
    provider: 'local-antigravity',
    model: 'default',
    sessionId: 'side:c1',
    sideMainCount: 1,
    messages: [mainMsg, q1, a1, q2],
  }))
  assert.match(r2.text, /readOk=true execOk=true/)
  assert.match(r2.text, /Main context anchor line/, 'retains main snapshot after permission preset change')
  assert.match(r2.text, /First side answer/, 'retains earlier side exchange after permission preset change')

  // 2. Main workspace moves to nextCwd -> side session reopens in nextCwd with full snapshot!
  currentCwd = nextCwd
  const a2 = { id: 'side-a-2', role: 'assistant', content: [{ type: 'text', text: 'Second side answer' }] }
  const q3 = user('side-q-3', 'side-inspect after-cwd-change')
  const r3 = await collect(adapter.stream({
    provider: 'local-antigravity',
    model: 'default',
    sessionId: 'side:c1',
    sideMainCount: 1,
    messages: [mainMsg, q1, a1, q2, a2, q3],
  }))
  assert.match(r3.text, new RegExp(`cwd=${nextCwd.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`), 'reopens side session in new main workspace folder')
  assert.match(r3.text, /Main context anchor line/, 'replays main snapshot in new workspace folder')
})

test('antigravity isolates danger-full-access workers with --dangerously-skip-permissions so background cross-dir reads never deadlock', async (t) => {
  const sharedCwd = realpathSync(mkdtempSync(join(tmpdir(), 'dsb-agy-perm-')))
  const txDir = mkdtempSync(join(tmpdir(), 'dsb-agy-bg-tx-'))
  const presets = { sFull: 'danger-full-access', sWrite: 'workspace-write', sRead: 'read-only' }
  const records = {}
  let wokeResolve
  const wokePromise = new Promise(resolve => { wokeResolve = resolve })
  const spawned = []
  const adapter = new LocalAgentAdapter({
    rt: { warn: () => {} },
    customAgents: () => [],
    findExecutable: path => path,
    spawnEngine: (_kind, route, spec, hooks = {}) => {
      const engine = new AcpEngine(route, { ...spec, command: process.execPath, args: [FIXTURES.acp, '--agy'] }, { findExecutable: path => path, agyHomeDir: false, ...hooks })
      spawned.push(engine)
      return engine
    },
    record: id => records[id],
    saveRecord: (id, rec) => { records[id] = rec },
    presetFor: id => presets[id] ?? 'workspace-write',
    cwdFor: async () => sharedCwd,
    probeCwd: () => sharedCwd,
    toolsFor: () => [],
    onBackgroundWake: async (sessionId) => {
      wokeResolve(sessionId)
      return true
    },
  })
  t.after(async () => {
    await adapter.dispose()
    rmSync(sharedCwd, { recursive: true, force: true })
    rmSync(txDir, { recursive: true, force: true })
  })

  // 1. Same cwd, different presets -> isolated worker pools (`skipPerms: true` ONLY for `danger-full-access`)
  const fullState = await collect(adapter.stream(options('sFull', [user('m1', 'show-state')], { provider: 'local-antigravity' })))
  const writeState = await collect(adapter.stream(options('sWrite', [user('m1', 'show-state')], { provider: 'local-antigravity' })))
  const readState = await collect(adapter.stream(options('sRead', [user('m1', 'show-state')], { provider: 'local-antigravity' })))
  assert.match(fullState.text, /skipPerms: true/, 'danger-full-access gets --dangerously-skip-permissions')
  assert.match(writeState.text, /skipPerms: false/, 'workspace-write stays on standard worker without flag')
  assert.match(readState.text, /skipPerms: false/, 'read-only shares standard worker without flag')
  assert.equal(spawned[0].workers.size, 2, 'one full-access worker and one standard worker for the shared folder')

  // 2. Ephemeral `ask()` (read-only steward call) never reuses the `--dangerously-skip-permissions` worker
  await adapter.closeSession('sWrite')
  await adapter.closeSession('sRead')
  assert.equal(spawned[0].workers.size, 1, 'standard worker closed once non-full-access sessions left')
  const steward = await adapter.ask('sFull', { system: 'Steward', prompt: 'show-state' })
  assert.match(steward, /skipPerms: false/, 'ephemeral read-only ask() never reuses a full-access worker')

  // 3. Background wakeup cross-workspace `read_file` completes immediately without waiting for a foreground `session/prompt`
  const turn1 = await collect(adapter.stream(options('sFull', [user('m1', 'show-state'), user('m2', `bg-cross-dir-read:${txDir}`)], { provider: 'local-antigravity' })))
  assert.equal(turn1.text, '[waiting for subagent]')
  const wokeSession = await Promise.race([
    wokePromise,
    new Promise((_, reject) => setTimeout(() => reject(new Error('background cross-dir read deadlocked')), 2_000)),
  ])
  assert.equal(wokeSession, 'sFull')
  const bgTrigger = { id: 'bg-x1', role: 'user', content: [{ type: 'text', text: '[Background update]' }], source: { kind: 'bot', form: 'relay', role: 'background' } }
  const bgOut = await collect(adapter.stream(options('sFull', [user('m1', 'show-state'), user('m2', `bg-cross-dir-read:${txDir}`), bgTrigger], { provider: 'local-antigravity' })))
  assert.equal(bgOut.text, 'Cross-workspace background synthesis delivered!')
})

test('AcpEngine auto-allows danger-full-access permissions across unknown subagent sessions, bare background MCP calls, and fail-closed option sets', async () => {
  // 1. Dedicated fullAccess worker: unknown child sessionId never fails with -32603
  const fullWorker = new AcpEngine('local-antigravity', { ...ENGINES['local-antigravity'], cwdPerProcess: false }, { findExecutable: () => undefined })
  fullWorker.fullAccess = true

  assert.deepEqual(
    await fullWorker.onRequest('session/request_permission', {
      sessionId: 'subagent-child-unknown',
      toolCall: { toolCallId: 'c1', kind: 'read' },
      options: [
        { optionId: 'reject-1', name: 'Reject', kind: 'reject_once' },
        { optionId: 'always-1', name: 'Allow always', kind: 'allow_always' },
        { optionId: 'once-1', name: 'Allow once', kind: 'allow_once' },
      ],
    }),
    { outcome: { outcome: 'selected', optionId: 'once-1' } },
    'prefers allow_once over allow_always and reject options',
  )

  assert.deepEqual(
    await fullWorker.onRequest('session/request_permission', {
      sessionId: 'subagent-child-unknown',
      toolCall: { toolCallId: 'c2', kind: 'read' },
      options: [
        { optionId: 'reject-1', name: 'Reject', kind: 'reject_once' },
        { optionId: 'always-1', name: 'Allow always', kind: 'allow_always' },
      ],
    }),
    { outcome: { outcome: 'selected', optionId: 'always-1' } },
    'falls back to allow_always when allow_once is absent',
  )

  assert.deepEqual(
    await fullWorker.onRequest('session/request_permission', {
      sessionId: 'subagent-child-unknown',
      toolCall: { toolCallId: 'c3', kind: 'other' },
      options: [
        { optionId: 'reject_once', name: 'Reject_once' },
        { optionId: 'denyAll', name: 'Deny all', kind: 'other' },
        { optionId: 'proceed_action', name: 'Proceed' },
      ],
    }),
    { outcome: { outcome: 'selected', optionId: 'proceed_action' } },
    'falls back to first non-reject option while filtering underscore/camelCase reject labels',
  )

  for (const emptyOrRejectOnly of [
    [],
    undefined,
    [null, undefined, { optionId: '   ', name: 'Allow', kind: 'allow_once' }],
    [{ optionId: 'skip_op', name: 'Skip this operation' }],
    [{ optionId: 'reject_once', name: 'Reject', kind: 'reject_once' }, { optionId: 'no_cancel', name: 'No, cancel' }],
  ]) {
    assert.deepEqual(
      await fullWorker.onRequest('session/request_permission', {
        sessionId: 'subagent-child-unknown',
        toolCall: { toolCallId: 'c4', kind: 'execute' },
        options: emptyOrRejectOnly,
      }),
      { outcome: { outcome: 'cancelled' } },
      'fails closed with cancelled when no validated affirmative allow option exists',
    )
  }

  assert.deepEqual(
    await fullWorker.onRequest('session/request_permission', {
      sessionId: 'subagent-child-unknown',
      toolCall: { toolCallId: 'c5', kind: 'read' },
      options: [
        { optionId: 'always-1', name: 'Allow always', kind: 'allow_always' },
        { optionId: 'session-1', name: 'Allow for session', kind: 'allow_session' },
      ],
    }),
    { outcome: { outcome: 'selected', optionId: 'session-1' } },
    'prefers session-scoped allow over allow_always when allow_once is absent',
  )

  // 2. Shared non-fullAccess engine: session.preset === 'danger-full-access' handles bare background MCP calls
  //    without tool_call metadata, while NEVER auto-allowing unattributed unknown sessionIds (even after forget()).
  const sharedEngine = new AcpEngine('local-cursor-cli', ENGINES['local-cursor-cli'], { findExecutable: () => undefined })
  await assert.rejects(
    () => sharedEngine.onRequest('session/request_permission', {
      sessionId: 'orphan-subagent',
      toolCall: { toolCallId: 'x0' },
      options: [{ optionId: 'allow', name: 'Allow', kind: 'allow_once' }],
    }),
    err => err?.code === -32603,
    'empty non-fullAccess engine rejects unknown sessionId',
  )

  const restrictedSess = sharedEngine.sessionFor('s-restricted')
  restrictedSess.acpSessionId = 'acp-main-restricted'
  restrictedSess.preset = 'read-only'

  const fullSess = sharedEngine.sessionFor('s-full')
  fullSess.acpSessionId = 'acp-main-full'
  fullSess.preset = 'danger-full-access'
  const queued = []
  fullSess.push = ev => queued.push(ev)

  assert.deepEqual(
    await sharedEngine.onRequest('session/request_permission', {
      sessionId: 'acp-main-full',
      toolCall: { toolCallId: 'silent-bg-remember' }, // no session.toolCalls entry and no mcp_dsbot_* name
      options: [
        { optionId: 'deny', name: 'Deny', kind: 'reject_once' },
        { optionId: 'allow', name: 'Allow', kind: 'allow_once' },
      ],
    }),
    { outcome: { outcome: 'selected', optionId: 'allow' } },
    'auto-allows bare background MCP call on danger-full-access session without queuing permission card',
  )
  assert.equal(queued.length, 0)

  assert.deepEqual(
    await sharedEngine.onRequest('session/request_permission', {
      sessionId: 'acp-main-full',
      toolCall: { toolCallId: 'silent-bg-no-allow' },
      options: [{ optionId: 'deny', name: 'Deny', kind: 'reject_once' }],
    }),
    { outcome: { outcome: 'cancelled' } },
    'known danger-full-access session also fails closed when no allow option is offered',
  )

  sharedEngine.forget('s-restricted')
  await assert.rejects(
    () => sharedEngine.onRequest('session/request_permission', {
      sessionId: 'acp-late-child-of-forgotten-restricted',
      toolCall: { toolCallId: 'child-read' },
      options: [{ optionId: 'allow', name: 'Allow', kind: 'allow_once' }],
    }),
    err => err?.code === -32603,
    'shared non-fullAccess process never auto-approves unknown sessionIds even after restricted session is forgotten',
  )
})

test('end-to-end: side subagent unknown sessionId, silent background MCP wakeup, note()/listModels isolation, preset downgrade, and wire session/cancel', async (t) => {
  const sharedCwd = realpathSync(mkdtempSync(join(tmpdir(), 'dsb-e2e-perm-')))
  const bgDir = mkdtempSync(join(tmpdir(), 'dsb-e2e-bg-'))
  const cancelLog = join(bgDir, 'cancels.log')
  const presets = { 'side:c-full': 'danger-full-access', sWrite: 'workspace-write', sBgFull: 'danger-full-access', sAgyFull: 'danger-full-access' }
  const records = {}
  const wakes = []
  let wakeNotify
  const waitNextWake = () => new Promise((resolve) => {
    if (wakes.length > 0) resolve(wakes.shift())
    else wakeNotify = resolve
  })
  const spawnedWorkerArgs = []
  const adapter = new LocalAgentAdapter({
    rt: { warn: () => {} },
    customAgents: () => [{ id: 'hook-nofullargs', name: 'Hook ACP', command: 'x', args: [] }],
    findExecutable: path => path,
    spawnEngine: (_kind, route, spec, hooks = {}) => {
      const effectiveSpec = route === 'local-acp-hook-nofullargs' ? { ...spec, hookSteer: true } : spec
      const engine = new AcpEngine(route, { ...effectiveSpec, command: process.execPath, args: [FIXTURES.acp, '--agy', `--cancel-log=${cancelLog}`] }, { findExecutable: path => path, agyHomeDir: false, ...hooks })
      const origSpawn = engine.spawnWorker.bind(engine)
      engine.spawnWorker = (cwd, opts) => {
        const w = origSpawn(cwd, opts)
        spawnedWorkerArgs.push({ fullAccess: Boolean(w.fullAccess), args: [...(w.engine.args ?? [])] })
        return w
      }
      return engine
    },
    record: id => records[id],
    saveRecord: (id, rec) => { records[id] = rec },
    presetFor: id => presets[id] ?? 'workspace-write',
    cwdFor: async () => sharedCwd,
    probeCwd: () => sharedCwd,
    toolsFor: () => [{ name: 'message_bot', description: 'Message bot', parameters: { type: 'object', properties: { to: { type: 'string' }, message: { type: 'string' } } } }],
    onBackgroundWake: async (sessionId) => {
      if (wakeNotify !== undefined) {
        const fn = wakeNotify
        wakeNotify = undefined
        fn(sessionId)
      } else {
        wakes.push(sessionId)
      }
      return true
    },
  })
  t.after(async () => {
    await adapter.dispose()
    rmSync(sharedCwd, { recursive: true, force: true })
    rmSync(bgDir, { recursive: true, force: true })
  })

  // 1. Side Chat subagent with unknown sessionId under danger-full-access succeeds; workspace-write rejects -32603
  const sideOut = await collect(adapter.stream(options('side:c-full', [user('q1', 'subagent-perm-probe')], { provider: 'local-antigravity' })))
  assert.match(sideOut.text, /subagentPerm=\{"outcome":"selected","optionId":"allow-once"\}/)
  assert.match(sideOut.text, /fallback=\{"outcome":"cancelled"\}/)

  const writeOut = await collect(adapter.stream(options('sWrite', [user('q1', 'subagent-perm-probe')], { provider: 'local-antigravity' })))
  assert.match(writeOut.text, /subagentPerm=\{"code":-32603,"message":"unknown session"\}/)
  await adapter.closeSession('sWrite')
  const cancelsAfterIdleClose = existsSync(cancelLog) ? readFileSync(cancelLog, 'utf8').trim().split('\n').filter(Boolean) : []
  assert.equal(cancelsAfterIdleClose.length, 0, 'closing a fully idle session sends zero session/cancel notifications')

  // 2. Model discovery (`probeModels`) never reuses a privileged `fullAccess` worker when no standard worker is live
  adapter.dropModelCache('local-antigravity')
  const beforeProbeWorkers = spawnedWorkerArgs.length
  await adapter.probeModels('local-antigravity')
  assert.equal(spawnedWorkerArgs.length, beforeProbeWorkers + 1, 'probeModels spawned a dedicated non-privileged probe worker')
  assert.equal(spawnedWorkerArgs.at(-1)?.fullAccess, false, 'probeModels never runs on a --dangerously-skip-permissions worker')

  // 3. Maintenance `note()` on an open `local-antigravity` `danger-full-access` session runs WITHOUT `--dangerously-skip-permissions`
  const agyTurn1 = await collect(adapter.stream(options('sAgyFull', [user('m1', 'show-state')], { provider: 'local-antigravity' })))
  assert.match(agyTurn1.text, /skipPerms: true/)
  const agyNoteState = await adapter.note('sAgyFull', 'show-state', AbortSignal.timeout(5_000))
  assert.match(agyNoteState, /skipPerms: false/, 'note() detaches from privileged worker and runs without --dangerously-skip-permissions')
  const agyNotePerm = await adapter.note('sAgyFull', 'ask-permission', AbortSignal.timeout(5_000))
  assert.equal(agyNotePerm, 'denied', 'note() denies permission prompts on Antigravity danger-full-access sessions')
  const agyAfterNote = await collect(adapter.stream(options('sAgyFull', [user('m1', 'show-state'), user('m2', 'show-state')], { provider: 'local-antigravity' })))
  assert.match(agyAfterNote.text, /skipPerms: true/, 'next normal turn resumes on full-access worker')

  // 4. Preset downgrade while background / cliFullyIdle===false is active closes the privileged worker (sending 1 session/cancel)
  //    and enforces the downgraded `read-only` preset on the next prompt!
  const downTurn1 = await collect(adapter.stream(options('sAgyFull', [user('m1', 'show-state'), user('m2', 'show-state'), user('m3', `bg-cross-dir-read:${join(bgDir, 'down')}`)], { provider: 'local-antigravity' })))
  assert.equal(downTurn1.text, '[waiting for subagent]')
  presets.sAgyFull = 'read-only'
  const cancelsBeforeDowngrade = existsSync(cancelLog) ? readFileSync(cancelLog, 'utf8').trim().split('\n').filter(Boolean).length : 0
  const downTurn2 = await collect(adapter.stream(options('sAgyFull', [
    user('m1', 'show-state'),
    user('m2', 'show-state'),
    user('m3', `bg-cross-dir-read:${join(bgDir, 'down')}`),
    user('m4', 'ask-permission'),
  ], { provider: 'local-antigravity' })))
  const cancelsAfterDowngrade = existsSync(cancelLog) ? readFileSync(cancelLog, 'utf8').trim().split('\n').filter(Boolean).length : 0
  assert.equal(cancelsAfterDowngrade, cancelsBeforeDowngrade + 1, 'stopping a session with background work emits session/cancel')
  assert.equal(downTurn2.text, 'denied', 'downgraded read-only session immediately denies change permissions on a non-privileged worker')

  // 5. Non-fullAccessArgs engine (`this.fullAccess !== true`) with `session.preset === 'danger-full-access'`:
  //    background wakeup calling MCP without stdout `tool_call` metadata auto-allows and bridges MCP cleanly!
  const t1 = await collect(adapter.stream(options('sBgFull', [user('m1', `bg-silent-mcp-perm:${bgDir}`)], { provider: 'local-acp-hook-nofullargs' })))
  assert.equal(t1.text, '[running background job]')

  const woke1 = await Promise.race([
    waitNextWake(),
    new Promise((_, reject) => setTimeout(() => reject(new Error('silent background MCP call deadlocked')), 2_000)),
  ])
  assert.equal(woke1, 'sBgFull')

  const bgMsg = { id: 'bg-m1', role: 'user', content: [{ type: 'text', text: '[Background update]' }], source: { kind: 'bot', form: 'relay', role: 'background' } }
  const step1 = await collect(adapter.stream(options('sBgFull', [user('m1', `bg-silent-mcp-perm:${bgDir}`), bgMsg], { provider: 'local-acp-hook-nofullargs' })))
  assert.equal(step1.finish?.kind, 'tool-calls')
  assert.equal(step1.calls[0]?.name, 'message_bot')

  const step2 = await collect(adapter.stream(options('sBgFull', [
    user('m1', `bg-silent-mcp-perm:${bgDir}`),
    bgMsg,
    toolResult('tr-1', step1.calls[0].id, 'synced-ok'),
  ], { provider: 'local-acp-hook-nofullargs' })))
  assert.equal(step2.text, 'Silent background MCP outcome: synced-ok')
})
