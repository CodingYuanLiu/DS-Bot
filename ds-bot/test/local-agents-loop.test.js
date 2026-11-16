// A real DSH agent turn through the local-agent adapter: the loop, its tool
// dispatch and its steering deliver to the fake ACP CLI, not stubs.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { Context } from '@deepseek-ai/cordis'
import { LocalAttachmentStore } from '@deepseek-ai/dsh-attachment-local'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { SessionId } from '@deepseek-ai/dsh-session'
import { mountAgentLoopTestDependencies, mountAgentLoopTestHarness } from '@deepseek-ai/dsh-agent-loop-testkit'
import { LocalAgentAdapter } from '../src/host/local-agents/adapter.js'
import { AcpEngine } from '../src/host/local-agents/acp.js'
import { textOf } from '../src/host/text.js'
import { output } from '../src/host/tools.js'

const FIXTURE = fileURLToPath(new URL('./fixtures/fake-acp.mjs', import.meta.url))
const TEAM = ['message_bot', 'approve_agent_action']

function makeAdapter(ctx, records = {}, extra = {}) {
  const agyHomeDir = mkdtempSync(join(tmpdir(), 'dsb-loop-agy-'))
  return new LocalAgentAdapter({
    rt: { warn: () => {}, ctx },
    customAgents: () => [{ id: 'fake', name: 'Fake ACP', command: 'x', args: [] }],
    findExecutable: path => path,
    spawnEngine: (kind, route, spec, hooks = {}) => new AcpEngine(
      route,
      { ...spec, command: process.execPath, args: [FIXTURE, ...(route === 'local-antigravity' ? ['--agy', '--thought-levels'] : [])] },
      { findExecutable: path => path, agyHomeDir, ...hooks },
    ),
    record: sessionId => records[sessionId],
    saveRecord: (sessionId, record) => { records[sessionId] = record },
    presetFor: () => 'workspace-write',
    cwdFor: async () => mkdtempSync(join(tmpdir(), 'dsb-cwd-')),
    probeCwd: () => mkdtempSync(join(tmpdir(), 'dsb-probe-')),
    toolsFor: options => (options.tools ?? []).filter(tool => TEAM.includes(tool.name)),
    ...extra,
  })
}

async function boot({ approval, attachmentsHome, adapterDeps, provider = 'local-acp-fake' } = {}) {
  const ctx = new Context()
  await mountAgentLoopTestDependencies(ctx)
  if (attachmentsHome !== undefined) ctx.plugin(LocalAttachmentStore, { dshHome: attachmentsHome })
  const isSteerable = msg => msg?.role === 'user' && msg.source?.kind !== 'runtime-context' && msg.source?.role !== 'background'
  const adapter = makeAdapter(ctx, {}, {
    peekInboxSteers: (sessionId, excludeIds) => (ctx.agents?.get(sessionId)?.inbox?.nextStep ?? [])
      .filter(msg => isSteerable(msg) && !(excludeIds?.has(msg.id)))
      .map(msg => ({ id: msg.id, text: textOf(msg.content) })),
    hasInboxSteers: sessionId => (ctx.agents?.get(sessionId)?.inbox?.nextStep ?? []).some(isSteerable),
    ...adapterDeps,
  })
  ctx.llm.registerAdapter(['local-acp-fake', 'local-antigravity'], adapter)
  if (approval !== undefined) ctx.provide('approval', approval)
  await mountAgentLoopTestHarness(ctx)
  const agent = await ctx.agentLoop.create(SessionId('s1'), { provider, model: 'default' })
  return { ctx, agent, adapter }
}

const send = (agent, text) => agent.followup(createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } }))

function waitForIdle(ctx, agent) {
  return new Promise((resolve) => {
    const dispose = ctx.on('agent/status', ({ agent: subject, status }) => {
      if (subject === agent && status === 'idle') { dispose(); resolve() }
    })
  })
}

const texts = agent => agent.session.snapshotEvents()
  .filter(event => event.type === 'assistant/message')
  .map(event => (event.data.message.content ?? []).map(block => block.text ?? '').join('')).join('\n')
const eventTypes = agent => agent.session.snapshotEvents().map(event => event.type)

test('a real loop turn streams the CLI answer and ends completed', async (t) => {
  const { ctx, agent, adapter } = await boot()
  t.after(async () => { await adapter.dispose(); await ctx.fiber.dispose() })
  // The first turn replays the conversation (system section first); the next
  // message goes out as a delta prompt carrying the new text verbatim.
  send(agent, 'kick off')
  await waitForIdle(ctx, agent)
  send(agent, 'hello there')
  await waitForIdle(ctx, agent)
  assert.match(texts(agent), /ok: hello there/)
  assert.equal(agent.session.snapshotEvents().findLast(event => event.type === 'turn/end')?.data.reason.kind, 'completed')
})

test('a bridged team tool round-trips through real DSH dispatch', async (t) => {
  const { ctx, agent, adapter } = await boot()
  t.after(async () => { await adapter.dispose(); await ctx.fiber.dispose() })
  const calls = []
  ctx.tools.register({
    name: 'message_bot',
    description: 'Message another Bot',
    parameters: { type: 'object', properties: { bot: { type: 'string' }, text: { type: 'string' } } },
    output,
    async execute(args) { calls.push(args); return `delivered to ${args.to ?? args.bot}` },
  })
  send(agent, 'call-tool please')
  await waitForIdle(ctx, agent)
  assert.equal(calls.length, 1)
  assert.match(texts(agent), /tool said: delivered to Chief/)
  const types = eventTypes(agent)
  assert.ok(types.includes('tool/call') && types.includes('tool/result'))
})

test('input during a running turn reaches the CLI as the next prompt', async (t) => {
  const { ctx, agent, adapter } = await boot()
  t.after(async () => { await adapter.dispose(); await ctx.fiber.dispose() })
  send(agent, 'first turn')
  send(agent, 'and this too')
  await waitForIdle(ctx, agent)
  const all = texts(agent)
  // Both inputs reached the CLI as prompts in order — never two prompts at once.
  assert.match(all, /ok: and this too/)
  assert.equal((all.match(/ok: /g) ?? []).length, 2)
  for (const event of agent.session.snapshotEvents().filter(event => event.type === 'turn/end'))
    assert.equal(event.data.reason.kind, 'completed')
})

test('a permission prompt reaches the approval service and fails closed', async (t) => {
  const requests = []
  const { ctx, agent, adapter } = await boot({
    approval: { request: (req) => { requests.push(req); return Promise.resolve('allowed-once') } },
  })
  t.after(async () => { await adapter.dispose(); await ctx.fiber.dispose() })
  ctx.tools.register({
    name: 'approve_agent_action',
    description: 'Ask the user to approve an action',
    parameters: { type: 'object', properties: { action: { type: 'string' }, detail: { type: 'string' } } },
    output,
    async execute(args, exec) {
      const outcome = await ctx.get('approval').request({ agent: exec.agent, toolName: exec.name, callId: exec.callId, reason: args.action })
      return JSON.stringify({ decision: outcome === 'allowed-once' ? 'allow' : 'deny' })
    },
  })
  send(agent, 'ask-permission')
  await waitForIdle(ctx, agent)
  assert.equal(requests.length, 1)
  assert.equal(requests[0].toolName, 'approve_agent_action')
  assert.match(texts(agent), /allowed via allow-once/)
})

test('user image saved in LocalAttachmentStore reaches the local agent through the real loop', async (t) => {
  const home = mkdtempSync(join(tmpdir(), 'dsb-att-'))
  t.after(() => rmSync(home, { recursive: true, force: true }))
  const { ctx, agent, adapter } = await boot({ attachmentsHome: home })
  t.after(async () => { await adapter.dispose(); await ctx.fiber.dispose() })

  const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==', 'base64')
  const ref = await ctx.get('attachments').saveImage({ mediaType: 'image/png', data: png, name: 'image.png' })
  const stored = await ctx.get('attachments').readImage(ref)
  const expectedBase64 = Buffer.from(stored.data).toString('base64')

  agent.followup(createUserMessage({
    content: [{ type: 'text', text: 'inspect attached image' }, { type: 'image', attachment: ref }],
    source: { kind: 'user' },
  }))
  await waitForIdle(ctx, agent)
  assert.match(texts(agent), new RegExp(`"mimeType":"${stored.ref.mediaType}","data":"${expectedBase64}"`))
})

test('spontaneous ACP background wakeup opens a real DSH turn and settles a valid assistant/message into Session', async (t) => {
  let agentRef
  const { ctx, agent, adapter } = await boot({
    adapterDeps: {
      bgIdleSettleMs: 60,
      bgActiveSettleMs: 300,
      onBackgroundWake: async () => {
        if (!agentRef) return false
        agentRef.followup(createUserMessage({
          content: [{ type: 'text', text: '[Background update]' }],
          source: { kind: 'bot', form: 'relay', role: 'background' },
        }))
        return true
      },
    },
  })
  agentRef = agent
  t.after(async () => { await adapter.dispose(); await ctx.fiber.dispose() })

  const calls = []
  ctx.tools.register({
    name: 'message_bot',
    description: 'Message another Bot',
    parameters: { type: 'object', properties: { to: { type: 'string' }, message: { type: 'string' } } },
    output,
    async execute(args) { calls.push(args); return `delivered to ${args.to}` },
  })

  // Turn 1 finishes immediately while background work continues inside the ACP process.
  const secondTurnDone = new Promise((resolve) => {
    let idleCount = 0
    const dispose = ctx.on('agent/status', ({ agent: subject, status }) => {
      if (subject === agent && status === 'idle') {
        idleCount += 1
        if (idleCount >= 2) { dispose(); resolve() }
      }
    })
  })
  send(agent, 'run bg-wakeup-mcp')
  await secondTurnDone

  const assistantEvents = agent.session.snapshotEvents().filter(event => event.type === 'assistant/message')
  assert.equal(assistantEvents.length >= 2, true)
  assert.match(texts(agent), /started in background/)
  assert.match(texts(agent), /background finished: delivered to Chief/)
  assert.equal(calls.length, 1)
  for (const end of agent.session.snapshotEvents().filter(event => event.type === 'turn/end')) {
    assert.equal(end.data.reason.kind, 'completed')
  }
})

test('antigravity hook steer injects mid-turn user input at PostInvocation and Stop without cancelling or duplicating the prompt', async (t) => {
  const { ctx, agent, adapter } = await boot({ provider: 'local-antigravity' })
  t.after(async () => { await adapter.dispose(); await ctx.fiber.dispose() })

  // 1. Mid-turn steer caught by PostInvocation hook (`injectSteps` + `force_continue`).
  const done1 = waitForIdle(ctx, agent)
  send(agent, 'run hook-steer now')
  await new Promise(resolve => setTimeout(resolve, 15))
  agent.steer(createUserMessage({ content: [{ type: 'text', text: 'use port 8080 instead' }], source: { kind: 'user' } }))
  await done1

  const events1 = agent.session.snapshotEvents()
  assert.equal(events1.filter(event => event.type === 'turn/start').length, 1, 'completes inside a single DSH turn')
  assert.equal(events1.findLast(event => event.type === 'turn/end')?.data.reason.kind, 'completed')
  const seqStep1 = events1.findIndex(event => event.type === 'assistant/message' && JSON.stringify(event.data).includes('step1 working'))
  const seqUserSteer = events1.findIndex(event => event.type === 'user/message' && JSON.stringify(event.data).includes('use port 8080 instead'))
  const seqStep2 = events1.findIndex(event => event.type === 'assistant/message' && JSON.stringify(event.data).includes('steered: use port 8080 instead'))
  assert.ok(seqStep1 >= 0 && seqUserSteer > seqStep1 && seqStep2 > seqUserSteer, 'user steer bubble sits cleanly between Step 1 and Step 2 outputs')
  assert.doesNotMatch(texts(agent), /ok: .*use port 8080 instead/, 'steered input must not fire a second session/prompt')

  // 2. Late steer caught at Stop hook (`decision: continue` -> `PreInvocation` `injectSteps`).
  const done2 = waitForIdle(ctx, agent)
  send(agent, 'run hook-steer-stop now')
  await new Promise(resolve => setTimeout(resolve, 15))
  agent.steer(createUserMessage({ content: [{ type: 'text', text: 'check edge cases too' }], source: { kind: 'user' } }))
  await done2
  assert.match(texts(agent), /stop-steered: check edge cases too/)
})

test('real DSH loop accepts reasoningEffort on a cold local-antigravity adapter without UNSUPPORTED_REASONING_EFFORT', async (t) => {
  const ctx = new Context()
  await mountAgentLoopTestDependencies(ctx)
  const adapter = makeAdapter(ctx)
  ctx.llm.registerAdapter(['local-antigravity'], adapter)
  await mountAgentLoopTestHarness(ctx)
  const agent = await ctx.agentLoop.create(SessionId('s-effort'), {
    provider: 'local-antigravity',
    model: 'fake-large',
    reasoningEffort: 'medium',
  })
  t.after(async () => { await adapter.dispose(); await ctx.fiber.dispose() })

  send(agent, 'show-state')
  await waitForIdle(ctx, agent)
  assert.equal(agent.session.snapshotEvents().findLast(event => event.type === 'turn/end')?.data.reason.kind, 'completed')
  assert.match(texts(agent), /effort: medium/)
})

test('antigravity Stop hook + transcriptPath delivers idle subagent wakeup without stdout session/update or duplicate Turn 1 text', { skip: process.platform === 'win32' }, async (t) => {
  const ctx = new Context()
  await mountAgentLoopTestDependencies(ctx)
  let agentRef
  const adapter = makeAdapter(ctx, {}, {
    onBackgroundWake: async () => {
      if (!agentRef) return false
      const already = (agentRef.inbox?.nextTurn ?? []).some(msg => msg?.source?.kind === 'bot' && msg.source.role === 'background')
      if (!already) {
        agentRef.followup(createUserMessage({
          content: [{ type: 'text', text: '[Background update]' }],
          source: { kind: 'bot', form: 'relay', role: 'background' },
        }))
      }
      await ctx.sessions.flush(agentRef.session)
      return true
    },
  })
  ctx.llm.registerAdapter(['local-antigravity'], adapter)
  await mountAgentLoopTestHarness(ctx)
  const agent = await ctx.agentLoop.create(SessionId('s-transcript-bg'), {
    provider: 'local-antigravity',
    model: 'default',
  })
  agentRef = agent
  const logDir = mkdtempSync(join(tmpdir(), 'dsb-agy-transcript-'))
  t.after(async () => {
    await adapter.dispose()
    await ctx.fiber.dispose()
    rmSync(logDir, { recursive: true, force: true })
  })

  const secondTurnDone = new Promise((resolve) => {
    let idleCount = 0
    const dispose = ctx.on('agent/status', ({ agent: subject, status }) => {
      if (subject === agent && status === 'idle') {
        idleCount += 1
        if (idleCount >= 2) { dispose(); resolve() }
      }
    })
  })
  send(agent, `transcript-bg-wakeup:${logDir}`)
  await secondTurnDone

  const assistantTexts = agent.session.snapshotEvents()
    .filter(event => event.type === 'assistant/message')
    .map(event => (event.data?.message?.content ?? []).filter(b => b.type === 'text').map(b => b.text).join('\n'))
  assert.equal(assistantTexts.length, 2)
  assert.equal(assistantTexts[0], 'Subagent dispatched in Turn 1.')
  assert.equal(assistantTexts[1], 'Full subagent report from transcript_full.jsonl!')
})
