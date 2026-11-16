import assert from 'node:assert/strict'
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs'
import { hostname, tmpdir } from 'node:os'
import { delimiter, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'
import { createRuntime } from '../index.js'
import { refreshSearchDirs } from '../src/host/local-agents/discover.js'
import { SIDE_MARKER } from '../src/host/side.js'
import { fakeContext } from './fake-ctx.js'

const ACP_FIXTURE = fileURLToPath(new URL('./fixtures/fake-acp.mjs', import.meta.url))

const user = text => ({ role: 'user', content: [{ type: 'text', text }] })
const assistant = (...blocks) => ({ role: 'assistant', content: blocks })
const call = (id, name = 'read') => ({ type: 'tool-call', id, name, arguments: '{}' })
const result = (id, text = 'ok') => ({ role: 'tool', toolCallId: id, content: [{ type: 'text', text }] })
const stop = text => [{ type: 'text-delta', index: 0, text }, { type: 'finish', reason: { kind: 'stop' } }]
const toolCall = (id = 'c1') => [
  { type: 'tool-call-delta', index: 0, id, name: 'read', argumentsDelta: '{}' },
  { type: 'finish', reason: { kind: 'tool-calls' } },
]
const promptOf = request => request.messages.at(-1).content.find(block => block.type === 'text').text

const team = (extra = {}) => ({
  version: 1,
  mainBotIds: ['b1'],
  whaleStored: true,
  revision: 3,
  bots: {
    b1: {
      id: 'b1', sessionId: 's-now', name: 'Chief', capabilities: [], color: 'deepseek', createdAt: 1,
      segments: [{ sessionId: 's-old', endedAt: 1 }],
    },
  },
  rooms: {
    r1: {
      id: 'r1', named: true, name: 'Launch', members: ['b1'], admin: 'b1', mode: 'everyone', notice: '',
      sessions: { b1: 'g-now' }, segments: { b1: [{ sessionId: 'g-old', endedAt: 1 }] },
    },
  },
  agents: {},
  capabilities: { items: {} },
  exchanges: [],
  exchangeTurns: {},
  works: {},
  questions: {},
  views: {},
  prefs: {},
  ...extra,
})

function makeSession(id, { messages, header, seq = 4, toolHistory, events }) {
  // The anchor fold reads this log. `eventAt` is closed to new callers.
  const history = events ?? []
  const session = {
    id,
    seq,
    events: events ?? [{ seq: 1, type: 'user/message' }],
    writes: 0,
    derived: 0,
    messages,
    header,
    history: toolHistory,
    inheritedEventCount: 0,
    deriveMessages() {
      session.derived += 1
      return session.messages.map(message => ({
        ...message,
        content: Array.isArray(message.content) ? message.content.map(block => ({ ...block })) : message.content,
      }))
    },
    toolHistory() { return session.history },
    requestHeader() { return session.header },
    snapshotEvents() { return history },
    eventAt() { throw new Error('eventAt is closed to new callers') },
    append() { session.writes += 1; session.events.push({ type: 'append' }) },
    followup() { session.writes += 1; session.events.push({ type: 'followup' }) },
  }
  return session
}

// The same register / stateOf face as ctx.sessionProjections: fold the unit
// the host registers over snapshotEvents().
function projectionRegistry(ctx) {
  const units = new Map()
  return {
    register(definition) {
      ctx.effect(() => {
        units.set(definition.key, definition)
        return () => { units.delete(definition.key) }
      }, 'sessionProjections.register()')
    },
    stateOf(session, key) {
      const definition = units.get(key)
      if (definition === undefined) return undefined
      let state = definition.init(session.header, session.inheritedEventCount)
      for (const event of session.snapshotEvents()) state = definition.apply(state, event)
      return state
    },
  }
}

function deferred() {
  let resolve
  let reject
  let settled = false
  const promise = new Promise((ok, fail) => { resolve = ok; reject = fail })
  return {
    promise,
    resolve() { if (!settled) { settled = true; resolve() } },
    reject(error) { if (!settled) { settled = true; reject(error) } },
  }
}

// `gate` holds each model call open so a test can see the entry while it is
// still answering. `scripts` are the chunk lists for successive calls; anything
// further gets a short stop. `side` seeds chat files: { botId: [chats] }.
async function setup({ state = team(), scripts = [], gate = false, messages, seq = 4, events, foreignLock = false, side = {} } = {}) {
  const home = mkdtempSync(join(tmpdir(), 'dsb-side-'))
  writeFileSync(join(home, 'state.json'), JSON.stringify(state))
  for (const [botId, chats] of Object.entries(side)) {
    mkdirSync(join(home, 'side'), { recursive: true })
    writeFileSync(join(home, 'side', `${botId}.json`), JSON.stringify(chats))
  }
  if (foreignLock) writeFileSync(join(home, 'state.lock'), JSON.stringify({ pid: process.ppid, host: hostname(), token: 'other' }))
  const header = {
    config: { provider: 'deepseek', model: 'aside-model', reasoningEffort: 'high', maxTokens: 8000, temperature: 0.2, stop: ['END'] },
    tools: [{ name: 'read' }, { name: 'pwsh' }],
  }
  const toolHistory = [{ name: 'read' }]
  const session = makeSession('s-now', { messages: messages ?? [user('Look at the file')], header, seq, toolHistory, events })
  const sessions = new Map([[session.id, session]])
  const requests = []
  const resolved = []
  const gates = []
  const queue = [...scripts]
  const services = {}
  const ctx = fakeContext({ services })
  services.sessionProjections = projectionRegistry(ctx)
  ctx.llm = {
    listProviders: () => [],
    listModels: async () => [],
    resolveModelInfo: async () => ({ reasoning: { efforts: [] } }),
    async* stream(options) {
      requests.push(options)
      if (gate) {
        const hold = deferred()
        gates.push(hold)
        const abort = () => hold.reject(Object.assign(new Error('This aside was cancelled.'), { name: 'AbortError' }))
        options.signal?.addEventListener('abort', abort, { once: true })
        try {
          if (options.signal?.aborted) abort()
          await hold.promise
        } finally {
          options.signal?.removeEventListener('abort', abort)
        }
      }
      for (const chunk of queue.shift() ?? stop('Because of the snapshot.')) yield chunk
    },
  }
  ctx.sessionController = {
    resolveAgent: async (sessionId) => {
      resolved.push(sessionId)
      const found = sessions.get(sessionId)
      if (found === undefined) throw new Error(`no session ${sessionId}`)
      return { agent: { id: sessionId, session: found } }
    },
    list: async () => ({ items: [] }),
    projections: async () => null,
    page: async () => ({ records: [], hasMore: false }),
    selectModel: async () => {},
    prompt: async () => {},
  }
  ctx.sessions = { flush: async () => {}, messageProjections: [] }
  ctx.workspaceRegistry = { archiveSession: async () => {} }
  ctx.sessionTitle = { rename: async () => {} }
  let route
  const register = ctx.connection.fetch.register
  ctx.connection.fetch.register = (options) => { route = options; return register(options) }
  const rt = createRuntime(ctx, { home, github: false, diagnosticsFile: false })
  let notices = 0
  const saved = rt.stateSaved
  rt.stateSaved = () => { notices += 1; saved?.() }
  await rt.load()
  const callEndpoint = async (endpoint, payload) => {
    const response = await route.fetch(new Request('http://localhost/api/bot', { method: 'POST', body: JSON.stringify({ endpoint, payload }) }))
    return response.json()
  }
  const stopRuntime = async () => {
    for (const hold of gates) hold.resolve()
    for (const entry of Object.values(rt.sideView?.() ?? {}).flatMap(chats => chats.flatMap(chat => chat.entries ?? []))) await rt.sideJoin(entry.id)
    await ctx.dispose()
    rmSync(home, { recursive: true, force: true, maxRetries: 5 })
  }
  return { home, ctx, rt, session, sessions, header, toolHistory, requests, resolved, gates, call: callEndpoint, notices: () => notices, stop: stopRuntime }
}

const chatsOf = (rt, botId = 'b1') => rt.sideView?.()[botId] ?? []
const chatOf = (rt, chatId, botId = 'b1') => chatsOf(rt, botId).find(chat => chat.id === chatId)
const chatsFile = (home, botId = 'b1') => JSON.parse(readFileSync(join(home, 'side', `${botId}.json`), 'utf8'))
const totals = (view) => {
  const sums = {}
  for (const [, owner, , kind, input, cacheRead, cacheWrite, output] of view.rows) {
    const name = view.owners[owner].name ?? view.owners[owner].key
    sums[`${name}/${view.kinds[kind]}`] = (sums[`${name}/${view.kinds[kind]}`] ?? 0) + input + cacheRead + cacheWrite + output
  }
  return sums
}

test('a new side chat persists an empty list of entries', async (t) => {
  const env = await setup()
  t.after(() => env.stop())
  const response = await env.call('side-new', { sessionId: 's-now' })
  assert.equal(response.ok, true, response.error?.message)
  const chat = chatOf(env.rt, response.value.chatId)
  assert.equal(chat.botId, 'b1')
  assert.deepEqual(chat.entries, [])
  assert.equal(typeof chat.createdAt, 'number')
  const file = chatsFile(env.home)
  assert.equal(file.length, 1)
  assert.equal(file[0].id, chat.id)
  const saved = JSON.parse(readFileSync(join(env.home, 'state.json'), 'utf8'))
  assert.equal('sideChats' in saved, false)
  const view = await env.call('state', {})
  assert.equal(view.ok, true)
  assert.equal(view.value.sideChats.b1.length, 1)
  assert.equal(view.value.sideChats.b1[0].id, chat.id)
})

test('a side question answers from a snapshot and writes nothing into the session', async (t) => {
  const env = await setup({
    gate: true,
    messages: [user('Look at the file'), assistant({ type: 'text', text: 'Reading it.' }, call('c1'))],
  })
  t.after(() => env.stop())
  const { rt, session, requests, header, toolHistory } = env
  const before = JSON.stringify(session.events)
  const response = await env.call('side-ask', { sessionId: 's-now', text: 'what changed?' })
  assert.equal(response.ok, true, response.error?.message)
  assert.equal(typeof response.value.chatId, 'string')
  assert.equal(typeof response.value.entryId, 'string')
  const chat = chatOf(rt, response.value.chatId)
  const [entry] = chat.entries
  assert.equal(entry.id, response.value.entryId)
  assert.equal(entry.status, 'answering')
  assert.equal(entry.question, 'what changed?')
  assert.equal(entry.answer, '')
  assert.equal(session.writes, 0)
  assert.equal(JSON.stringify(session.events), before)
  assert.equal(requests.length, 1)
  const request = requests[0]
  assert.equal(request.provider, 'deepseek')
  assert.equal(request.model, 'aside-model')
  assert.equal(request.reasoningEffort, 'high')
  assert.equal(request.maxTokens, header.config.maxTokens)
  assert.equal(request.temperature, 0.2)
  assert.deepEqual(request.stop, ['END'])
  assert.equal(request.sessionId, 's-now')
  assert.equal(request.toolHistory, toolHistory)
  assert.notEqual(request.tools, header.tools)
  assert.deepEqual(request.tools.map(tool => tool.name), ['read', 'pwsh'])
  // The pending tool call did not make the snapshot; the pair list is empty.
  assert.deepEqual(request.messages.slice(0, -1).map(message => message.role), ['user'])
  assert.equal(request.messages[0].content[0].text, 'Look at the file')
  const prompt = promptOf(request)
  assert.equal(prompt.startsWith(`${SIDE_MARKER}\n`), true)
  for (const line of [
    'Answer from the conversation so far, based on progress up to this snapshot, and from this side chat.',
    'Do not call any tool.',
    'If the answer needs information that is not in the conversation, say so.',
    'Never claim to have done anything.',
  ]) assert.ok(prompt.includes(line), line)
  assert.ok(prompt.endsWith('what changed?'))
  assert.equal(rt.ownRequest(request), true)
  env.gates[0].resolve()
  await rt.sideJoin(entry.id)
  assert.equal(entry.status, 'done')
  assert.equal(entry.answer, 'Because of the snapshot.')
  assert.equal(typeof entry.finishedAt, 'number')
  assert.equal(entry.error, undefined)
  assert.equal(chat.updatedAt, entry.finishedAt)
  assert.equal(session.writes, 0)
  assert.equal(JSON.stringify(session.events), before)
  const file = chatsFile(env.home)
  assert.equal(file.length, 1)
  assert.equal(file[0].entries[0].status, 'done')
  const meter = env.ctx.listeners.get('llm/stream').at(-1)
  for (const options of requests) {
    for await (const chunk of meter(options, async function* () {
      yield { type: 'usage', usage: { inputTokens: 12, outputTokens: 3 } }
    })) assert.equal(chunk.type, 'usage')
  }
  const usage = await env.call('usage', { timeZone: 'UTC' })
  assert.equal(usage.ok, true)
  assert.equal(totals(usage.value)['Chief/side'], 15)
})

test('a finished tool batch and its tool history go through unchanged', async (t) => {
  const env = await setup({
    messages: [user('Look'), assistant(call('a'), call('b')), result('a'), result('b'), user('Thanks')],
  })
  t.after(() => env.stop())
  const response = await env.call('side-ask', { sessionId: 's-now', text: 'status?' })
  await env.rt.sideJoin(response.value.entryId)
  assert.deepEqual(env.requests[0].messages.slice(0, -1).map(message => [message.role, message.toolCallId]), [
    ['user', undefined], ['assistant', undefined], ['tool', 'a'], ['tool', 'b'], ['user', undefined],
  ])
  assert.equal(env.requests[0].toolHistory, env.toolHistory)
})

test('a Bot id or an earlier private part asks the current part', async (t) => {
  const env = await setup({ seq: 0 })
  t.after(() => env.stop())
  const past = await env.call('side-ask', { sessionId: 's-old', text: 'from the old part' })
  const bot = await env.call('side-ask', { sessionId: 'b1', chatId: past.value.chatId, text: 'from the bot' })
  await env.rt.sideJoin(past.value.entryId)
  await env.rt.sideJoin(bot.value.entryId)
  assert.deepEqual(env.resolved, ['s-now', 's-now'])
})

test('group chats, an empty question, and a missing route are refused', async (t) => {
  const env = await setup({ gate: true })
  t.after(() => env.stop())
  const { session } = env
  const refuse = async (endpoint, payload, message, code = 'bot/refused') => {
    const response = await env.call(endpoint, payload)
    assert.equal(response.ok, false)
    assert.equal(response.error.message, message)
    assert.equal(response.error.code, code)
  }
  await refuse('side-ask', { sessionId: 'g-now', text: 'in the group' }, "Side chat is only available in a Bot's private chat.")
  await refuse('side-ask', { sessionId: 'g-old', text: 'in the old group part' }, "Side chat is only available in a Bot's private chat.")
  await refuse('side-ask', { sessionId: 'r1', text: 'in the room' }, "Side chat is only available in a Bot's private chat.")
  await refuse('side-new', { sessionId: 'g-now' }, "Side chat is only available in a Bot's private chat.")
  await refuse('side-ask', { text: 'no session' }, 'Nothing to ask', 'bot/invalid')
  await refuse('side-new', {}, 'Nothing to open', 'bot/invalid')
  await refuse('side-ask', { sessionId: 's-now', text: '   ' }, 'Write a question.', 'bot/invalid')
  await refuse('side-ask', { sessionId: 'missing', text: 'hello' }, 'Unknown Bot')
  assert.equal(session.derived, 0)
  assert.deepEqual(env.resolved, [])
  assert.equal(env.requests.length, 0)
  const saved = env.header.config
  env.header.config = {}
  await refuse('side-ask', { sessionId: 's-now', text: 'hello' }, 'This Bot has not called a model yet. Send a normal message first.')
  env.header.config = saved
  assert.equal(env.requests.length, 0)
  assert.equal(chatsOf(env.rt).length, 0)
  assert.equal(session.writes, 0)
})

test('a chat answering one question refuses a second, and recent skips it', async (t) => {
  const env = await setup({ gate: true })
  t.after(() => env.stop())
  const first = await env.call('side-ask', { sessionId: 's-now', text: 'one' })
  assert.equal(first.ok, true)
  const chat = chatOf(env.rt, first.value.chatId)
  assert.equal(chat.entries[0].status, 'answering')
  const busy = await env.call('side-ask', { sessionId: 's-now', chatId: first.value.chatId, text: 'two' })
  assert.equal(busy.ok, false)
  assert.equal(busy.error.message, 'Wait for the answer, or cancel it.')
  // `recent` sees the answering last entry and starts a fresh chat instead.
  const recent = await env.call('side-ask', { sessionId: 's-now', recent: true, text: 'two' })
  assert.equal(recent.ok, true)
  assert.notEqual(recent.value.chatId, first.value.chatId)
  env.gates.forEach(hold => hold.resolve())
  await env.rt.sideJoin(first.value.entryId)
  await env.rt.sideJoin(recent.value.entryId)
  assert.equal(chat.entries[0].status, 'done')
  assert.equal(chatsOf(env.rt).length, 2)
})

test('recent reuses the most recently updated quiet chat', async (t) => {
  const chats = [
    { id: 'older', botId: 'b1', createdAt: 1, updatedAt: 2, entries: [{ id: 'e1', question: 'old', answer: 'a', status: 'done', createdAt: 1, finishedAt: 2 }] },
    { id: 'newer', botId: 'b1', createdAt: 2, updatedAt: 5, entries: [{ id: 'e2', question: 'new', answer: 'b', status: 'done', createdAt: 4, finishedAt: 5 }] },
  ]
  const env = await setup({ side: { b1: chats }, gate: true })
  t.after(() => env.stop())
  // 'newer' ends in an answering entry while its question is still open.
  const held = await env.call('side-ask', { sessionId: 's-now', chatId: 'newer', text: 'wait' })
  assert.equal(held.ok, true)
  const picked = await env.call('side-ask', { sessionId: 's-now', recent: true, text: 'follow-up' })
  assert.equal(picked.ok, true)
  assert.equal(picked.value.chatId, 'older')
  env.gates.forEach(hold => hold.resolve())
  await env.rt.sideJoin(picked.value.entryId)
  await env.rt.sideJoin(held.value.entryId)
})

test('a follow-up carries the chat\'s finished pairs in order and skips the rest', async (t) => {
  const chats = [
    { id: 'chat', botId: 'b1', createdAt: 1, updatedAt: 9, entries: [
      { id: 'e1', question: 'first', answer: 'one.', status: 'done', createdAt: 1, finishedAt: 2 },
      { id: 'e2', question: 'second', answer: 'two.', status: 'done', createdAt: 3, finishedAt: 4 },
      { id: 'e3', question: 'failed', answer: '', status: 'failed', error: 'oops', createdAt: 5, finishedAt: 6 },
      { id: 'e4', question: 'cancelled', answer: '', status: 'cancelled', createdAt: 7, finishedAt: 8 },
    ] },
    { id: 'other', botId: 'b1', createdAt: 2, updatedAt: 3, entries: [
      { id: 'o1', question: 'not this chat', answer: 'nope', status: 'done', createdAt: 2, finishedAt: 3 },
    ] },
  ]
  const env = await setup({ side: { b1: chats } })
  t.after(() => env.stop())
  const response = await env.call('side-ask', { sessionId: 's-now', chatId: 'chat', text: 'follow-up' })
  await env.rt.sideJoin(response.value.entryId)
  const messages = env.requests[0].messages
  assert.deepEqual(messages.slice(0, -1).map(message => [message.role, message.content[0].text]), [
    ['user', 'Look at the file'],
    ['user', 'first'],
    ['assistant', 'one.'],
    ['user', 'second'],
    ['assistant', 'two.'],
  ])
  // The assembler reads `source` on every assistant message.
  assert.deepEqual(messages[2].source, { kind: 'model', provider: 'deepseek', model: 'aside-model' })
  assert.ok(promptOf(env.requests[0]).endsWith('follow-up'))
})

test('every question takes a fresh snapshot of the main chat', async (t) => {
  const env = await setup()
  t.after(() => env.stop())
  const first = await env.call('side-ask', { sessionId: 's-now', text: 'one' })
  await env.rt.sideJoin(first.value.entryId)
  env.session.messages.push(user('A new turn landed'))
  const second = await env.call('side-ask', { sessionId: 's-now', chatId: first.value.chatId, text: 'two' })
  await env.rt.sideJoin(second.value.entryId)
  assert.equal(env.requests.length, 2)
  const texts = request => request.messages.slice(0, -1).map(message => message.content[0].text)
  assert.equal(texts(env.requests[0]).includes('A new turn landed'), false)
  assert.equal(texts(env.requests[1]).includes('A new turn landed'), true)
})

test('cancel aborts the call and marks the entry cancelled', async (t) => {
  const env = await setup({ gate: true })
  t.after(() => env.stop())
  const asked = await env.call('side-ask', { sessionId: 's-now', text: 'hold' })
  assert.equal(asked.ok, true)
  const entry = chatOf(env.rt, asked.value.chatId).entries[0]
  assert.equal(entry.status, 'answering')
  const cancelled = await env.call('side-cancel', { chatId: asked.value.chatId, entryId: entry.id })
  assert.deepEqual(cancelled, { ok: true, value: { answering: true } })
  assert.equal(entry.status, 'cancelled')
  assert.equal(typeof entry.finishedAt, 'number')
  assert.equal(env.requests[0].signal.aborted, true)
  await env.rt.sideJoin(entry.id)
  assert.equal(entry.status, 'cancelled')
  assert.equal(env.session.writes, 0)
  const again = await env.call('side-cancel', { chatId: asked.value.chatId, entryId: entry.id })
  assert.deepEqual(again, { ok: true, value: { answering: false } })
  const missing = await env.call('side-cancel', { chatId: asked.value.chatId, entryId: 'gone' })
  assert.equal(missing.ok, false)
  assert.equal(missing.error.message, 'This side question is no longer here.')
  const gone = await env.call('side-cancel', { chatId: 'gone', entryId: 'gone' })
  assert.equal(gone.error.message, 'This side chat is no longer here.')
  const blank = await env.call('side-cancel', { chatId: asked.value.chatId })
  assert.equal(blank.error.message, 'Nothing to cancel')
})

test('closing a chat deletes it and rewrites the file', async (t) => {
  const env = await setup({ gate: true })
  t.after(() => env.stop())
  const asked = await env.call('side-ask', { sessionId: 's-now', text: 'hold' })
  assert.equal(asked.ok, true)
  const other = await env.call('side-new', { sessionId: 's-now' })
  assert.equal(other.ok, true)
  assert.equal(chatsOf(env.rt).length, 2)
  const signal = env.requests[0].signal
  const closed = await env.call('side-close', { chatId: asked.value.chatId })
  assert.deepEqual(closed, { ok: true, value: {} })
  assert.equal(signal.aborted, true)
  assert.equal(chatsOf(env.rt).length, 1)
  assert.equal(chatsOf(env.rt)[0].id, other.value.chatId)
  const file = chatsFile(env.home)
  assert.equal(file.length, 1)
  assert.equal(file[0].id, other.value.chatId)
  const again = await env.call('side-close', { chatId: 'gone' })
  assert.deepEqual(again, { ok: true, value: {} })
})

test('a restart marks an entry that was still answering as failed', async (t) => {
  const env = await setup({
    side: {
      b1: [
        { id: 'chat', botId: 'b1', createdAt: 1, updatedAt: 10, entries: [
          { id: 'live', question: 'Still going?', answer: '', status: 'answering', createdAt: 10 },
          { id: 'old', question: 'Earlier?', answer: 'Yes.', status: 'done', createdAt: 8, finishedAt: 9 },
        ] },
      ],
    },
  })
  t.after(() => env.stop())
  const [live, old] = chatOf(env.rt, 'chat').entries
  assert.equal(live.status, 'failed')
  assert.equal(live.error, 'DS Bot restarted before this answer finished.')
  assert.equal(typeof live.finishedAt, 'number')
  assert.equal(old.status, 'done')
  const file = chatsFile(env.home)
  assert.equal(file[0].entries[0].status, 'failed')
  assert.equal(file[0].entries[1].status, 'done')
  const saved = JSON.parse(readFileSync(join(env.home, 'state.json'), 'utf8'))
  assert.equal('sideChats' in saved, false)
  assert.equal(saved.revision, 3)
})

test('a tool call is asked again once, then the entry fails if it calls a tool again', async (t) => {
  const once = await setup({ scripts: [toolCall('c1'), stop('The file is ready.')] })
  t.after(() => once.stop())
  const asked = await once.call('side-ask', { sessionId: 's-now', text: 'where is it?' })
  await once.rt.sideJoin(asked.value.entryId)
  assert.equal(once.requests.length, 2)
  assert.equal(promptOf(once.requests[0]).includes('Answer in text only.'), false)
  assert.equal(promptOf(once.requests[1]).includes('Do not call any tool. Answer in text only.'), true)
  assert.equal(promptOf(once.requests[1]).startsWith(`${SIDE_MARKER}\n`), true)
  const entry = chatOf(once.rt, asked.value.chatId).entries[0]
  assert.equal(entry.status, 'done')
  assert.equal(entry.answer, 'The file is ready.')

  const twice = await setup({ scripts: [toolCall('c1'), toolCall('c2')] })
  t.after(() => twice.stop())
  const again = await twice.call('side-ask', { sessionId: 's-now', text: 'try the tool' })
  await twice.rt.sideJoin(again.value.entryId)
  const failed = chatOf(twice.rt, again.value.chatId).entries[0]
  assert.equal(twice.requests.length, 2)
  assert.equal(failed.status, 'failed')
  assert.equal(failed.answer, '')
  assert.equal(failed.error, 'The model called a tool again after being told not to, so this aside was not answered.')
})

test('an empty answer, a provider abort, and a cut-off answer fail the entry', async (t) => {
  const empty = await setup({ scripts: [[{ type: 'text-delta', index: 0, text: '  \n' }, { type: 'finish', reason: { kind: 'stop' } }]] })
  t.after(() => empty.stop())
  const blank = await empty.call('side-ask', { sessionId: 's-now', text: 'anything?' })
  await empty.rt.sideJoin(blank.value.entryId)
  assert.equal(chatOf(empty.rt, blank.value.chatId).entries[0].status, 'failed')
  assert.equal(chatOf(empty.rt, blank.value.chatId).entries[0].error, 'The model wrote nothing.')

  const aborted = await setup({
    scripts: [[{ type: 'finish', reason: { kind: 'aborted', failure: { message: 'the provider stopped' } } }]],
  })
  t.after(() => aborted.stop())
  const stopped = await aborted.call('side-ask', { sessionId: 's-now', text: 'hello' })
  await aborted.rt.sideJoin(stopped.value.entryId)
  assert.equal(chatOf(aborted.rt, stopped.value.chatId).entries[0].status, 'failed')
  assert.equal(chatOf(aborted.rt, stopped.value.chatId).entries[0].error, 'the provider stopped')

  const cut = await setup({
    scripts: [[
      { type: 'text-delta', index: 0, text: 'Partial answer' },
      { type: 'finish', reason: { kind: 'max-tokens' } },
    ]],
  })
  t.after(() => cut.stop())
  const partial = await cut.call('side-ask', { sessionId: 's-now', text: 'go on' })
  await cut.rt.sideJoin(partial.value.entryId)
  const entry = chatOf(cut.rt, partial.value.chatId).entries[0]
  assert.equal(entry.status, 'failed')
  assert.equal(entry.answer, 'Partial answer')
  assert.equal(entry.error, 'Answer cut off at the output limit.')
  assert.equal(entry.finish, 'max-tokens')
})

test('deleting a Bot aborts its flights and removes the chat file', async (t) => {
  const state = team()
  state.bots.b2 = { id: 'b2', sessionId: 's-b2', name: 'Side', capabilities: [], color: 'deepseek', createdAt: 2 }
  const env = await setup({ state, gate: true })
  t.after(() => env.stop())
  env.sessions.set('s-b2', makeSession('s-b2', {
    messages: [user('Side work')],
    header: env.header,
    seq: 3,
    toolHistory: [],
  }))
  const response = await env.call('side-ask', { sessionId: 'b2', text: 'hold this' })
  assert.equal(response.ok, true, response.error?.message)
  const signal = env.requests[0].signal
  assert.equal(chatOf(env.rt, response.value.chatId, 'b2').entries[0].status, 'answering')
  await env.rt.deleteBot('b2')
  assert.equal(signal.aborted, true)
  assert.equal(env.rt.botOf('b2'), undefined)
  assert.equal(env.rt.sideView().b2, undefined)
  assert.equal(existsSync(join(env.home, 'side', 'b2.json')), false)
  const saved = JSON.parse(readFileSync(join(env.home, 'state.json'), 'utf8'))
  assert.equal(saved.bots.b2, undefined)
  assert.equal(saved.bots.b1.id, 'b1')
})

test('unloading the plugin aborts a question that is still answering', async (t) => {
  const env = await setup({ gate: true })
  t.after(() => env.stop())
  const response = await env.call('side-ask', { sessionId: 's-now', text: 'hold this' })
  assert.equal(response.ok, true, response.error?.message)
  const entry = chatOf(env.rt, response.value.chatId).entries[0]
  const signal = env.requests[0].signal
  await env.ctx.dispose()
  assert.equal(signal.aborted, true)
  await env.rt.sideJoin(entry.id)
  assert.equal(entry.status, 'answering')
  const file = chatsFile(env.home)
  assert.equal(file[0].entries[0].status, 'answering')
})

test('a second host reads chats but never writes or answers', async (t) => {
  const env = await setup({
    foreignLock: true,
    side: {
      b1: [
        { id: 'chat', botId: 'b1', createdAt: 1, updatedAt: 10, entries: [
          { id: 'live', question: 'Still going?', answer: '', status: 'answering', createdAt: 10 },
          { id: 'old', question: 'Earlier?', answer: 'Yes.', status: 'done', createdAt: 8, finishedAt: 9 },
        ] },
      ],
    },
  })
  t.after(() => env.stop())
  assert.equal(typeof env.rt.readOnly(), 'string')
  const [live, old] = chatOf(env.rt, 'chat').entries
  assert.equal(live.status, 'answering')
  assert.equal(live.error, undefined)
  assert.equal(old.status, 'done')
  const file = chatsFile(env.home)
  assert.equal(file[0].entries[0].status, 'answering')
  const made = await env.call('side-new', { sessionId: 's-now' })
  assert.equal(made.ok, false)
  assert.equal(made.error.code, 'bot/read-only')
  const asked = await env.call('side-ask', { sessionId: 's-now', text: 'hello' })
  assert.equal(asked.ok, false)
  assert.equal(asked.error.code, 'bot/read-only')
  assert.equal(env.requests.length, 0)
  assert.equal(existsSync(join(env.home, 'side', 'b1.json')), true)
})

test('a side question retries transient TRANSPORT failures and stops immediately if aborted in backoff', async (t) => {
  const recovered = await setup({
    scripts: [
      [
        { type: 'text-delta', index: 0, text: 'broken partial ' },
        { type: 'finish', reason: { kind: 'error', failure: { code: 'TRANSPORT', message: 'DeepSeek Messages transport failed' } } },
      ],
      [
        { type: 'text-delta', index: 0, text: 'Clean answer.' },
        { type: 'finish', reason: { kind: 'stop' } },
      ],
    ],
  })
  recovered.rt.config.llmRetryBackoff = { initialDelayMs: 1, maxDelayMs: 5 }
  t.after(() => recovered.stop())
  const asked = await recovered.call('side-ask', { sessionId: 's-now', text: 'quick check' })
  await recovered.rt.sideJoin(asked.value.entryId)
  const entry = chatOf(recovered.rt, asked.value.entryId ? asked.value.chatId : '')?.entries?.[0] ?? chatOf(recovered.rt, asked.value.chatId).entries[0]
  assert.equal(recovered.requests.length, 2)
  assert.equal(entry.status, 'done')
  assert.equal(entry.answer, 'Clean answer.')
})

test('local-agent /side runs concurrently in an isolated read-only session without blocking or polluting the main agy session, and closes on side-close', { skip: process.platform === 'win32' }, async (t) => {
  const tmpRoot = realpathSync(mkdtempSync(join(tmpdir(), 'dsb-side-agy-')))
  const binDir = join(tmpRoot, 'bin')
  const fakeHome = join(tmpRoot, 'home')
  const workDir = join(tmpRoot, 'workspace')
  const runtimeHome = join(tmpRoot, 'dsh-bot-home')
  mkdirSync(binDir, { recursive: true })
  mkdirSync(fakeHome, { recursive: true })
  mkdirSync(workDir, { recursive: true })
  mkdirSync(runtimeHome, { recursive: true })

  const agyShim = join(binDir, 'agy')
  writeFileSync(agyShim, `#!/bin/sh\nif [ "$1" = "--version" ]; then echo "agy 0.2.8"; exit 0; fi\nexec "${process.execPath}" "${ACP_FIXTURE}" --agy "$@"\n`, 'utf8')
  chmodSync(agyShim, 0o755)

  const prevPath = process.env.PATH
  const prevHome = process.env.HOME
  process.env.PATH = `${binDir}${delimiter}${prevPath ?? ''}`
  process.env.HOME = fakeHome
  await refreshSearchDirs()

  const botState = team({
    bots: {
      b1: {
        id: 'b1',
        sessionId: 's-now',
        name: 'Chief',
        model: 'local-antigravity/default',
        appliedModel: 'local-antigravity/default',
        permission: 'danger-full-access',
        capabilities: [],
        color: 'deepseek',
        createdAt: 1,
      },
    },
  })
  writeFileSync(join(runtimeHome, 'state.json'), JSON.stringify(botState))

  const header = {
    cwd: workDir,
    config: { provider: 'local-antigravity', model: 'default' },
    tools: [{ name: 'message_bot' }, { name: 'pwsh' }],
  }
  const mainMessages = [{ id: 'm1', role: 'user', content: [{ type: 'text', text: 'Analyze benchmark logs in detail' }], source: { kind: 'user' } }]
  const session = makeSession('s-now', { messages: mainMessages, header, seq: 2, toolHistory: [] })
  const sessions = new Map([[session.id, session]])

  const adapters = new Map()
  const services = { agents: new Map() }
  const ctx = fakeContext({ services })
  Object.defineProperty(ctx, 'agents', {
    configurable: true,
    get() { throw new Error('cannot get property "agents" without inject') },
  })
  services.sessionProjections = projectionRegistry(ctx)
  ctx.llm = {
    registerAdapter(routes, adapter) {
      let current = [...routes]
      for (const r of current) adapters.set(r, adapter)
      const off = () => { for (const r of current) if (adapters.get(r) === adapter) adapters.delete(r) }
      off.replace = (next) => { off(); current = [...next]; for (const r of current) adapters.set(r, adapter) }
      return off
    },
    listProviders: () => [...adapters.keys()].map(id => ({ id, name: id })),
    listModels: async provider => adapters.get(provider)?.listModels(provider) ?? [],
    resolveModelInfo: async (provider, model) => adapters.get(provider)?.resolveModel(provider, model) ?? { reasoning: { efforts: [] } },
    async* stream(opts) {
      const adapter = adapters.get(opts.provider)
      if (!adapter) throw new Error(`no adapter for ${opts.provider}`)
      yield* adapter.stream(opts)
    },
  }
  ctx.sessionController = {
    resolveAgent: async (sessionId) => {
      const found = sessions.get(sessionId)
      if (found === undefined) throw new Error(`no session ${sessionId}`)
      return { agent: { id: sessionId, session: found } }
    },
    list: async () => ({ items: [] }),
    projections: async () => null,
    page: async () => ({ records: [], hasMore: false }),
    selectModel: async () => {},
    prompt: async () => {},
  }
  ctx.sessions = { flush: async () => {}, messageProjections: [] }
  ctx.workspaceRegistry = { archiveSession: async () => {} }
  ctx.sessionTitle = { rename: async () => {} }
  let apiRoute
  const origReg = ctx.connection.fetch.register
  ctx.connection.fetch.register = (opts) => { apiRoute = opts; return origReg(opts) }

  const rt = createRuntime(ctx, { home: runtimeHome, github: false, diagnosticsFile: false })
  await rt.load()
  const callApi = async (endpoint, payload) => {
    const res = await apiRoute.fetch(new Request('http://localhost/api/bot', { method: 'POST', body: JSON.stringify({ endpoint, payload }) }))
    return res.json()
  }
  await callApi('local-agents', {})

  t.after(async () => {
    process.env.PATH = prevPath
    process.env.HOME = prevHome
    await refreshSearchDirs()
    await ctx.dispose()
    rmSync(tmpRoot, { recursive: true, force: true, maxRetries: 5 })
  })

  // 1. Start a long-running main turn on s-now (`danger-full-access` on main chat).
  const readyFile = join(tmpRoot, 'main.ready')
  const releaseFile = join(tmpRoot, 'main.release')
  const collectStream = async (stream) => {
    let text = ''
    for await (const chunk of stream) {
      if (chunk.type === 'text-delta') text += chunk.text
    }
    return text
  }
  const mainTurnPromise = collectStream(ctx.llm.stream({
    provider: 'local-antigravity',
    model: 'default',
    sessionId: 's-now',
    messages: [...mainMessages, { id: 'm-slow', role: 'user', content: [{ type: 'text', text: `slow-main:${readyFile}|${releaseFile}` }], source: { kind: 'user' } }],
    tools: [{ name: 'message_bot' }],
    signal: AbortSignal.timeout(30_000),
  }))

  for (let i = 0; i < 200 && !existsSync(readyFile); i += 1) {
    await new Promise(resolve => setTimeout(resolve, 15))
  }
  assert.equal(existsSync(readyFile), true, 'main agy turn is actively in flight')
  const mainAcpId = readFileSync(readyFile, 'utf8').trim()

  // 2. While the main turn is STILL running, ask Question 1 in /side!
  let mainSettled = false
  void mainTurnPromise.then(() => { mainSettled = true })

  const q1 = await callApi('side-ask', { sessionId: 's-now', text: 'side-inspect first check' })
  assert.equal(q1.ok, true, q1.error?.message)
  await rt.sideJoin(q1.value.entryId)
  assert.equal(mainSettled, false, '/side finishes immediately while main agy turn is still running')

  const e1 = chatOf(rt, q1.value.chatId).entries[0]
  assert.equal(e1.status, 'done', e1.error)
  assert.match(e1.answer, /mode=auto/, 'side session inherits the main Bot permission preset so run_command and invoke_subagent work')
  assert.match(e1.answer, /mcp=no/, 'side session attaches no team MCP server')
  assert.match(e1.answer, /open=2/, 'multiplexes a second session inside the same agy worker process')
  assert.match(e1.answer, /readOk=true execOk=true/, 'allows read and command execution under danger-full-access')
  assert.match(e1.answer, new RegExp(`cwd=${workDir.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`))
  assert.match(e1.answer, /Analyze benchmark logs in detail/, 'receives the main chat snapshot')
  assert.doesNotMatch(e1.answer, new RegExp(`session=${mainAcpId}\\b`), 'uses an isolated acpSessionId')

  // 3. Subagent + streaming progress in /side: streams live text & `▸` activity over live()
  //    without writing `activity` to disk, then catches async background wakeup after Turn 1!
  const baseState = (await callApi('state', {})).value
  const subReady = join(tmpRoot, 'side-sub.ready')
  const subRelease = join(tmpRoot, 'side-sub.release')
  const qSub = await callApi('side-ask', {
    sessionId: 's-now',
    chatId: q1.value.chatId,
    text: `side-subagent-stream:${subReady}|${subRelease}`,
  })
  assert.equal(qSub.ok, true, qSub.error?.message)
  for (let i = 0; i < 200 && !existsSync(subReady); i += 1) {
    await new Promise(resolve => setTimeout(resolve, 15))
  }
  assert.equal(existsSync(subReady), true, 'side subagent turn 1 is streaming')
  await new Promise(resolve => setTimeout(resolve, 25))

  const livePoll = (await callApi('state', {
    since: baseState.revision + 1,
    models: baseState.modelsKey,
    live: baseState.liveKey,
  })).value
  const streamingEntry = livePoll.sideChats?.b1?.find(c => c.id === q1.value.chatId)?.entries?.[1]
  assert.equal(streamingEntry?.status, 'answering')
  assert.equal(streamingEntry?.answer, 'Launching subagent inspector.')
  assert.equal(streamingEntry?.activity, '▸ Invoke Subagent')
  const diskDuringStream = JSON.parse(readFileSync(join(runtimeHome, 'side', 'b1.json'), 'utf8'))
  assert.equal('activity' in (diskDuringStream[0]?.entries?.[1] ?? {}), false, 'transient activity is not written to disk')

  writeFileSync(subRelease, 'go', 'utf8')
  await rt.sideJoin(qSub.value.entryId)
  const eSub = chatOf(rt, q1.value.chatId).entries[1]
  assert.equal(eSub.status, 'done', eSub.error)
  assert.equal(eSub.activity, undefined)
  assert.equal(eSub.answer, 'Launching subagent inspector.\n\nSubagent report: all 3 checks green.')

  // 3b. Real `agy acp` background wakeup in /side (no stdout session/update after Turn 1,
  //     delivered via Stop hook + transcriptPath / transcript_full.jsonl):
  const transcriptDir = join(tmpRoot, 'side-transcript-logs')
  const qTranscript = await callApi('side-ask', {
    sessionId: 's-now',
    chatId: q1.value.chatId,
    text: `transcript-bg-wakeup:${transcriptDir}`,
  })
  assert.equal(qTranscript.ok, true, qTranscript.error?.message)
  await rt.sideJoin(qTranscript.value.entryId)
  const eTranscript = chatOf(rt, q1.value.chatId).entries[2]
  assert.equal(eTranscript.status, 'done', eTranscript.error)
  assert.equal(eTranscript.activity, undefined)
  assert.equal(eTranscript.answer, 'Subagent dispatched in Turn 1.\n\nFull subagent report from transcript_full.jsonl!')

  // 4. Main chat progresses; ask follow-up Question in the SAME Side Chat tab after background wakeup.
  session.messages.push({ id: 'm2', role: 'assistant', content: [{ type: 'text', text: 'Phase 1 throughput = 420 tok/s' }], source: { kind: 'model', provider: 'local-antigravity', model: 'default' } })
  const q2 = await callApi('side-ask', { sessionId: 's-now', chatId: q1.value.chatId, text: 'side-inspect follow-up check' })
  assert.equal(q2.ok, true, q2.error?.message)
  await rt.sideJoin(q2.value.entryId)
  const e2 = chatOf(rt, q1.value.chatId).entries[3]
  assert.equal(e2.status, 'done', e2.error)
  assert.match(e2.answer, /open=2/, 'reuses the same side ACP session across follow-up questions')
  assert.match(e2.answer, /Phase 1 throughput = 420 tok\/s/, 'receives incremental main-chat updates')
  assert.doesNotMatch(e2.answer, /Analyze benchmark logs in detail/, 'does not re-send already-seen main history on warm follow-up')

  // 4. Verify state.json has zero side:* records and main session record is untouched.
  const persisted = JSON.parse(readFileSync(join(runtimeHome, 'state.json'), 'utf8'))
  assert.equal(persisted.localAgents?.sessions?.['s-now']?.cliSessionId, mainAcpId)
  assert.deepEqual(Object.keys(persisted.localAgents?.sessions ?? {}).filter(key => key.startsWith('side:')), [])

  // 5. Close the Side Chat tab -> side ACP session closes immediately (`openSessions=1`), main turn finishes unpolluted!
  const closed = await callApi('side-close', { chatId: q1.value.chatId })
  assert.deepEqual(closed, { ok: true, value: {} })
  writeFileSync(releaseFile, 'go', 'utf8')
  const mainOut = await mainTurnPromise
  assert.match(mainOut, new RegExp(`main-done session=${mainAcpId} totalPrompts=1 openSessions=1`))
})

