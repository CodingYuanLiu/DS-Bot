// Lossless step-boundary steering for Antigravity (`agy acp`) via native
// Lifecycle Hooks (`PreInvocation`, `PostInvocation`, `Stop`).
//
// On POSIX, DS Bot listens on a private 0600 Unix domain socket inside a 0700
// directory (`DSBOT_STEER_DIR/steer.sock`, no TCP port) and touches a per-session
// `<conversationId>.steer` marker file only while a steer is queued. `hooks.json`
// checks that marker in pure `/bin/sh`, so 99.9% of tool/model steps exit in
// <0.3ms without spawning `node` at all (both in standalone terminal `agy` and
// inside DS Bot when no steer is waiting).
import { createServer, request as httpRequest } from 'node:http'
import { randomBytes } from 'node:crypto'
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { homedir, tmpdir } from 'node:os'
import { join } from 'node:path'

const TERMINAL_STOP_REASONS = new Set([
  'USER_CANCELED',
  'ERROR',
  'MAX_INVOCATIONS',
  'MAX_FORCED_INVOCATIONS',
  'MAX_TOKEN_BUDGET_EXCEEDED',
  'TERMINAL_CUSTOM_HOOK',
])

export const STEER_HOOK_SCRIPT = `import { readFileSync } from 'node:fs'
import { request as httpRequest } from 'node:http'

const sock = process.env.DSBOT_STEER_SOCK
const url = process.env.DSBOT_STEER_URL
const token = process.env.DSBOT_STEER_TOKEN
let wrote = false
const finish = (text) => {
  if (wrote) return
  wrote = true
  process.stdout.write(text)
}

if ((!sock && !url) || !token) {
  finish('{}')
  process.exit(0)
}

let input = {}
try {
  input = JSON.parse(readFileSync(0, 'utf8'))
} catch {
  finish('{}')
  process.exit(0)
}

const body = JSON.stringify({
  conversationId: input.conversationId,
  hookEventName: input.hookEventName,
  terminationReason: input.terminationReason,
  fullyIdle: input.fullyIdle,
})

let target
try {
  if (sock) target = { socketPath: sock, path: '/steer' }
  else {
    const parsed = new URL(url)
    target = { hostname: parsed.hostname, port: parsed.port, path: parsed.pathname }
  }
} catch {
  finish('{}')
  process.exit(0)
}

const req = httpRequest({
  ...target,
  method: 'POST',
  headers: {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(body),
    authorization: \`Bearer \${token}\`,
  },
  timeout: 3_000,
}, (res) => {
  let out = ''
  res.setEncoding('utf8')
  res.on('data', (chunk) => { out += chunk })
  res.on('end', () => {
    const trimmed = res.statusCode === 200 ? out.trim() : ''
    finish(trimmed !== '' ? trimmed : '{}')
  })
})
req.on('timeout', () => req.destroy(new Error('timeout')))
req.on('error', () => finish('{}'))
req.end(body)
`

function writeIfChanged(filePath, content) {
  try {
    if (readFileSync(filePath, 'utf8') === content) return
  } catch { /* missing file is written below */ }
  writeFileSync(filePath, content, 'utf8')
}

const safeMarkerName = id => (typeof id === 'string' && /^[\w.-]{1,128}$/.test(id) ? `${id}.steer` : undefined)

export function ensureAgySteerPlugin({ homeDir = homedir(), nodePath = process.execPath, platform = process.platform } = {}) {
  try {
    const dir = join(homeDir, '.gemini', 'config', 'plugins', 'dsbot-steer')
    mkdirSync(dir, { recursive: true })
    const quotedNode = `'${nodePath.replaceAll("'", "'\\''")}'`
    const stepCommand = platform === 'win32'
      ? `"${nodePath}" ./steer-hook.mjs`
      : `if [ -z "$DSBOT_STEER_DIR" ] || { [ -n "$ANTIGRAVITY_CONVERSATION_ID" ] && [ ! -f "$DSBOT_STEER_DIR/$ANTIGRAVITY_CONVERSATION_ID.steer" ]; }; then printf '{}'; else ${quotedNode} ./steer-hook.mjs; fi`
    const stopCommand = platform === 'win32'
      ? `"${nodePath}" ./steer-hook.mjs`
      : `if [ -z "$DSBOT_STEER_DIR" ]; then printf '{}'; else ${quotedNode} ./steer-hook.mjs; fi`
    const stepHandlers = [{ type: 'command', command: stepCommand, timeout: 5 }]
    const stopHandlers = [{ type: 'command', command: stopCommand, timeout: 5 }]
    const hooksJson = `${JSON.stringify({
      'dsbot-steer': {
        PreInvocation: stepHandlers,
        PostInvocation: stepHandlers,
        Stop: stopHandlers,
      },
    }, null, 2)}\n`
    const pluginJson = `${JSON.stringify({ name: 'dsbot-steer' }, null, 2)}\n`
    writeIfChanged(join(dir, 'plugin.json'), pluginJson)
    writeIfChanged(join(dir, 'hooks.json'), hooksJson)
    writeIfChanged(join(dir, 'steer-hook.mjs'), STEER_HOOK_SCRIPT)
    return dir
  } catch {
    return undefined
  }
}

export function postAgySteer({ socketPath, url, token }, payload) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(payload)
    const target = socketPath
      ? { socketPath, path: '/steer' }
      : (() => { const parsed = new URL(url); return { hostname: parsed.hostname, port: parsed.port, path: parsed.pathname } })()
    const req = httpRequest({
      ...target,
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'content-length': Buffer.byteLength(body),
        authorization: `Bearer ${token}`,
      },
      timeout: 3_000,
    }, (res) => {
      let raw = ''
      res.setEncoding('utf8')
      res.on('data', chunk => { raw += chunk })
      res.on('end', () => {
        let parsed = {}
        try { parsed = JSON.parse(raw) } catch {}
        resolve({ status: res.statusCode ?? 0, body: parsed })
      })
    })
    req.on('timeout', () => req.destroy(new Error('timeout')))
    req.on('error', reject)
    req.end(body)
  })
}

export class AgySteerServer {
  constructor({ drain, hasPending, onStopState, platform = process.platform } = {}) {
    this.token = randomBytes(32).toString('hex')
    this.drain = drain
    this.hasPending = hasPending
    this.onStopState = onStopState
    this.platform = platform
    this.dir = undefined
    this.socketPath = undefined
    this.url = undefined
    this.onExit = undefined
    this.server = createServer((request, response) => void this.serve(request, response).catch(() => {
      if (!response.headersSent) response.writeHead(500, { 'content-type': 'application/json' })
      response.end('{}')
    }))
  }

  async listen() {
    if (this.dir !== undefined) return this.env()
    // Keep Unix socket path short (<104 bytes SUN_PATH limit on macOS).
    const baseTmp = this.platform !== 'win32' && existsSync('/tmp') ? '/tmp' : tmpdir()
    const dir = mkdtempSync(join(baseTmp, 'dsb-steer-'))
    try { chmodSync(dir, 0o700) } catch {}
    this.dir = dir
    this.onExit = () => { try { rmSync(dir, { recursive: true, force: true }) } catch {} }
    process.once('exit', this.onExit)
    if (this.platform === 'win32') {
      await new Promise((resolve, reject) => this.server.listen(0, '127.0.0.1', () => resolve()).once('error', reject))
      this.server.unref?.()
      const { port } = this.server.address()
      this.url = `http://127.0.0.1:${port}/steer`
    } else {
      const sock = join(dir, 'steer.sock')
      await new Promise((resolve, reject) => this.server.listen(sock, () => resolve()).once('error', reject))
      try { chmodSync(sock, 0o600) } catch {}
      this.server.unref?.()
      this.socketPath = sock
    }
    return this.env()
  }

  env() {
    return {
      ...(this.dir ? { DSBOT_STEER_DIR: this.dir } : {}),
      ...(this.socketPath ? { DSBOT_STEER_SOCK: this.socketPath } : {}),
      ...(this.url ? { DSBOT_STEER_URL: this.url } : {}),
      DSBOT_STEER_TOKEN: this.token,
    }
  }

  setPending(conversationId, pending) {
    if (this.dir === undefined) return
    const name = safeMarkerName(conversationId)
    if (name === undefined) return
    const marker = join(this.dir, name)
    if (pending) {
      try { writeFileSync(marker, '', { mode: 0o600 }) } catch {}
    } else {
      try { rmSync(marker, { force: true }) } catch {}
    }
  }

  close() {
    const dir = this.dir
    this.dir = undefined
    this.socketPath = undefined
    this.url = undefined
    if (this.onExit !== undefined) {
      process.removeListener('exit', this.onExit)
      this.onExit = undefined
    }
    try { this.server.close() } catch {}
    try { this.server.closeAllConnections?.() } catch {}
    if (dir !== undefined) {
      try { rmSync(dir, { recursive: true, force: true }) } catch {}
    }
  }

  async serve(request, response) {
    if (request.headers.authorization !== `Bearer ${this.token}`) {
      response.writeHead(401, { 'content-type': 'application/json' })
      response.end(JSON.stringify({ error: 'unauthorized' }))
      return
    }
    if (request.method !== 'POST') {
      response.writeHead(405, { 'content-type': 'application/json' })
      response.end('{}')
      return
    }
    const raw = await readBody(request)
    let payload
    try { payload = JSON.parse(raw) } catch {
      response.writeHead(400, { 'content-type': 'application/json' })
      response.end('{}')
      return
    }
    const conversationId = typeof payload?.conversationId === 'string' ? payload.conversationId : ''
    const hookEventName = payload?.hookEventName
    let result = {}
    if (conversationId !== '') {
      if (hookEventName === 'PreInvocation' || hookEventName === 'PostInvocation') {
        const items = (this.drain?.(conversationId) ?? []).filter(text => typeof text === 'string' && text !== '')
        if (this.hasPending?.(conversationId) !== true) this.setPending(conversationId, false)
        if (items.length > 0) {
          result = {
            injectSteps: items.map(userMessage => ({ userMessage })),
            ...(hookEventName === 'PostInvocation' ? { terminationBehavior: 'force_continue' } : {}),
          }
        }
      } else if (hookEventName === 'Stop') {
        if (!TERMINAL_STOP_REASONS.has(payload?.terminationReason) && this.hasPending?.(conversationId) === true) {
          this.setPending(conversationId, true)
          result = {
            decision: 'continue',
            reason: 'A new message arrived while finishing this step; read and address it before ending the turn.',
          }
        } else {
          this.setPending(conversationId, false)
          if (typeof payload?.fullyIdle === 'boolean') {
            this.onStopState?.(conversationId, payload.fullyIdle)
          }
        }
      }
    }
    response.writeHead(200, { 'content-type': 'application/json' })
    response.end(JSON.stringify(result))
  }
}

const readBody = request => new Promise((resolve, reject) => {
  let body = ''
  request.setEncoding('utf8')
  request.on('data', chunk => { body += chunk })
  request.on('end', () => resolve(body))
  request.on('error', reject)
})
