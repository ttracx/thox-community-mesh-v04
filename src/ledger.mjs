// @ts-nocheck
/**
 * Permissioned metadata ledger for the community-mesh laboratory.
 *
 * Blockchain-like properties that are in scope: append-only hash chain,
 * replay rejection, non-transferable credit reservations, schema that
 * refuses payloads, and tamper evidence.
 *
 * Properties that are out of scope, because they would sit on the token
 * path: mining, a public mempool, gas, per-token commits, per-hop
 * consensus, and BFT finality before the next activation.
 *
 * CALCULATED and SIMULATED only. No device measurement. No throughput claim.
 */
import { sha256 } from "js-sha256";
import { makeNode, planShards, qwen36 } from "./engine.mjs";

/** @param {Record<string, unknown>} obj */
export function canonicalJson(obj) {
  const parts = Object.keys(obj)
    .sort()
    .map((k) => {
      const v = obj[k];
      const rendered =
        typeof v === "string"
          ? JSON.stringify(v)
          : Array.isArray(v) || (v && typeof v === "object")
            ? canonicalJson(/** @type {Record<string, unknown>} */ (v))
            : String(v);
      return `${JSON.stringify(k)}: ${rendered}`;
    });
  return `{${parts.join(", ")}}`;
}

const FORBIDDEN = new Set([
  "prompt",
  "prompts",
  "activation",
  "activations",
  "kv",
  "weights",
  "payload",
  "plaintext",
  "tokens",
  "embedding",
]);

/**
 * @param {Record<string, unknown>} tx
 */
export function schemaViolation(tx) {
  for (const key of Object.keys(tx)) {
    if (FORBIDDEN.has(key)) return key;
  }
  return null;
}

/**
 * @param {string} prev
 * @param {Record<string, unknown>} body
 */
export function blockHash(prev, body) {
  return sha256(`${prev}\n${canonicalJson(body)}`);
}

/**
 * @typedef {{ index: number, prev: string, hash: string, tx: Record<string, unknown> }} Block
 * @typedef {{ balances: Record<string, number>, reservations: Record<string, string>, seen: Record<string, boolean>, epoch: number, revoked: boolean, payloadBytes: number }} LedgerState
 */

export function emptyState() {
  return {
    balances: /** @type {Record<string, number>} */ ({}),
    reservations: /** @type {Record<string, string>} */ ({}),
    seen: /** @type {Record<string, boolean>} */ ({}),
    epoch: 1,
    revoked: false,
    payloadBytes: 0,
  };
}

/**
 * @param {LedgerState} state
 * @param {Record<string, unknown>} tx
 * @returns {{ ok: true, state: LedgerState } | { ok: false, reason: string }}
 */
export function applyTx(state, tx) {
  const forbidden = schemaViolation(tx);
  if (forbidden) return { ok: false, reason: `reject:payload-field:${forbidden}` };
  const id = typeof tx.txid === "string" ? tx.txid : "";
  if (!id) return { ok: false, reason: "reject:missing-txid" };
  if (state.seen[id]) return { ok: false, reason: "reject:replay" };
  const type = tx.type;
  /** @type {LedgerState} */
  const next = {
    balances: { ...state.balances },
    reservations: { ...state.reservations },
    seen: { ...state.seen, [id]: true },
    epoch: state.epoch,
    revoked: state.revoked,
    payloadBytes: 0,
  };
  if (type === "genesis" || type === "enroll" || type === "advertise") {
    return { ok: true, state: next };
  }
  if (type === "earn" || type === "endow") {
    const who = String(tx.principal);
    const amount = Number(tx.credits);
    if (!Number.isInteger(amount) || amount <= 0) return { ok: false, reason: "reject:bad-amount" };
    next.balances[who] = (next.balances[who] || 0) + amount;
    return { ok: true, state: next };
  }
  if (type === "transfer") {
    return { ok: false, reason: "reject:credits-non-transferable" };
  }
  if (type === "reserve") {
    if (state.revoked || tx.epoch !== state.epoch) return { ok: false, reason: "reject:stale-epoch" };
    const who = String(tx.principal);
    const amount = Number(tx.credits);
    const rid = String(tx.reservationId);
    if (!Number.isInteger(amount) || amount <= 0) return { ok: false, reason: "reject:bad-amount" };
    if (state.reservations[rid]) return { ok: false, reason: "reject:double-spend" };
    if ((state.balances[who] || 0) < amount) return { ok: false, reason: "reject:insufficient-credits" };
    next.balances[who] = (next.balances[who] || 0) - amount;
    next.reservations[rid] = who;
    return { ok: true, state: next };
  }
  if (type === "placement" || type === "grant" || type === "settle") {
    if (state.revoked || tx.epoch !== state.epoch) return { ok: false, reason: "reject:stale-epoch" };
    return { ok: true, state: next };
  }
  if (type === "revoke") {
    next.revoked = true;
    next.epoch = state.epoch + 1;
    return { ok: true, state: next };
  }
  return { ok: false, reason: "reject:unknown-type" };
}

/**
 * @param {Block[]} chain
 * @param {LedgerState} state
 * @param {Record<string, unknown>} tx
 */
export function tryAppend(chain, state, tx) {
  const applied = applyTx(state, tx);
  if (!applied.ok) return { ok: false, reason: applied.reason, chain, state };
  const prev = chain.length ? chain[chain.length - 1].hash : "genesis";
  const body = { index: chain.length, prev, tx };
  const hash = blockHash(prev, body);
  /** @type {Block} */
  const block = { index: chain.length, prev, hash, tx };
  return { ok: true, reason: "appended", chain: chain.concat([block]), state: applied.state };
}

/**
 * @param {Block[]} chain
 */
export function verifyChain(chain) {
  let prev = "genesis";
  let state = emptyState();
  for (let i = 0; i < chain.length; i++) {
    const block = chain[i];
    if (block.index !== i || block.prev !== prev) {
      return { ok: false, at: i, reason: "reject:link", state };
    }
    const expect = blockHash(prev, { index: block.index, prev: block.prev, tx: block.tx });
    if (expect !== block.hash) return { ok: false, at: i, reason: "reject:hash", state };
    const applied = applyTx(state, block.tx);
    if (!applied.ok) return { ok: false, at: i, reason: applied.reason, state };
    state = applied.state;
    prev = block.hash;
  }
  return { ok: true, at: chain.length, reason: "verified", state };
}

/**
 * @param {Block[]} chain
 */
export function tamperedCopy(chain) {
  if (chain.length < 2) return chain;
  return chain.map((block, index) => {
    if (index !== 1) return block;
    const tx = { ...block.tx, note: "tampered" };
    return { ...block, tx };
  });
}

/**
 * Added delay on the token loop under three designs.
 * RTT is an assumption supplied by the caller, not a measurement.
 * Off-path settlement adds nothing inside the loop. The other two are
 * rejected designs shown only as arithmetic.
 * @param {{ tokens: number, stages: number, rttMs: number, batchTokens: number }} input
 */
export function latencyModel(input) {
  const tokens = input.tokens;
  const stages = input.stages;
  const rttMs = input.rttMs;
  const batchTokens = input.batchTokens;
  const hops = Math.max(0, stages - 1);
  const perTokenCommitMs = tokens * rttMs;
  const perHopCommitMs = tokens * hops * rttMs;
  const batches = Math.max(1, Math.ceil(tokens / batchTokens));
  return {
    label: "CALCULATED",
    tokens,
    stages,
    hops,
    rttMs,
    batchTokens,
    batches,
    offPathTokenLoopMs: 0,
    perTokenCommitMs,
    perHopCommitMs,
    asyncSettlementLagMs: rttMs,
    assumption:
      "One commit costs one assumed round trip. The household mesh does not wait on that round trip. Absolute tokens per second are not claimed.",
  };
}

export function placementFixture() {
  const model = qwen36(4096);
  const nodes = [0, 1, 2].map((i) =>
    makeNode({
      id: `c${i}`,
      role: "contributor",
      principal: `owner-c${i}`,
      kind: "opt-in-host",
      ramGb: 16,
      offers: true,
    }),
  );
  const subscriber = makeNode({
    id: "sub-a",
    role: "subscriber",
    principal: "subscriber-a",
    kind: "thoxkey",
    ramGb: 8,
    offers: false,
  });
  const plan = planShards(model, nodes);
  return { model, nodes, subscriber, plan };
}

/**
 * Deterministic protocol. Logical time is the block index. No wall clock.
 */
export function runProtocol() {
  const { model, nodes, plan } = placementFixture();
  /** @type {Block[]} */
  let chain = [];
  let state = emptyState();
  /** @type {{ ok: boolean, reason: string, tx: Record<string, unknown> }[]} */
  const rejected = [];

  /**
   * @param {Record<string, unknown>} tx
   */
  function push(tx) {
    const result = tryAppend(chain, state, tx);
    if (!result.ok) {
      rejected.push({ ok: false, reason: result.reason, tx });
      return result;
    }
    chain = result.chain;
    state = result.state;
    return result;
  }

  push({ type: "genesis", txid: "tx-genesis", scope: "community-mesh-lab" });
  for (const node of nodes) {
    push({ type: "enroll", txid: `tx-enroll-${node.id}`, principal: node.principal, nodeId: node.id, ramGb: node.ramGb });
    const credits = Math.round(node.ramGb * 0.6 * 1000);
    push({
      type: "advertise",
      txid: `tx-adv-${node.id}`,
      principal: node.principal,
      nodeId: node.id,
      usableMilliGib: credits,
    });
    push({ type: "earn", txid: `tx-earn-${node.id}`, principal: node.principal, credits, hours: 1 });
  }
  push({ type: "endow", txid: "tx-endow-sub", principal: "subscriber-a", credits: 8000, source: "subscription-grant" });

  const stages = plan.shards.map((s) => ({
    nodeId: s.nodeId,
    layers: `${s.start}-${s.end - 1}`,
  }));
  if (!plan.ok) {
    rejected.push({
      ok: false,
      reason: "reject:unplaceable-no-block",
      tx: { type: "placement", txid: "tx-place", modelFp: model.fingerprint },
    });
  } else {
    const cost = plan.shards.length * 1000;
    push({
      type: "reserve",
      txid: "tx-reserve-1",
      principal: "subscriber-a",
      credits: cost,
      reservationId: "rsv-1",
      epoch: 1,
    });
    push({
      type: "placement",
      txid: "tx-place-1",
      principal: "subscriber-a",
      reservationId: "rsv-1",
      modelFp: model.fingerprint,
      mode: plan.mode,
      stages: stages.length,
      epoch: 1,
    });
    push({
      type: "grant",
      txid: "tx-grant-1",
      principal: "subscriber-a",
      modelFp: model.fingerprint,
      scope: "stage-exec",
      epoch: 1,
    });
    push({
      type: "settle",
      txid: "tx-settle-1",
      principal: "subscriber-a",
      reservationId: "rsv-1",
      tokenCount: 128,
      epoch: 1,
    });
  }

  push({ type: "reserve", txid: "tx-reserve-1", principal: "subscriber-a", credits: 1000, reservationId: "rsv-1", epoch: 1 });
  push({ type: "reserve", txid: "tx-reserve-replay", principal: "subscriber-a", credits: 1000, reservationId: "rsv-1", epoch: 1 });
  push({ type: "transfer", txid: "tx-xfer", from: "subscriber-a", to: "subscriber-b", credits: 1000 });
  push({ type: "settle", txid: "tx-prompt", principal: "subscriber-a", epoch: 1, prompt: "secret prompt" });
  push({ type: "revoke", txid: "tx-revoke-1", principal: "subscriber-a", reason: "subscriber-revoke" });
  push({ type: "settle", txid: "tx-settle-late", principal: "subscriber-a", reservationId: "rsv-1", tokenCount: 16, epoch: 1 });

  const verification = verifyChain(chain);
  const tamper = verifyChain(tamperedCopy(chain));
  return {
    chain,
    state,
    rejected,
    verification,
    tamper,
    plan: {
      ok: plan.ok,
      mode: plan.mode,
      stages: plan.shards.map((s) => ({
        nodeId: s.nodeId,
        layers: `${s.start}–${s.end - 1}`,
        layerCount: s.layers,
      })),
      stageCount: plan.shards.length,
      model: model.name,
      fingerprint: model.fingerprint,
      kvPerToken: model.kvPerToken,
      weightBytes: model.weightBytes,
    },
    payloadBytes: chain.reduce((sum, block) => sum + (schemaViolation(block.tx) ? 1 : 0), 0),
  };
}

function isDirectRun() {
  const argv = globalThis.process?.argv;
  return Array.isArray(argv) && typeof argv[1] === "string" && argv[1].endsWith("ledger.mjs");
}

if (isDirectRun()) {
  const protocol = runProtocol();
  const latency = latencyModel({ tokens: 256, stages: protocol.plan.stageCount || 1, rttMs: 40, batchTokens: 64 });
  console.log(
    JSON.stringify(
      {
        blocks: protocol.chain.length,
        verified: protocol.verification.ok,
        tamperOk: protocol.tamper.ok,
        tamperReason: protocol.tamper.reason,
        tamperAt: protocol.tamper.at,
        rejected: protocol.rejected.map((r) => r.reason),
        mode: protocol.plan.mode,
        stages: protocol.plan.stages,
        balances: protocol.state.balances,
        epoch: protocol.state.epoch,
        revoked: protocol.state.revoked,
        latency,
        payloadBytes: protocol.payloadBytes,
      },
      null,
      2,
    ),
  );
}
