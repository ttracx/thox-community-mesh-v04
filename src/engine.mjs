// @ts-nocheck
/**
 * Community MeshStack laboratory arithmetic.
 *
 * Implements the v2 reference admission rule (decisions D22 and D23 in
 * ttracx/thox-meshstack-inference, reviewed 2026-11-10): weights, per-token
 * KV, fixed recurrent state, and a runtime-reserve floor admit together
 * against usable RAM; host workspace admits against a separate system budget.
 * The domains are never summed. Usable budgets are 60% of advertised RAM
 * (P4 RAM_BUDGET_FRACTION).
 *
 * This module establishes no device measurement and no community-mesh
 * measurement. Outputs are CALCULATED or SIMULATED.
 */
import { sha256 } from "js-sha256";

export const RAM_BUDGET_FRACTION = 0.6;
export const RESERVE_FRACTION = 0.25;
export const GIB = 1024 ** 3;

/** Frozen Qwen3.6-27B-Q4_0 geometry from the v2 model preset. Weight bytes are the published on-disk size. */
export const QWEN36_FIELDS = {
  name: "Qwen3.6-27B-Q4_0",
  paramsB: 27.78,
  layers: 64,
  hidden: 5120,
  kvHeads: 4,
  headDim: 256,
  quant: "q4_0",
  attentionLayers: 16,
  ssmStateBytesPerLayer: 128 * 6144 * 4,
  weightBytesOverride: 15_791_278_304,
  illustrative: false,
  source:
    "v2 preset qwen36_27b(): unsloth/Qwen3.6-27B-GGUF Q4_0 on-disk bytes; 16 of 64 layers full attention; SSM state is an fp32 arithmetic estimate from GGUF metadata, not a measured buffer.",
};

const DTYPE_BYTES = { fp16: 2, bf16: 2, q8_0: 1, q4_k: 0.5625, q4_0: 0.5625, fp8: 1 };

function canonicalJson(obj) {
  const parts = Object.keys(obj)
    .sort()
    .map((k) => `${JSON.stringify(k)}: ${typeof obj[k] === "string" ? JSON.stringify(obj[k]) : String(obj[k])}`);
  return `{${parts.join(", ")}}`;
}

export function gib(bytes) {
  return bytes / GIB;
}

export function usableBytes(ramGb) {
  return Math.trunc(ramGb * GIB * RAM_BUDGET_FRACTION);
}

export function makeModel(fields, context) {
  const attention =
    fields.attentionLayers == null ? fields.layers : fields.attentionLayers;
  const weightBytes =
    fields.weightBytesOverride != null
      ? fields.weightBytesOverride
      : Math.trunc(fields.paramsB * 1e9 * DTYPE_BYTES[fields.quant]);
  const recurrent = (fields.layers - attention) * (fields.ssmStateBytesPerLayer || 0);
  const kvPerToken = 2 * attention * fields.kvHeads * fields.headDim * 2;
  const blob = {
    head_dim: fields.headDim,
    hidden: fields.hidden,
    kv_heads: fields.kvHeads,
    layers: fields.layers,
    name: fields.name,
    params_b: fields.paramsB,
    quant: fields.quant,
  };
  if (fields.attentionLayers != null) blob.attention_layers = fields.attentionLayers;
  if (fields.ssmStateBytesPerLayer) blob.ssm_state_bytes_per_layer = fields.ssmStateBytesPerLayer;
  if (fields.weightBytesOverride != null) blob.weight_bytes_override = fields.weightBytesOverride;
  const canonical = canonicalJson(blob);
  return {
    ...fields,
    context,
    attention,
    weightBytes,
    recurrent,
    kvPerToken,
    kvTotal: kvPerToken * context,
    fingerprint: sha256(canonical),
    canonical,
  };
}

export function qwen36(context = 4096) {
  return makeModel(QWEN36_FIELDS, context);
}

/** Illustrative dense classes. Bytes are params × Q4 factor, not a weighed file. */
export function illustrative(name, paramsB, layers, hidden, kvHeads, headDim, context) {
  return makeModel(
    {
      name,
      paramsB,
      layers,
      hidden,
      kvHeads,
      headDim,
      quant: "q4_k",
      attentionLayers: null,
      ssmStateBytesPerLayer: 0,
      weightBytesOverride: null,
      illustrative: true,
      source: "ILLUSTRATIVE class geometry. Not a measured GGUF and not a THOX model card.",
    },
    context,
  );
}

export function makeNode(spec) {
  const system = spec.systemRamGb || 0;
  return {
    id: spec.id,
    role: spec.role,
    principal: spec.principal,
    kind: spec.kind || "host",
    ramGb: spec.ramGb,
    alive: spec.alive !== false,
    offers: spec.offers !== false,
    foreground: !!spec.foreground,
    thermalOk: spec.thermalOk !== false,
    usableRam: usableBytes(spec.ramGb),
    usableSystem: usableBytes(system || spec.ramGb),
  };
}

function stageAccounting(model, node, start, end, kvFraction, runtimeWorkspace, headroomBytes) {
  const weight = Math.floor((model.weightBytes * end) / model.layers) - Math.floor((model.weightBytes * start) / model.layers);
  const kv = Math.floor((model.kvTotal * end) / model.layers) - Math.floor((model.kvTotal * start) / model.layers);
  const recurrent =
    Math.floor((model.recurrent * end) / model.layers) - Math.floor((model.recurrent * start) / model.layers);
  const reserveFloor = Math.ceil(node.usableRam * kvFraction);
  const headroom = Math.max(headroomBytes, reserveFloor - kv - runtimeWorkspace);
  const workspace = (end - start) * model.hidden * model.context * 4;
  const required = weight + kv + runtimeWorkspace + headroom + recurrent;
  return {
    nodeId: node.id,
    role: node.role,
    principal: node.principal,
    ramGb: node.ramGb,
    start,
    end,
    layers: end - start,
    weight,
    kv,
    recurrent,
    headroom,
    workspace,
    required,
    usableRam: node.usableRam,
    usableSystem: node.usableSystem,
    ramFit: required <= node.usableRam,
    workspaceFit: workspace <= node.usableSystem,
  };
}

function fitsD23(stage) {
  return stage.ramFit && stage.workspaceFit;
}

function candidates(nodes) {
  return nodes
    .filter((n) => n.alive && n.offers && n.thermalOk && !n.foreground)
    .slice()
    .sort((a, b) => b.usableRam - a.usableRam || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

/**
 * Greedy contiguous pack, largest usable RAM first. Fail closed.
 * Returns { ok, reason, shards, mode } .
 */
export function planShards(model, nodes, opts = {}) {
  const kvFraction = opts.kvFraction ?? RESERVE_FRACTION;
  const runtimeWorkspace = opts.runtimeWorkspace ?? 0;
  const headroomBytes = opts.headroomBytes ?? 0;
  const pool = candidates(nodes);
  const shards = [];
  let layer = 0;
  for (const node of pool) {
    if (layer >= model.layers) break;
    let low = layer + 1;
    let high = model.layers;
    let selected = null;
    while (low <= high) {
      const middle = Math.floor((low + high) / 2);
      const candidate = stageAccounting(model, node, layer, middle, kvFraction, runtimeWorkspace, headroomBytes);
      if (fitsD23(candidate)) {
        selected = candidate;
        low = middle + 1;
      } else {
        high = middle - 1;
      }
    }
    if (!selected) continue;
    shards.push(selected);
    layer = selected.end;
  }
  if (layer < model.layers) {
    return {
      ok: false,
      rule: "D23",
      reason: `${model.layers - layer} layers unplaced`,
      unplaced: model.layers - layer,
      shards,
      mode: "unplaceable",
    };
  }
  return {
    ok: true,
    rule: "D23",
    reason: "admitted",
    unplaced: 0,
    shards,
    mode: shards.length === 1 ? "whole-request" : "pipeline",
  };
}

/**
 * Cross-check shaped like the household report's non-summed statement:
 * stage weights (plus recurrent state) must fit usable RAM, AND stage KV
 * must fit usable RAM, separately. No 25% reserve floor. Workspace is
 * reported but does not gate this cross-check. Not the code of record.
 */
export function planNonsummed(model, nodes) {
  const pool = candidates(nodes);
  const shards = [];
  let layer = 0;
  for (const node of pool) {
    if (layer >= model.layers) break;
    let low = layer + 1;
    let high = model.layers;
    let selected = null;
    while (low <= high) {
      const middle = Math.floor((low + high) / 2);
      const s = stageAccounting(model, node, layer, middle, 0, 0, 0);
      const weightSide = s.weight + s.recurrent <= node.usableRam;
      const kvSide = s.kv <= node.usableRam;
      if (weightSide && kvSide) {
        selected = s;
        low = middle + 1;
      } else high = middle - 1;
    }
    if (!selected) continue;
    shards.push(selected);
    layer = selected.end;
  }
  if (layer < model.layers) {
    return { ok: false, rule: "v1.1-nonsummed-crosscheck", unplaced: model.layers - layer, shards, mode: "unplaceable" };
  }
  return {
    ok: true,
    rule: "v1.1-nonsummed-crosscheck",
    unplaced: 0,
    shards,
    mode: shards.length === 1 ? "whole-request" : "pipeline",
  };
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return function rand() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Seeded contributor churn. SIMULATED. Does not model WAN latency. */
export function simulateChurn(model, nodes, { trials = 200, p = 0.2, seed = 20261110 } = {}) {
  const rand = mulberry32(seed);
  let placed = 0;
  let whole = 0;
  let pipeline = 0;
  const hist = {};
  for (let i = 0; i < trials; i++) {
    const shocked = nodes.map((n) => {
      if (n.role !== "contributor" || !n.offers) return n;
      return rand() < p ? { ...n, alive: false } : n;
    });
    const plan = planShards(model, shocked);
    const key = plan.ok ? String(plan.shards.length) : "unplaceable";
    hist[key] = (hist[key] || 0) + 1;
    if (plan.ok) {
      placed += 1;
      if (plan.mode === "whole-request") whole += 1;
      else pipeline += 1;
    }
  }
  return {
    label: "SIMULATED",
    trials,
    leaveProbability: p,
    seed,
    placed,
    placeableFraction: placed / trials,
    wholeRequest: whole,
    pipeline,
    histogram: hist,
    note: "Independent Bernoulli leave per contributor per trial. No latency, no partial stage failure, no measured bandwidth.",
  };
}

export function householdFloorNodes() {
  return [
    makeNode({ id: "thoxmini", role: "owner", principal: "household", kind: "thoxmini", ramGb: 2, offers: true }),
    makeNode({ id: "thoxkey", role: "owner", principal: "household", kind: "thoxkey", ramGb: 4, offers: true }),
    makeNode({ id: "thoxair", role: "owner", principal: "household", kind: "thoxair", ramGb: 6, offers: true }),
  ];
}

export function communityPoolNodes() {
  const nodes = [];
  const layout = [
    ["c1", 8], ["c2", 8], ["c3", 8], ["c4", 8],
    ["c5", 16], ["c6", 16], ["c7", 16], ["c8", 16],
    ["c9", 32], ["c10", 32], ["c11", 64],
  ];
  for (const [id, ram] of layout) {
    nodes.push(makeNode({ id, role: "contributor", principal: `owner-${id}`, kind: "opt-in-host", ramGb: ram, offers: true }));
  }
  nodes.push(makeNode({ id: "sub-a", role: "subscriber", principal: "subscriber-a", kind: "thoxkey", ramGb: 8, offers: false }));
  nodes.push(makeNode({ id: "sub-b", role: "subscriber", principal: "subscriber-b", kind: "host", ramGb: 16, offers: false }));
  return nodes;
}

export function singleNode(ramGb, id = `n${ramGb}`) {
  return [makeNode({ id, role: "contributor", principal: "solo", kind: "host", ramGb, offers: true })];
}

function summarize(model, nodes, plan) {
  return {
    model: model.name,
    illustrative: !!model.illustrative,
    context: model.context,
    weightBytes: model.weightBytes,
    kvPerToken: model.kvPerToken,
    kvTotal: model.kvTotal,
    recurrent: model.recurrent,
    fingerprint: model.fingerprint,
    nodes: nodes.filter((n) => n.offers && n.alive).map((n) => ({ id: n.id, ramGb: n.ramGb, usableRam: n.usableRam })),
    plan: {
      ok: plan.ok,
      rule: plan.rule,
      mode: plan.mode,
      unplaced: plan.unplaced,
      stages: plan.shards.map((s) => ({
        nodeId: s.nodeId,
        ramGb: s.ramGb,
        layers: `${s.start}–${s.end - 1}`,
        layerCount: s.layers,
        weightBytes: s.weight,
        kvBytes: s.kv,
        recurrentBytes: s.recurrent,
        reserveBytes: s.headroom,
        requiredBytes: s.required,
        workspaceBytes: s.workspace,
        ramFit: s.ramFit,
        workspaceFit: s.workspaceFit,
      })),
    },
  };
}

export function placementRow(model, nodes) {
  const d23 = planShards(model, nodes);
  const cross = planNonsummed(model, nodes);
  return {
    label: "CALCULATED",
    d23: summarize(model, nodes, d23),
    nonsummed: {
      ok: cross.ok,
      mode: cross.mode,
      unplaced: cross.unplaced,
      stages: cross.shards.length,
    },
  };
}

/** Deterministic privacy drill. SIMULATED protocol, not a security proof. */
export function grantDrill(modelFingerprint) {
  const checks = [];
  const log = [];
  function check(id, ok, detail) {
    checks.push({ id, ok, detail });
  }
  const root = {
    id: "g-root",
    principal: "subscriber-a",
    epoch: 7,
    scope: "community",
    modelFp: modelFingerprint,
    parent: null,
    kind: "root",
    live: true,
  };
  const kv = { id: "g-kv", principal: "subscriber-a", epoch: 7, scope: "kv-transfer", modelFp: modelFingerprint, parent: "g-root", kind: "attenuated", live: true };
  const stage = { id: "g-stage", principal: "subscriber-a", epoch: 7, scope: "stage-exec", modelFp: modelFingerprint, parent: "g-root", kind: "attenuated", live: true };
  const grants = { "g-root": root, "g-kv": kv, "g-stage": stage };

  function consume(grantId, principal, epoch, fp) {
    const g = grants[grantId];
    if (!g || !g.live) return "reject:shredded-or-missing";
    let cursor = g;
    const seen = new Set();
    while (cursor) {
      if (seen.has(cursor.id)) return "reject:cycle";
      seen.add(cursor.id);
      if (!cursor.live) return "reject:ancestor-dead";
      if (cursor.epoch !== epoch) return "reject:stale-epoch";
      cursor = cursor.parent ? grants[cursor.parent] : null;
    }
    if (principal !== g.principal) return "reject:cross-principal";
    if (fp !== g.modelFp) return "reject:fingerprint";
    return "accept";
  }

  check("live-kv", consume("g-kv", "subscriber-a", 7, modelFingerprint) === "accept", "matching principal, epoch, and fingerprint");
  check("stale-epoch", consume("g-kv", "subscriber-a", 6, modelFingerprint) === "reject:stale-epoch", "epoch 6 against lease 7");
  check("cross-principal", consume("g-stage", "subscriber-b", 7, modelFingerprint) === "reject:cross-principal", "subscriber-b presenting subscriber-a's grant");
  check("fingerprint", consume("g-kv", "subscriber-a", 7, "0".repeat(64)) === "reject:fingerprint", "mismatched model fingerprint");

  // Crypto-shred: revocation deletes descendants. Key material is the live bit.
  for (const g of Object.values(grants)) {
    if (g.id === "g-root" || g.parent === "g-root") g.live = false;
  }
  check("shred-kv", consume("g-kv", "subscriber-a", 7, modelFingerprint) === "reject:shredded-or-missing", "root revocation shreds the KV descendant");
  check("shred-stage", consume("g-stage", "subscriber-a", 7, modelFingerprint) === "reject:shredded-or-missing", "root revocation shreds the stage descendant");

  log.push({ channel: "coordinator", fields: ["capacity_gib", "block_id", "token_count", "load"], payloadBytes: 0 });
  check("coordinator-metadata-only", log.every((e) => e.payloadBytes === 0 && !("prompt" in e)), "coordinator record has no payload bytes and no prompt field");

  return {
    label: "SIMULATED",
    pass: checks.every((c) => c.ok),
    checks,
    note: "Protocol drill of fail-closed grant rules. Not an implementation of the production secrets graph and not a penetration test.",
  };
}

export const CREDIT_PER_GIB_HOUR = 1;
export const MILLI_PER_STAGE = 1000;

/** Definitional accounting identity. Not a market and not a measured price. */
export function creditSketch(nodes, hours = 1) {
  const offered = nodes
    .filter((n) => n.role === "contributor" && n.offers && n.alive)
    .reduce((s, n) => s + n.ramGb * RAM_BUDGET_FRACTION, 0);
  return {
    label: "CALCULATED",
    definition: "1 MeshCredit per usable GiB-hour offered. Stated constant, not a fitted price.",
    hours,
    usableGibOffered: offered,
    credits: offered * hours * CREDIT_PER_GIB_HOUR,
    ownerPriority: "P0 foreground excludes a node from community admission for that interval.",
  };
}

function briefPlan(plan) {
  return {
    ok: plan.ok,
    mode: plan.mode,
    unplaced: plan.unplaced,
    stageCount: plan.shards.length,
    stages: plan.shards.map((s) => ({
      nodeId: s.nodeId,
      ramGb: s.ramGb,
      layers: `${s.start}–${s.end - 1}`,
    })),
  };
}

/**
 * Product recommendation, calculated on the frozen model.
 * Home whole-request spends nothing on the community.
 * A spill reserves MILLI_PER_STAGE per stage. Idle hardware earns the same unit back.
 * RAM figures for the key are the stated 128 GB product capacity. Other rows are class floors, not SKUs.
 */
export function freedomPaths(context = 4096) {
  const model = qwen36(context);
  const key = makeNode({
    id: "thoxkey",
    role: "contributor",
    principal: "owner-key",
    kind: "thoxkey",
    ramGb: 128,
    offers: true,
  });
  const keyBusy = makeNode({
    id: "thoxkey",
    role: "contributor",
    principal: "owner-key",
    kind: "thoxkey",
    ramGb: 128,
    offers: true,
    foreground: true,
  });
  const desks = [0, 1, 2].map((i) =>
    makeNode({
      id: `desk-${i}`,
      role: "contributor",
      principal: `neighbor-${i}`,
      kind: "opt-in-host",
      ramGb: 16,
      offers: true,
    }),
  );
  const alone8 = planShards(model, [
    makeNode({ id: "endpoint", role: "contributor", principal: "owner-small", kind: "endpoint", ramGb: 8, offers: true }),
  ]);
  const onKey = planShards(model, [key]);
  const spill = planShards(model, desks);
  const busy = planShards(model, [keyBusy, ...desks]);
  const earnKey = Math.round(128 * RAM_BUDGET_FRACTION * MILLI_PER_STAGE);
  const spillMilli = spill.ok ? spill.shards.length * MILLI_PER_STAGE : 0;
  return {
    label: "CALCULATED",
    model: model.name,
    context,
    milliPerStage: MILLI_PER_STAGE,
    grantMilli: 8000,
    sessionsPerGrant: spillMilli ? Math.floor(8000 / spillMilli) : 0,
    grantRemainder: spillMilli ? 8000 % spillMilli : 8000,
    sessionsPerIdleKeyHour: spillMilli ? Math.floor(earnKey / spillMilli) : 0,
    rule: "A model that fits at home spends 0 community milli-credits. A spill reserves 1000 per stage. An idle node earns 1000 per usable GiB-hour. Stated constants, not a price and not a coin.",
    paths: [
      {
        id: "key",
        product: "ThoxKey",
        ramLabel: "128 GB",
        stance: "Run it on the key. Nothing leaves.",
        plan: briefPlan(onKey),
        communityMilli: 0,
        earnMilliPerHour: earnKey,
        enhances: "The key remains the private runtime. Idle, the same box funds the overflow for everyone else.",
      },
      {
        id: "endpoint",
        product: "Small endpoint",
        ramLabel: "8 GB class",
        stance: "It cannot hold the model. Neighbors can.",
        alone: briefPlan(alone8),
        plan: briefPlan(spill),
        communityMilli: spillMilli,
        earnMilliPerHour: 0,
        enhances: "The endpoint keeps the grant and the prompt. It does not need a bigger box, and it does not call a public API.",
      },
      {
        id: "busy",
        product: "ThoxKey in use",
        ramLabel: "128 GB, foreground",
        stance: "Your work wins. The mesh is the overflow.",
        plan: briefPlan(busy),
        communityMilli: busy.ok ? busy.shards.length * MILLI_PER_STAGE : 0,
        earnMilliPerHour: 0,
        enhances: "Owner priority is unchanged. Community capacity does not replace the product you are holding.",
      },
    ],
  };
}

export function kvCorrection(context = 4096) {
  const model = qwen36(context);
  const allLayerKvPerToken = 2 * model.layers * model.kvHeads * model.headDim * 2;
  return {
    label: "CALCULATED",
    context,
    attentionLayers: model.attention,
    layers: model.layers,
    correctKvPerToken: model.kvPerToken,
    allLayerKvPerToken,
    overstatementFactor: allLayerKvPerToken / model.kvPerToken,
    correctKvTotal: model.kvPerToken * context,
    allLayerKvTotal: allLayerKvPerToken * context,
    recurrentEstimate: model.recurrent,
    activationBytesPerBoundary: model.hidden * 4,
    weightBytes: model.weightBytes,
    fingerprint: model.fingerprint,
    note: "D22: per-token KV counts full-attention layers only. The all-layer figure is the pre-D22 overstatement (4× on this geometry). Recurrent state is an fp32 estimate, not a measured buffer. Activation bytes are hidden × 4, the fp32 residual D34 derives as ~20,480 B/token/boundary.",
  };
}

function pair(ram, count, prefix) {
  return Array.from({ length: count }, (_, i) =>
    makeNode({ id: `${prefix}${i}`, role: "contributor", principal: `p-${prefix}${i}`, kind: "opt-in-host", ramGb: ram, offers: true }),
  );
}

/** Context × topology admission. CALCULATED. Nodes are class floors, not a bill of materials. */
export function contextSweep(contexts = [1024, 2048, 4096, 8192, 16384, 32768]) {
  const topologies = [
    { title: "Household 2/4/6 GB", nodes: householdFloorNodes() },
    { title: "One 8 GB", nodes: singleNode(8, "s8") },
    { title: "One 16 GB", nodes: singleNode(16, "s16") },
    { title: "One 32 GB", nodes: singleNode(32, "s32") },
    { title: "One 64 GB", nodes: singleNode(64, "s64") },
    { title: "One 128 GB", nodes: singleNode(128, "s128") },
    { title: "Two 16 GB", nodes: pair(16, 2, "t16") },
    { title: "Three 16 GB", nodes: pair(16, 3, "h16") },
    { title: "Four 16 GB", nodes: pair(16, 4, "f16") },
    { title: "Six 16 GB", nodes: pair(16, 6, "x16") },
    { title: "Two 32 GB", nodes: pair(32, 2, "t32") },
  ];
  const cells = [];
  for (const context of contexts) {
    const model = qwen36(context);
    for (const spec of topologies) {
      const row = placementRow(model, spec.nodes);
      cells.push({
        context,
        title: spec.title,
        d23: row.d23.plan.ok ? row.d23.plan.mode : `unplaceable:${row.d23.plan.unplaced}`,
        stages: row.d23.plan.stages.length,
        cross: row.nonsummed.ok ? row.nonsummed.mode : `unplaceable:${row.nonsummed.unplaced}`,
        disagree: (row.d23.plan.ok ? row.d23.plan.mode : "unplaceable") !== (row.nonsummed.ok ? row.nonsummed.mode : "unplaceable"),
      });
    }
  }
  return { label: "CALCULATED", model: "Qwen3.6-27B-Q4_0", contexts, topologies: topologies.map((t) => t.title), cells };
}

/** Byte-level accounting for the canonical three-host spill. CALCULATED. */
export function workedPipeline(context = 4096) {
  const model = qwen36(context);
  const nodes = pair(16, 3, "desk");
  const plan = planShards(model, nodes);
  const cross = planNonsummed(model, nodes);
  return {
    label: "CALCULATED",
    context,
    model: model.name,
    fingerprint: model.fingerprint,
    weightBytes: model.weightBytes,
    kvPerToken: model.kvPerToken,
    kvTotal: model.kvTotal,
    recurrent: model.recurrent,
    ok: plan.ok,
    mode: plan.mode,
    unplaced: plan.unplaced,
    crossMode: cross.ok ? cross.mode : `unplaceable:${cross.unplaced}`,
    stages: plan.shards.map((s) => ({
      nodeId: s.nodeId,
      ramGb: s.ramGb,
      layers: `${s.start}–${s.end - 1}`,
      layerCount: s.layers,
      weight: s.weight,
      kv: s.kv,
      recurrent: s.recurrent,
      reserve: s.headroom,
      required: s.required,
      usableRam: s.usableRam,
      workspace: s.workspace,
      usableSystem: s.usableSystem,
      ramFit: s.ramFit,
      workspaceFit: s.workspaceFit,
      slack: s.usableRam - s.required,
    })),
  };
}

export function buildEvidence(context = 4096) {
  const frozen = qwen36(context);
  const frozen8k = qwen36(8192);
  const household = householdFloorNodes();
  const community = communityPoolNodes();
  const classes = [
    { title: "Household floors 2/4/6 GB", nodes: household, model: frozen },
    { title: "Household floors, context 8192", nodes: household, model: frozen8k },
    { title: "One 16 GB contributor", nodes: singleNode(16), model: frozen },
    { title: "One 32 GB contributor", nodes: singleNode(32), model: frozen },
    { title: "One 64 GB contributor", nodes: singleNode(64), model: frozen },
    { title: "Two 16 GB contributors", nodes: [16, 16].map((r, i) => makeNode({ id: `a${i}`, role: "contributor", principal: `p${i}`, ramGb: r, offers: true })), model: frozen },
    { title: "Two 32 GB contributors", nodes: [32, 32].map((r, i) => makeNode({ id: `b${i}`, role: "contributor", principal: `p${i}`, ramGb: r, offers: true })), model: frozen },
    { title: "Three 16 GB contributors", nodes: [16, 16, 16].map((r, i) => makeNode({ id: `d${i}`, role: "contributor", principal: `p${i}`, ramGb: r, offers: true })), model: frozen },
    { title: "Community pool (8×4, 16×4, 32×2, 64×1)", nodes: community, model: frozen },
    { title: "Community pool without the 64 GB host", nodes: community.filter((n) => n.ramGb !== 64), model: frozen },
    {
      title: "Six 16 GB contributors only",
      nodes: [0, 1, 2, 3, 4, 5].map((i) => makeNode({ id: `s16-${i}`, role: "contributor", principal: `p${i}`, ramGb: 16, offers: true })),
      model: frozen,
    },
    {
      title: "64 GB host held at owner priority P0",
      nodes: community.map((n) => (n.ramGb === 64 ? { ...n, foreground: true } : n)),
      model: frozen,
    },
  ];
  const illustrativeRows = [8, 32, 70].map((b) => {
    const geom = b === 8
      ? illustrative("illustrative-8B-q4", 8, 32, 4096, 8, 128, context)
      : b === 32
        ? illustrative("illustrative-32B-q4", 32, 64, 5120, 8, 128, context)
        : illustrative("illustrative-70B-q4", 70, 80, 8192, 8, 128, context);
    return {
      title: `${geom.name} on community pool`,
      row: placementRow(geom, community),
    };
  });

  const weightExceedsHousehold =
    frozen.weightBytes > household.reduce((s, n) => s + n.ramGb * GIB, 0);

  return {
    generatedBy: "src/lib/lab/engine.mjs",
    labels: {
      measured: "Only figures cited from Ross & Xaypanya 2026, DOI 10.5281/zenodo.23969070, on stated host classes.",
      calculated: "Admission arithmetic in this module.",
      simulated: "Seeded churn and the grant-protocol drill.",
    },
    frozen: {
      name: frozen.name,
      weightBytes: frozen.weightBytes,
      kvPerToken: frozen.kvPerToken,
      recurrentEstimate: frozen.recurrent,
      fingerprint: frozen.fingerprint,
      context,
    },
    householdWeightExceedsAggregateAdvertised: weightExceedsHousehold,
    householdAggregateBytes: household.reduce((s, n) => s + n.ramGb * GIB, 0),
    rows: classes.map((c) => ({ title: c.title, ...placementRow(c.model, c.nodes) })),
    illustrative: illustrativeRows,
    churn: simulateChurn(frozen, community, { trials: 200, p: 0.2, seed: 20261110 }),
    churnWithout64: simulateChurn(frozen, community.filter((n) => n.ramGb !== 64), { trials: 200, p: 0.2, seed: 20261110 }),
    churn16Only: simulateChurn(
      frozen,
      [0, 1, 2, 3, 4, 5].map((i) => makeNode({ id: `s16-${i}`, role: "contributor", principal: `p${i}`, ramGb: 16, offers: true })),
      { trials: 200, p: 0.35, seed: 20261110 },
    ),
    grants: grantDrill(frozen.fingerprint),
    credits: creditSketch(community, 1),
    measuredReuse: [
      { label: "MEASURED", item: "Byte-split zstd on real GGUF KV, bit-exact", value: "1.17×–1.20×", host: "CPU host class", rate: "422–935 MB/s encode" },
      { label: "MEASURED", item: "Parent-chained HMAC-SHA256", value: "9.7 µs per 64-token chunk", host: "host class in the household report" },
      { label: "MEASURED", item: "Live lease-fencing drill", value: "0 stale authorisations accepted", host: "deployed coordinator" },
    ],
  };
}

export function evidenceMarkdown(evidence) {
  const lines = [];
  lines.push("# Community MeshStack laboratory evidence");
  lines.push("");
  lines.push("Research specification only. No production community mesh is claimed. No THOX device measurement is claimed.");
  lines.push("");
  lines.push(`Generator: \`${evidence.generatedBy}\`. Context for the frozen model rows: ${evidence.frozen.context}.`);
  lines.push("");
  lines.push("## Frozen geometry");
  lines.push("");
  lines.push(`- Model: ${evidence.frozen.name}`);
  lines.push(`- On-disk weight bytes: ${evidence.frozen.weightBytes}`);
  lines.push(`- KV bytes/token (fp16, attention layers only): ${evidence.frozen.kvPerToken}`);
  lines.push(`- Recurrent-state estimate (bytes, not measured): ${evidence.frozen.recurrentEstimate}`);
  lines.push(`- Geometry fingerprint: \`${evidence.frozen.fingerprint}\``);
  lines.push(`- Household 2+4+6 GiB aggregate bytes: ${evidence.householdAggregateBytes}`);
  lines.push(`- Weights exceed that aggregate: ${evidence.householdWeightExceedsAggregateAdvertised}`);
  lines.push("");
  lines.push("## Placement (CALCULATED, rule D23)");
  lines.push("");
  lines.push("| Scenario | Result | Stages | Cross-check (non-summed) |");
  lines.push("|---|---|---|---|");
  for (const row of evidence.rows) {
    const mode = row.d23.plan.ok ? row.d23.plan.mode : `unplaceable (${row.d23.plan.unplaced} layers)`;
    const stages = row.d23.plan.stages.map((s) => `${s.nodeId} ${s.layers}`).join("; ") || "—";
    const cross = row.nonsummed.ok ? `${row.nonsummed.mode} (${row.nonsummed.stages})` : `unplaceable (${row.nonsummed.unplaced})`;
    lines.push(`| ${row.title} | ${mode} | ${stages} | ${cross} |`);
  }
  lines.push("");
  lines.push("The non-summed column is a cross-check shaped like the household report's statement that weight and KV domains are not added. It is not the v2 code of record. Where the columns disagree, D23 is the stricter planner.");
  lines.push("");
  lines.push("## Illustrative classes on the community pool (CALCULATED, not weighed files)");
  lines.push("");
  for (const row of evidence.illustrative) {
    const p = row.row.d23.plan;
    lines.push(`- ${row.title}: ${p.ok ? p.mode : "unplaceable"}, ${p.stages.length} stage(s)`);
  }
  lines.push("");
  lines.push("## Churn (SIMULATED)");
  lines.push("");
  const c = evidence.churn;
  lines.push(`- Full pool, p=${c.leaveProbability}, n=${c.trials}, seed=${c.seed}: placed ${c.placed}, whole-request ${c.wholeRequest}, pipeline ${c.pipeline}. Histogram ${JSON.stringify(c.histogram)}.`);
  const a = evidence.churnWithout64;
  lines.push(`- Pool without 64 GB, p=${a.leaveProbability}, n=${a.trials}, seed=${a.seed}: placed ${a.placed}, whole-request ${a.wholeRequest}, pipeline ${a.pipeline}. Histogram ${JSON.stringify(a.histogram)}.`);
  const b = evidence.churn16Only;
  lines.push(`- Six 16 GB nodes, p=${b.leaveProbability}, n=${b.trials}, seed=${b.seed}: placed ${b.placed}, pipeline ${b.pipeline}. Histogram ${JSON.stringify(b.histogram)}.`);
  lines.push("");
  lines.push(c.note);
  lines.push("");
  lines.push("## Grant drill (SIMULATED)");
  lines.push("");
  lines.push(`Pass: ${evidence.grants.pass}.`);
  for (const check of evidence.grants.checks) {
    lines.push(`- ${check.ok ? "PASS" : "FAIL"} ${check.id}: ${check.detail}`);
  }
  lines.push("");
  lines.push(evidence.grants.note);
  lines.push("");
  lines.push("## Credits (CALCULATED definition)");
  lines.push("");
  lines.push(`${evidence.credits.definition} Usable GiB offered: ${evidence.credits.usableGibOffered}. Credits for ${evidence.credits.hours} h: ${evidence.credits.credits}.`);
  lines.push("");
  lines.push("## Reused household measurements (MEASURED, not re-run here)");
  lines.push("");
  for (const m of evidence.measuredReuse) {
    lines.push(`- ${m.item}: ${m.value} (${m.host})`);
  }
  lines.push("");
  return lines.join("\n");
}

function isDirectRun() {
  const argv = globalThis.process?.argv;
  return Array.isArray(argv) && typeof argv[1] === "string" && argv[1].endsWith("engine.mjs");
}

if (isDirectRun()) {
  const evidence = buildEvidence(4096);
  const brief = {
    fingerprint: evidence.frozen.fingerprint,
    kvPerToken: evidence.frozen.kvPerToken,
    weightExceeds: evidence.householdWeightExceedsAggregateAdvertised,
    rows: evidence.rows.map((r) => ({
      title: r.title,
      d23: `${r.d23.plan.ok ? r.d23.plan.mode : "unplaceable"} (${r.d23.plan.unplaced} unplaced, ${r.d23.plan.stages.length} stages)`,
      stages: r.d23.plan.stages.map((s) => `${s.nodeId}:${s.layers}`),
      cross: `${r.nonsummed.ok ? r.nonsummed.mode : "unplaceable"} stages=${r.nonsummed.stages} unplaced=${r.nonsummed.unplaced}`,
    })),
    illustrative: evidence.illustrative.map((r) => ({
      title: r.title,
      mode: r.row.d23.plan.ok ? r.row.d23.plan.mode : "unplaceable",
      stages: r.row.d23.plan.stages.length,
    })),
    churn: {
      full: { placed: evidence.churn.placed, whole: evidence.churn.wholeRequest, pipeline: evidence.churn.pipeline, hist: evidence.churn.histogram },
      no64: { placed: evidence.churnWithout64.placed, whole: evidence.churnWithout64.wholeRequest, pipeline: evidence.churnWithout64.pipeline, hist: evidence.churnWithout64.histogram },
      sixteen: { placed: evidence.churn16Only.placed, whole: evidence.churn16Only.wholeRequest, pipeline: evidence.churn16Only.pipeline, hist: evidence.churn16Only.histogram, p: evidence.churn16Only.leaveProbability },
    },
    grantsPass: evidence.grants.pass,
    credits: evidence.credits.credits,
  };
  console.log(JSON.stringify(brief, null, 2));
}
