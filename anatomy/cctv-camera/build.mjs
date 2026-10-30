import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "../kit/iso-kit.mjs";
import * as G from "../kit/lathe.mjs";
import { tubePieces, tubeSvg } from "../kit/tube.mjs";
import * as A from "../kit/audit.mjs";
import { kitScript } from "../scripts/inline-kit.mjs";

// Per-stage latency of the pipeline, in milliseconds per frame, measured by the notebook's
// benchmark (docs/results.json, laptop CPU, live configuration). The three detector branches
// run in parallel, so the frame time is the measured total, not the sum of the stages.
const RESULTS = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "docs", "results.json");
const MEASURED = existsSync(RESULTS) ? JSON.parse(readFileSync(RESULTS, "utf8")).latency : null;
const r0 = (ms) => Math.round(ms);
export const STAGE_MS = MEASURED
  ? {
      pose: r0(MEASURED.stages.pose),
      body: r0(MEASURED.stages.body_parts),
      hands: r0(MEASURED.stages.hands + (MEASURED.stages.hands_roi ?? 0)),
      devices: r0(MEASURED.stages.devices + (MEASURED.stages.devices_roi ?? 0)),
      mask: Math.max(1, r0(MEASURED.stages.mask)),
    }
  : { pose: 18, body: 14, hands: 12, devices: 16, mask: 1 };

// The model the readout and the frame tile are driven from.
const STAGES = [
  ["pose", "pose"],
  ["body", "body"],
  ["hands", "hands"],
  ["devices", "devices"],
  ["mask", "mask"],
];
const FRAME_MS = MEASURED ? r0(MEASURED.mean_ms) : Object.values(STAGE_MS).reduce((sum, ms) => sum + ms, 0);
const FPS = 1000 / FRAME_MS;
const MASKED = [
  [0, 2],
  [1, 1],
  [2, 0],
];
const MODEL = {
  frameMs: FRAME_MS,
  fps: Math.round(FPS * 10) / 10,
  npuLine: `${STAGES.filter(([id]) => id !== "mask").map(([id, label]) => `${label} ${STAGE_MS[id]}`).join(" · ")} → ${FRAME_MS} ms`,
  masked: MASKED.length,
};

const HERE = dirname(fileURLToPath(import.meta.url));
const THEME = process.argv.includes("--light") ? "light" : "dark";
const DEG = Math.PI / 180;

const W = 980;
const H = 680;
const AZIMUTH = 54;
const ELEVATION = 30;

// World units are millimetres. x runs along the optical axis (front is +x), z is up.
const BENCH = { x: -238, y: -52, w: 388, d: 110, r: 10 };
const BENCH_H = 12;
const FOOT_H = 6;
const CHASSIS = { x: -98, y: -42, w: 196, d: 84, r: 2 };
const CH_Z = 82;
const FLOOR = 84.5;
const AXIS_Z = 116;
const COVER_TOP = 154;
const SHELL = 44.5;
const COVER_LIFT = 112;

const PIECES = {
  stand: { name: "Bench, bracket and chassis", say: "", move: [0, 0, 0] },
  cover: { name: "Cover and sunshield", say: "cover · sunshield keeps glare off the window", move: [0, 0, COVER_LIFT] },
  front: { name: "Front window", say: "front window · glass, gasket and retaining ring", move: [100, 0, 0] },
  ir: { name: "IR ring", say: "IR ring · 8 × 850 nm LEDs and a light sensor", move: [76, 0, 0] },
  lens: { name: "M12 lens", say: "M12 lens · the focus ring sets the image on the sensor", move: [50, 0, 0] },
  sensor: { name: "Image sensor board", say: "sensor board · CMOS sensor, IR-cut filter, one FPC out", move: [28, 0, 0] },
  soc: { name: "Main board · SoC + NPU", say: "main board · NPU masks each frame before the encoder sees it", move: [0, 0, 0] },
  io: { name: "I/O board", say: "I/O board · PoE, PHY and microSD sit after the NPU", move: [-36, 0, 0] },
  rear: { name: "Rear cap and PoE pigtail", say: "rear cap · cable gland and PoE pigtail", move: [-48, 0, 0] },
};
const STAGGER = ["cover", "front", "ir", "lens", "rear", "sensor", "soc", "io", "stand"];
const SPRING = { stagger: 0.035, stiffness: 64, damping: 14.2, substeps: 4 };
const ORDER = ["front", "ir", "lens", "sensor", "soc", "io", "rear", "cover"];

const P = k.fitProjection(
  [
    ...k.boxCorners(BENCH, -BENCH_H - FOOT_H, 0),
    [-98, -48, COVER_TOP + 5 + PIECES.cover.move[2]],
    [126, -48, COVER_TOP + 5 + PIECES.cover.move[2]],
    [126, 48, COVER_TOP + 5 + PIECES.cover.move[2]],
    [208, 44, 156],
    [208, -44, 156],
  ],
  W,
  H,
  { pad: 22, azimuth: AZIMUTH, elevation: ELEVATION },
);
const V = G.viewOf(P);
const at = (point) => k.iso(point, P);
const S = (paths, style) => k.solidSvg(paths, style);
const Ln = (d, style) => k.lineSvg(d, style);
const Dots = (points, style) => k.dotsSvg(points, style);

const R = A.recorder(P, { order: STAGGER, spring: SPRING });
const { nameFor } = R;
function put(piece, ref, svg, { bias = 0, shapes = [], name, extra = {} } = {}) {
  R.put({ piece, ride: piece, at: ref, bias, svg, move: PIECES[piece].move, shapes, name: name ?? nameFor(`${piece}.item`), ...extra });
}

const planOf = (x0, y0, x1, y1, r = 0.4) => ({ x: x0, y: y0, w: x1 - x0, d: y1 - y0, r });
function block(piece, name, [x0, y0, z0], [x1, y1, z1], style = {}, { r = 0.4, steps = 3, bevel = 0, extra = "", inner = "", bias = 0, owner } = {}) {
  const plan = planOf(x0, y0, x1, y1, r);
  const id = nameFor(`${piece}.${name}`);
  const shape = R.solid(A.slab(plan, z0, z1 - z0, { name: id, owner }, steps));
  put(piece, [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2], S(k.slabOf(plan, z0, z1 - z0, P, steps, bevel), { tone: "hi", ...style, inner }) + extra, { bias, shapes: [shape], name: id });
  return shape;
}

// A frame whose axis runs along x (dir = 1 points to the front), u = +y, v = up.
const AX = (x, y, z, dir = 1) => G.frameOf([x, y, z], [dir, 0, 0], [0, 1, 0], [0, 0, dir]);
const AY = (x, y, z) => G.frameOf([x, y, z], [0, 1, 0], [0, 0, 1], [1, 0, 0]);
const AZ = (x, y, z) => G.frameOf([x, y, z], [0, 0, 1], [1, 0, 0], [0, 1, 0]);

function turned(piece, name, F, profile, style = {}, { extra = "", inner = "", bias = 0, owner, shapesExtra = [] } = {}) {
  const paths = G.solidOf(profile, F, P, { smooth: false });
  const id = nameFor(`${piece}.${name}`);
  const shape = R.solid(A.solid(F, profile, { name: id, owner }));
  const mid = (profile[0][0] + profile[profile.length - 1][0]) / 2;
  put(piece, G.add3(F.o, G.mul3(F.a, mid)), S(paths, { tone: "hi", ...style, inner }) + extra, { bias, shapes: [shape, ...shapesExtra], name: id });
  return shape;
}
function band(piece, name, F, s0, s1, rIn, rOut, style = {}, { extra = "", inner = "", bias = 0, owner } = {}) {
  const paths = G.ringBand(s0, s1, rIn, rOut, F, P);
  const id = nameFor(`${piece}.${name}`);
  const shape = R.solid(A.ring(F, s0, s1, rIn, rOut, { name: id, owner }));
  put(piece, G.add3(F.o, G.mul3(F.a, (s0 + s1) / 2)), S(paths, { tone: "hi", ...style, inner }) + extra, { bias, shapes: [shape], name: id });
  return shape;
}
function disc(piece, name, F, s0, s1, r, style = {}, { extra = "", inner = "", bias = 0, owner } = {}) {
  const paths = G.disc(s0, s1, r, F, P);
  const id = nameFor(`${piece}.${name}`);
  const shape = R.solid(A.disc(F, s0, s1, r, { name: id, owner }));
  put(piece, G.add3(F.o, G.mul3(F.a, (s0 + s1) / 2)), S(paths, { tone: "hi", ...style, inner }) + extra, { bias, shapes: [shape], name: id });
  return shape;
}

// Lines drawn on a face whose normal is +x (board faces, the bezel, chips).
const onX = (x, points, closed = true) => k.pathOf(points.map(([y, z]) => at([x, y, z])), closed);
const rectX = (x, y0, z0, y1, z1) => onX(x, [[y0, z0], [y1, z0], [y1, z1], [y0, z1]]);
const ringX = (x, y, z, r) => G.circleOf(AX(x, y, z), 0, r, P);
const onY = (y, points, closed = true) => k.pathOf(points.map(([x, z]) => at([x, y, z])), closed);
const slotY = (y, x0, x1, z0, z1) => k.pathOf(k.roundedPlan({ x: x0, y: z0, w: x1 - x0, d: z1 - z0, r: Math.min(x1 - x0, z1 - z0) / 2 }, 4).map(([u, z]) => at([u, y, z])), true);

function board(piece, name, x0, x1, y0, y1, z0, z1) {
  const holes = [[y0 + 4, z0 + 4], [y1 - 4, z0 + 4], [y1 - 4, z1 - 4], [y0 + 4, z1 - 4]];
  const extra =
    Ln(holes.map(([y, z]) => ringX(x1, y, z, 1.9)).join(""), { tone: "lo" }) +
    Dots(holes.map(([y, z]) => at([x1, y, z])), { size: 0.5 }) +
    Ln(rectX(x1, y0 + 1.6, z0 + 1.6, y1 - 1.6, z1 - 1.6), { tone: "faint" });
  return block(piece, name, [x0, y0, z0], [x1, y1, z1], { tone: "mid" }, { r: 0.6, extra });
}
function chip(piece, name, x0, depth, y0, y1, z0, z1, style = {}, { extra = "", pin = true, bias = 0 } = {}) {
  const x1 = x0 + depth;
  const inset = Math.min(1.4, (y1 - y0) / 5, (z1 - z0) / 5);
  const face = Ln(rectX(x1, y0 + inset, z0 + inset, y1 - inset, z1 - inset), { tone: "faint" }) + (pin ? Dots([at([x1, y1 - inset - 1.2, z1 - inset - 1.2])], { size: 0.45, tone: "mid" }) : "");
  return block(piece, name, [x0, y0, z0], [x1, y1, z1], style, { r: 0.5, extra: face + extra, bias });
}
// A small two-terminal part (capacitor, resistor) on a board face.
function smd(piece, x0, y, z, wide = 2.2, tall = 1.2, deep = 1) {
  return block(piece, "smd", [x0, y - wide / 2, z - tall / 2], [x0 + deep, y + wide / 2, z + tall / 2], { tone: "mid", crease: "none" }, { r: 0.2, steps: 2 });
}
// A copper trace between two vias on a board face.
function trace(x, points) {
  return Ln(onX(x, points, false), { tone: "lo" }) + Dots([at([x, ...points[0]]), at([x, ...points[points.length - 1]])], { size: 0.55, tone: "mid" });
}
// A flat flex cable: a run on the chassis floor with a riser at each end.
function flex(piece, name, { x0, x1, y0, y1, riserA, riserB, top }) {
  const z0 = FLOOR;
  const z1 = FLOOR + 0.6;
  const lanes = [0.25, 0.5, 0.75].map((t) => y0 + (y1 - y0) * t);
  block(piece, `${name}.run`, [x0, y0, z0], [x1, y1, z1], { tone: "hi", crease: "faint" }, { r: 0.2, steps: 2, extra: Ln(lanes.map((y) => k.lineOnTop([x0, y], [x1, y], z1, P)).join(""), { tone: "faint" }) });
  for (const [rx0, rx1, rTop] of [riserA, riserB]) {
    const face = rx1;
    block(piece, `${name}.riser`, [rx0, y0, z1], [rx1, y1, rTop], { tone: "hi", crease: "faint" }, { r: 0.2, steps: 2, extra: Ln(lanes.map((y) => k.segment([face, y, z1], [face, y, rTop], P)).join(""), { tone: "faint" }) });
  }
}

const back = [];

// ---------- Stand: bench plate, wall-mount bracket on the bench, chassis plate ----------
back.push(`<path class="iso-halo" d="${k.haloOf(BENCH, -BENCH_H - FOOT_H, BENCH_H, P)}"/>`);
const byDepth = (points) => [...points].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));
for (const [x, y] of byDepth(k.corners(BENCH, 22))) {
  back.push(S(k.cylinder(x, y, 7, -BENCH_H - FOOT_H, 1.6, P, 28), { tone: "lo" }));
  back.push(S(k.cylinder(x, y, 5.4, -BENCH_H - FOOT_H + 1.6, FOOT_H - 1.6, P, 28), { tone: "mid" }));
  R.solid(A.cylinder(x, y, 7, -BENCH_H - FOOT_H, 1.6, { name: "stand.foot" }));
  R.solid(A.cylinder(x, y, 5.4, -BENCH_H - FOOT_H + 1.6, FOOT_H - 1.6, { name: "stand.foot" }));
}
back.push(S(k.slabOf(BENCH, -BENCH_H, BENCH_H, P, 8, 1.6), { tone: "mid" }));
R.solid(A.slab(BENCH, -BENCH_H, BENCH_H, { name: "stand.bench" }));
back.push(Ln(k.planOutline(k.insetPlan(BENCH, 5), 0, P), { tone: "lo" }));
const benchScrews = k.corners(BENCH, 10);
back.push(Ln(benchScrews.map(([x, y]) => k.ring(x, y, 2.2, 0, P, 16)).join(""), { tone: "lo" }));
back.push(Dots(benchScrews.map(([x, y]) => at([x, y, 0])), { size: 0.5 }));
const RULER = { x: -150, y: BENCH.y + BENCH.d - 16, w: 220, d: 7, r: 1 };
back.push(S(k.slabOf(RULER, 0, 0.6, P, 3), { tone: "lo", crease: "none" }));
const ticks = k.topTicks(RULER.x + 4, RULER.x + RULER.w - 4, 4, 5, RULER.y + 0.8, 0.6, [2, 3.6], P);
back.push(Ln(ticks.minor, { tone: "lo" }), Ln(ticks.major, { tone: "mid" }));
const LABEL = { x: 88, y: BENCH.y + BENCH.d - 20, w: 52, d: 12, r: 1.2 };
back.push(S(k.slabOf(LABEL, 0, 0.8, P, 3), { tone: "lo", crease: "none" }));
back.push(Ln(k.planOutline(k.insetPlan(LABEL, 1.6), 0.8, P, 3), { tone: "faint" }));
back.push(Ln([36, 26, 30].map((w, i) => k.lineOnTop([LABEL.x + 5, LABEL.y + 3.2 + i * 2.8], [LABEL.x + 5 + w, LABEL.y + 3.2 + i * 2.8], 0.8, P)).join(""), { tone: "lo", free: true }));
back.push(Dots([at([LABEL.x + 3, LABEL.y + LABEL.d / 2, 0.8]), at([LABEL.x + LABEL.w - 3, LABEL.y + LABEL.d / 2, 0.8])], { size: 0.45 }));

// Bracket: a foot flange, a column, a clamp collar, a ball joint and a cup under the mount plate.
const BALL = { x: 0, y: 0, z: 71, r: 8 };
const RIM = 6;
const RIM_DZ = Math.sqrt(BALL.r * BALL.r - RIM * RIM);
put("stand", [0, 0, 2], `<path class="iso-halo" d="${k.haloOf({ x: -30, y: -30, w: 60, d: 60, r: 30 }, 0, 4, P)}"/>` + S(k.cylinder(0, 0, 26, 0, 4, P, 64, 0.8), { tone: "mid" }) + Dots(G.dotsOf(AZ(0, 0, 0), 4, 20, 4, P, { all: true, phase: 45 * DEG }), { size: 0.55 }) + Ln([0, 1, 2, 3].map((i) => k.ring(20 * Math.cos((45 + 90 * i) * DEG), 20 * Math.sin((45 + 90 * i) * DEG), 2.4, 4, P, 16)).join(""), { tone: "lo" }), { name: "stand.flange", shapes: [R.solid(A.cylinder(0, 0, 26, 0, 4, { name: "stand.flange" }))] });
put("stand", [0, 0, 30], S(k.cylinder(0, 0, 9, 4, 52, P, 40), { tone: "mid", inner: Ln(k.sideArc(0, 0, 9, 18, P, 24) + k.sideArc(0, 0, 9, 46, P, 24), { tone: "faint" }) }), { name: "stand.column", shapes: [R.solid(A.cylinder(0, 0, 9, 4, 52, { name: "stand.column" }))] });
put("stand", [0, 0, 59], S(k.cylinder(0, 0, 12, 56, 6, P, 48, 0.6), { tone: "mid" }) + Ln(k.knurl(0, 0, 12, 57, 61, 44, P), { tone: "lo" }), { name: "stand.collar", shapes: [R.solid(A.cylinder(0, 0, 12, 56, 6, { name: "stand.collar" }))] });
const BALL_SHAPE = A.ball([0, 0, BALL.z], BALL.r, { name: "stand.ball" });
put("stand", [0, 0, 63.5], S(k.cylinder(0, 0, RIM, 62, BALL.z - RIM_DZ - 62, P, 32), { tone: "mid" }), { name: "stand.neck", shapes: [R.solid(A.cylinder(0, 0, RIM, 62, BALL.z - RIM_DZ - 62, { name: "stand.neck", seated: true, cut: [BALL_SHAPE] }))] });
const THUMB = G.frameOf([0, 12, 61.2], [0, 1, 0], [0, 0, 1], [1, 0, 0]);
put("stand", [0, 16, 61], S(G.disc(0, 3, 1.6, THUMB, P), { tone: "mid" }) + S(G.disc(3, 8, 3.2, THUMB, P), { tone: "mid" }) + Ln(G.ribsOf(THUMB, 3, 8, 3.2, 16, P, { seams: true }), { tone: "lo" }), { name: "stand.thumb", shapes: [R.solid(A.disc(THUMB, 0, 3, 1.6, { name: "stand.thumb" })), R.solid(A.disc(THUMB, 3, 8, 3.2, { name: "stand.thumb" }))] });
put("stand", [0, 0, BALL.z], S(G.sphereOf([0, 0, BALL.z], BALL.r, P), { tone: "hi" }), { name: "stand.ball", shapes: [R.solid(BALL_SHAPE)] });
put("stand", [0, 0, 79], S(k.cylinder(0, 0, RIM, BALL.z + RIM_DZ, 80 - BALL.z - RIM_DZ, P, 32), { tone: "mid" }), { name: "stand.cup", shapes: [R.solid(A.cylinder(0, 0, RIM, BALL.z + RIM_DZ, 80 - BALL.z - RIM_DZ, { name: "stand.cup", seated: true, cut: [BALL_SHAPE] }))] });
const MOUNT = planOf(-30, -22, 30, 22, 3);
put("stand", [0, 0, 81], S(k.slabOf(MOUNT, 80, 2, P, 4, 0.5), { tone: "mid" }) + Dots(k.corners(MOUNT, 5).map(([x, y]) => at([x, y, 82])), { size: 0.5 }), { name: "stand.mount", shapes: [R.solid(A.slab(MOUNT, 80, 2, { name: "stand.mount" }, 4))] });

const chassisScrews = [-90, -40, 40, 90].flatMap((x) => [[x, -39], [x, 39]]);
put(
  "stand",
  [0, 0, (CH_Z + FLOOR) / 2],
  S(k.slabOf(CHASSIS, CH_Z, FLOOR - CH_Z, P, 4, 0.8), { tone: "mid" }) +
    Ln(k.planOutline(k.insetPlan(CHASSIS, 5.5), FLOOR, P, 4), { tone: "faint" }) +
    Dots(chassisScrews.map(([x, y]) => at([x, y, FLOOR])), { size: 0.5 }),
  { name: "stand.chassis", bias: -0.5, shapes: [R.solid(A.slab(CHASSIS, CH_Z, FLOOR - CH_Z, { name: "stand.chassis" }, 4))] },
);

// ---------- Cover with sunshield (lifts off) ----------
{
  const vents = [];
  for (let x = -70; x <= -20; x += 7) vents.push(slotY(SHELL, x, x + 3.2, 98, 124));
  const WALL = COVER_TOP - 2.5;
  block("cover", "wall.far", [-98, -SHELL, CH_Z], [98, -42, WALL], { tone: "mid" }, { r: 0.4 });
  block("cover", "wall.near", [-98, 42, CH_Z], [98, SHELL, WALL], { tone: "mid" }, { r: 0.4, extra: Ln(vents.join(""), { tone: "lo" }) });
  const screws = [-80, -30, 30, 80].flatMap((x) => [[x, SHELL - 6], [x, -SHELL + 6]]);
  block("cover", "top", [-98, -SHELL, WALL], [98, SHELL, COVER_TOP], { tone: "mid" }, { r: 1, bevel: 0.8, extra: Dots(screws.map(([x, y]) => at([x, y, COVER_TOP])), { size: 0.5 }) });
  for (const x of [-60, 40]) block("cover", "spacer", [x, -30, COVER_TOP], [x + 10, 30, COVER_TOP + 3], { tone: "mid" }, { r: 1 });
  const shade = planOf(-92, -48, 128, 48, 4);
  block("cover", "sunshield", [shade.x, shade.y, COVER_TOP + 3], [shade.x + shade.w, shade.y + shade.d, COVER_TOP + 5], { tone: "mid" }, {
    r: 4,
    steps: 4,
    bevel: 0.8,
    extra: Ln(k.planOutline(k.insetPlan(shade, 6), COVER_TOP + 5, P, 4), { tone: "faint" }) + Dots([[-80, -40], [116, -40], [116, 40], [-80, 40]].map(([x, y]) => at([x, y, COVER_TOP + 5])), { size: 0.5 }),
  });
}

// ---------- Front window: bezel, gasket, retaining ring, glass ----------
{
  const X0 = 98;
  const X1 = 103;
  const corners = [[-38, 88], [38, 88], [38, 148], [-38, 148]];
  block("front", "bezel", [X0, -SHELL, CH_Z], [X1, SHELL, COVER_TOP], { tone: "mid" }, {
    r: 1,
    bevel: 0.6,
    extra: Ln(corners.map(([y, z]) => ringX(X1, y, z, 2.2)).join(""), { tone: "lo" }) + Dots(corners.map(([y, z]) => at([X1, y, z])), { size: 0.5 }),
  });
  const F = AX(X1, 0, AXIS_Z);
  band("front", "gasket", F, 0, 0.8, 24, 30, { tone: "mid" });
  const glassFace = G.circleOf(F, 3.4, 22, P);
  const glassMarks = Ln(G.arcOf(F, 3.4, 17, P, { least: -1, from: 110 * DEG, to: 150 * DEG }) + G.arcOf(F, 3.4, 13, P, { least: -1, from: 115 * DEG, to: 140 * DEG }), { tone: "faint", free: true });
  put(
    "front",
    G.add3(F.o, [2.1, 0, 0]),
    S(G.ringBand(0.8, 3.4, 22, 28.5, F, P), { tone: "hi" }) + Dots(G.dotsOf(F, 3.4, 25.2, 6, P, { all: true, phase: 30 * DEG }), { size: 0.45 }) + `<path class="cc-glass" d="${glassFace}"/>` + glassMarks + Ln(glassFace, { tone: "mid" }),
    { name: "front.window", shapes: [R.solid(A.ring(F, 0.8, 3.4, 22, 28.5, { name: "front.retainer" })), R.solid(A.disc(F, 0.8, 3.4, 22, { name: "front.glass" }))] },
  );
}

// ---------- IR ring: annular board, 8 LEDs, light sensor, standoffs to the bezel ----------
{
  const X = 86;
  const T = 1.6;
  const F = AX(X, 0, AXIS_Z);
  band("ir", "board", F, 0, T, 14, 30, { tone: "mid" }, { extra: Ln(G.circleOf(F, T, 28.4, P) + G.circleOf(F, T, 15.6, P), { tone: "faint" }) });
  for (let i = 0; i < 3; i++) {
    const a = (90 + 120 * i) * DEG;
    const y = 27 * Math.cos(a);
    const z = AXIS_Z + 27 * Math.sin(a);
    disc("ir", "standoff", AX(X + T, y, z), 0, 98 - X - T, 1.5, { tone: "mid" });
  }
  for (let i = 0; i < 8; i++) {
    const a = (22.5 + 45 * i) * DEG;
    const y = 22 * Math.cos(a);
    const z = AXIS_Z + 22 * Math.sin(a);
    const L = AX(X + T, y, z);
    const id = nameFor("ir.led");
    const shapes = [R.solid(A.disc(L, 0, 2.4, 2.6, { name: id, owner: id })), R.solid(A.ball([X + T + 2.4, y, z], 2.4, { name: id, owner: id }))];
    put("ir", [X + T + 2.4, y, z], S(G.disc(0, 2.4, 2.6, L, P), { tone: "mid" }) + S(G.sphereOf([X + T + 2.4, y, z], 2.4, P), { tone: "hi" }), { name: id, shapes });
  }
  disc("ir", "lightsensor", AX(X + T, 0, AXIS_Z + 18), 0, 1.4, 2.4, { tone: "hi" }, { extra: Ln(G.circleOf(AX(X + T, 0, AXIS_Z + 18), 1.4, 1.2, P), { tone: "lo" }) });
}

// ---------- M12 lens ----------
{
  const F = AX(52, 0, AXIS_Z);
  const profile = [[0, 6.2], [7, 6.2], [7, 8.4], [14, 8.4], [14, 9.6], [24, 9.6], [24, 8.2], [29, 8.2], [29, 6.8], [31, 6.8]];
  turned("lens", "barrel", F, profile, { tone: "hi" }, {
    extra:
      k.fadedSvg(G.ribsOf(F, 14, 24, 9.6, 48, P, { fade: [0.1, 0.55], seams: true }), { tone: "lo" }) +
      Ln(G.circleOf(F, 31, 5.2, P), { tone: "mid" }) +
      Ln(G.arcOf(F, 31, 3.6, P, { least: -1, from: 110 * DEG, to: 160 * DEG }), { tone: "faint", free: true }),
  });
}

// ---------- Image sensor board ----------
const SEN = { x0: 40, x1: 41.6, y0: -24, y1: 24, z0: 92, z1: 140 };
{
  board("sensor", "pcb", SEN.x0, SEN.x1, SEN.y0, SEN.y1, SEN.z0, SEN.z1);
  for (const [y0, y1] of [[12, 20], [-20, -12]]) {
    block("sensor", "bracket.foot", [30, y0, FLOOR], [40, y1, FLOOR + 2], { tone: "mid" }, { r: 0.4, extra: Dots([at([33, (y0 + y1) / 2, FLOOR + 2])], { size: 0.5 }) });
    block("sensor", "bracket.tab", [38, y0, FLOOR + 2], [40, y1, 100], { tone: "mid" }, { r: 0.4 });
  }
  const holder = [SEN.x1, -9, 107];
  block("sensor", "holder", holder, [52, 9, 125], { tone: "hi" }, {
    r: 1,
    bevel: 0.4,
    extra: Ln(ringX(52, 0, AXIS_Z, 7.4), { tone: "faint" }) + Dots([[-7, 109], [7, 109], [7, 123], [-7, 123]].map(([y, z]) => at([52, y, z])), { size: 0.45 }),
  });
  block("sensor", "ircut", [43, -9, 125], [51, 9, 127.4], { tone: "mid" }, { r: 0.6, extra: Ln(k.lineOnTop([43, -2], [51, -2], 127.4, P), { tone: "faint" }) });
  disc("sensor", "solenoid", AY(47, -7, 127.4 + 2.2), 0, 10, 2.2, { tone: "hi" }, { extra: Ln(G.ribsOf(AY(47, -7, 129.6), 2, 8, 2.2, 10, P, { seams: true }), { tone: "lo" }) });
  for (const [y, z] of [[-16, 100], [-16, 132], [16, 100], [16, 132], [-12, 96], [12, 136], [-20, 116], [20, 116]]) smd("sensor", SEN.x1, y, z);
  chip("sensor", "ldo", SEN.x1, 1, -21, -15, 130, 136, { tone: "mid" });
  block("sensor", "fpc.connector", [37, -7, 94], [SEN.x0, 7, 98], { tone: "mid" }, { r: 0.3 });
}

// ---------- Main board: SoC (ISP + H.265 encoder) under a heatsink, NPU, DDR, eMMC, secure element ----------
const SOC = { x0: -8, x1: -6.4, y0: -34, y1: 34, z0: 88, z1: 148 };
const NPU = { y0: 10, y1: 28, z0: 106, z1: 124 };
const SOCCHIP = { y0: -24, y1: -4, z0: 106, z1: 126 };
const CON1 = { y0: -6, y1: 6, z0: 93, z1: 99 };
const CON2 = { y0: -26, y1: -14, z0: 93, z1: 99 };
{
  board("soc", "pcb", SOC.x0, SOC.x1, SOC.y0, SOC.y1, SOC.z0, SOC.z1);
  for (const [y0, y1] of [[8, 16], [22, 30]]) {
    block("soc", "bracket.foot", [-18, y0, FLOOR], [SOC.x0, y1, FLOOR + 2], { tone: "mid" }, { r: 0.4, extra: Dots([at([-15, (y0 + y1) / 2, FLOOR + 2])], { size: 0.5 }) });
    block("soc", "bracket.tab", [SOC.x0 - 2, y0, FLOOR + 2], [SOC.x0, y1, 100], { tone: "mid" }, { r: 0.4 });
  }
  const X = SOC.x1;
  const core = [];
  for (let y = NPU.y0 + 5; y <= NPU.y1 - 4.9; y += 2) for (let z = NPU.z0 + 5; z <= NPU.z1 - 4.9; z += 2) core.push(at([X + 1.4, y, z]));
  chip("soc", "npu", X, 1.4, NPU.y0, NPU.y1, NPU.z0, NPU.z1, { tone: "hi", lit: true }, {
    extra: `<g class="cc-core" data-core>${Dots(core, { size: 0.42, tone: "hi" })}</g>` + Ln(rectX(X + 1.4, NPU.y0 + 4, NPU.z0 + 4, NPU.y1 - 4, NPU.z1 - 4), { tone: "lit" }),
  });
  chip("soc", "soc", X, 1.2, SOCCHIP.y0, SOCCHIP.y1, SOCCHIP.z0, SOCCHIP.z1, { tone: "hi" });
  block("soc", "heatsink", [X + 1.2, SOCCHIP.y0 + 0.5, SOCCHIP.z0 + 0.5], [X + 2.2, SOCCHIP.y1 - 0.5, SOCCHIP.z1 - 0.5], { tone: "mid" }, { r: 0.4 });
  for (let i = 0; i < 7; i++) {
    const y = SOCCHIP.y0 + 0.8 + i * 3;
    block("soc", "fin", [X + 2.2, y, SOCCHIP.z0 + 0.5], [X + 11, y + 0.8, SOCCHIP.z1 - 0.5], { tone: "mid" }, { r: 0.2, steps: 2 });
  }
  chip("soc", "ddr", X, 1.2, -24, -4, 132, 142, { tone: "hi" });
  chip("soc", "emmc", X, 1.2, 12, 26, 132, 141, { tone: "hi" });
  chip("soc", "secure", X, 1, 2, 8, 134, 140, { tone: "hi" });
  chip("soc", "pmic", X, 1, 12, 22, 92, 100, { tone: "mid" });
  chip("soc", "crystal", X, 1.4, 8, 12, 101, 104, { tone: "mid" }, { pin: false });
  block("soc", "fpc1.connector", [X, CON1.y0, CON1.z0], [X + 2, CON1.y1, CON1.z1], { tone: "mid" }, { r: 0.3 });
  block("soc", "fpc2.connector", [X, CON2.y0, CON2.z0], [X + 2, CON2.y1, CON2.z1], { tone: "mid" }, { r: 0.3 });
  for (const [y, z] of [[0, 128], [-2, 145], [30, 104], [30, 120], [-30, 103], [30, 136], [-14, 128]]) smd("soc", X, y, z);
  // Traces: NPU to SoC, SoC to DDR, SoC to eMMC.
  const traces =
    trace(X, [[NPU.y0 - 0.8, 113], [SOCCHIP.y1 + 0.8, 113]]) +
    trace(X, [[NPU.y0 - 0.8, 117], [SOCCHIP.y1 + 0.8, 117]]) +
    trace(X, [[-14, SOCCHIP.z1 + 0.8], [-14, 131.2]]) +
    trace(X, [[19, 131.2], [19, NPU.z1 + 0.8]]) +
    trace(X, [[0, CON1.z1 + 0.8], [0, 104], [-6, 104]]);
  put("soc", [X + 0.01, 0, 118], traces, { name: "soc.traces", bias: -0.2, shapes: [R.solid(A.box([X, -32, 92], [X + 0.05, 32, 146], { name: "soc.traces" }))] });
}
// FPC from the sensor board to the main board: owned by the sensor board, latched against the SoC connector.
flex("sensor", "fpc1", { x0: SOC.x1 + 2, x1: 38.6, y0: CON1.y0, y1: CON1.y1, riserA: [SOC.x1 + 2, SOC.x1 + 2.6, CON1.z1], riserB: [38, 38.6, 94] });

// ---------- I/O board: PoE magnetics, PHY, DC-DC, microSD, reset, bulk capacitor ----------
const IO = { x0: -58, x1: -56.4, y0: -30, y1: 30, z0: 88, z1: 140 };
const PHY = { y0: -10, y1: 0, z0: 110, z1: 118 };
{
  board("io", "pcb", IO.x0, IO.x1, IO.y0, IO.y1, IO.z0, IO.z1);
  for (const [y0, y1] of [[8, 16], [20, 28]]) {
    block("io", "bracket.foot", [-68, y0, FLOOR], [IO.x0, y1, FLOOR + 2], { tone: "mid" }, { r: 0.4, extra: Dots([at([-65, (y0 + y1) / 2, FLOOR + 2])], { size: 0.5 }) });
    block("io", "bracket.tab", [IO.x0 - 2, y0, FLOOR + 2], [IO.x0, y1, 100], { tone: "mid" }, { r: 0.4 });
  }
  const X = IO.x1;
  block("io", "magnetics", [X, 6, 104], [X + 8, 20, 118], { tone: "hi" }, { r: 0.8, bevel: 0.4, extra: Ln(rectX(X + 8, 8.5, 106.5, 17.5, 115.5), { tone: "faint" }) + Dots([at([X + 8, 18, 116])], { size: 0.45 }) });
  chip("io", "phy", X, 1.2, PHY.y0, PHY.y1, PHY.z0, PHY.z1, { tone: "hi" });
  block("io", "inductor", [X, 8, 124], [X + 5, 16, 132], { tone: "mid" }, { r: 1.2, extra: Ln(ringX(X + 5, 12, 128, 2.4), { tone: "lo" }) });
  chip("io", "dcdc", X, 1, 18, 24, 126, 131, { tone: "mid" });
  block("io", "microsd", [X, -28, 122], [X + 1.8, -14, 136], { tone: "mid" }, { r: 0.4, extra: Ln(rectX(X + 1.8, -26.5, 123.5, -15.5, 134.5), { tone: "faint" }) });
  block("io", "card", [X + 0.4, -26, 136], [X + 1.2, -16, 141], { tone: "hi" }, { r: 0.3, steps: 2 });
  block("io", "reset", [X, 20, 132], [X + 2, 26, 136], { tone: "mid" }, { r: 0.4 });
  disc("io", "reset.plunger", AX(X + 2, 23, 134), 0, 1.6, 1.2, { tone: "hi" });
  disc("io", "capacitor", AX(X, 25, 106), 0, 9, 3.4, { tone: "hi" }, { extra: Ln(G.arcOf(AX(X, 25, 106), 7.6, 3.4, P), { tone: "faint" }) + Ln(G.circleOf(AX(X, 25, 106), 9, 2.2, P), { tone: "lo" }) });
  block("io", "fpc2.connector", [X, CON2.y0, CON2.z0], [X + 2, CON2.y1, CON2.z1], { tone: "mid" }, { r: 0.3 });
  block("io", "header", [IO.x0 - 3, 14, 126], [IO.x0, 26, 132], { tone: "mid" }, { r: 0.3 });
  for (const [y, z] of [[2, 104], [-4, 122], [4, 122], [-20, 104], [-14, 112], [26, 100], [26, 116]]) smd("io", X, y, z);
  const traces = trace(X, [[PHY.y1 + 0.8, 114], [5.2, 114]]) + trace(X, [[-20, CON2.z1 + 0.8], [-20, 108], [PHY.y0 - 0.8, 112]]) + trace(X, [[-5, PHY.z1 + 0.8], [-5, 120], [-13.2, 124]]);
  put("io", [X + 0.01, 0, 114], traces, { name: "io.traces", bias: -0.2, shapes: [R.solid(A.box([X, -28, 96], [X + 0.05, 28, 138], { name: "io.traces" }))] });
}
// FPC from the main board to the I/O board: owned by the main board, latched against the I/O connector.
flex("soc", "fpc2", { x0: IO.x1 + 2, x1: SOC.x1 + 2.6, y0: CON2.y0, y1: CON2.y1, riserA: [IO.x1 + 2, IO.x1 + 2.6, CON2.z1], riserB: [SOC.x1 + 2, SOC.x1 + 2.6, CON2.z1] });

// ---------- Rear cap, cable gland, lead to the I/O board, PoE pigtail to an RJ45 coupler on the bench ----------
function cable(piece, name, route, r, { tone = "hi", maxLength = 24, free = [false, false] } = {}) {
  const id = nameFor(`${piece}.${name}`);
  R.route(id, route, r, { piece, move: PIECES[piece].move, ride: piece, free, limp: true });
  tubePieces(route, r, P, { maxLength, spacing: Math.max(0.6, r * 0.3) }).forEach((part, index) =>
    put(piece, part.mid, tubeSvg(part, { tone }), { name: `${id}:${index}`, extra: { route: id, chunk: index }, shapes: [A.tube(part.points, part.radii)] }),
  );
}
{
  const X0 = -103;
  const X1 = -98;
  const corners = [[-38, 88], [38, 88], [38, 148], [-38, 148]];
  block("rear", "cap", [X0, -SHELL, CH_Z], [X1, SHELL, COVER_TOP], { tone: "mid" }, { r: 1, bevel: 0.6 });
  const GL = AX(X0, 18, 128, -1);
  turned("rear", "gland", GL, [[0, 7.2], [4, 7.2], [4, 6], [10, 6], [10, 4.6], [13, 4.6]], { tone: "hi" }, { extra: Ln(G.arcOf(GL, 2, 7.2, P), { tone: "faint" }) + k.fadedSvg(G.ribsOf(GL, 4, 10, 6, 6, P, { fade: [0.1, 0.55], seams: true }), { tone: "lo" }) });
  const PIG = 3;
  const PIG_Z = 4.6;
  const route = G.fillet([[X0 - 13, 18, 128], [-140, 18, 128], [-140, 18, PIG_Z], [-158, 18, PIG_Z]], 9, 12, { P, tube: PIG });
  cable("rear", "pigtail", route, PIG, { tone: "hi" });
  const RJ = planOf(-186, 8, -162, 28, 2);
  block("rear", "rj45.boot", [-162, 14, 0.6], [-158, 22, 8.6], { tone: "mid" }, { r: 1.2 });
  block("rear", "rj45", [RJ.x, RJ.y, 0], [RJ.x + RJ.w, RJ.y + RJ.d, 14], { tone: "hi" }, {
    r: 2,
    bevel: 0.6,
    extra: Ln(k.planOutline(planOf(-182, 13, -168, 23, 1.2), 14, P, 3), { tone: "lo" }) + Ln(onY(28, [[-183, 3], [-165, 3], [-165, 11], [-183, 11]]), { tone: "faint" }),
  });
  cable("rear", "lead", [[X1, 20, 129], [IO.x0 - 6, 20, 129]], 1.3, { tone: "mid", free: [false, true] });
  block("rear", "lead.shell", [IO.x0 - 6, 16, 126.5], [IO.x0 - 3, 24, 131.5], { tone: "mid" }, { r: 0.4 });
}

// ---------- Frame tile lanes: reserved slots in the settled order, one per surface the tile crosses ----------
const LANES = {
  fpc1: { piece: "sensor", box: [[SOC.x1 + 2.6, CON1.y0, FLOOR + 0.6], [38, CON1.y1, FLOOR + 0.8]] },
  riser1: { piece: "sensor", box: [[SOC.x1 + 2.6, CON1.y0, FLOOR + 0.6], [SOC.x1 + 2.8, CON1.y1, CON1.z1]] },
  soc: { piece: "soc", box: [[SOC.x1, SOC.y0 + 2, SOC.z0 + 2], [SOC.x1 + 0.2, SOC.y1 - 2, SOC.z1 - 2]] },
  riser2: { piece: "soc", box: [[SOC.x1 + 2.6, CON2.y0, FLOOR + 0.6], [SOC.x1 + 2.8, CON2.y1, CON2.z1]] },
  fpc2: { piece: "soc", box: [[IO.x1 + 2.6, CON2.y0, FLOOR + 0.6], [SOC.x1 + 2, CON2.y1, FLOOR + 0.8]] },
  riser3: { piece: "soc", box: [[IO.x1 + 2.6, CON2.y0, FLOOR + 0.6], [IO.x1 + 2.8, CON2.y1, CON2.z1]] },
  io: { piece: "io", box: [[IO.x1, IO.y0 + 2, IO.z0 + 2], [IO.x1 + 0.2, IO.y1 - 2, IO.z1 - 2]] },
};
for (const [id, lane] of Object.entries(LANES)) {
  const [min, max] = lane.box;
  put(lane.piece, G.mul3(G.add3(min, max), 0.5), `<g class="cc-lane" data-lane="${id}"></g>`, { name: `${lane.piece}.lane-${id}`, bias: 0.05, shapes: [R.solid(A.box(min, max, { name: `${lane.piece}.lane-${id}` }))] });
}

// The frame's route: each leg lies on one surface; hidden legs are inside a part.
const XF = SOC.x1 + 2.6;
const PATH = [
  { lane: "fpc1", plane: "z", from: [33, 0, FLOOR + 0.6], to: [XF + 6.2, 0, FLOOR + 0.6], t: 1.5, stage: "raw" },
  { lane: "riser1", plane: "x", from: [XF, 0, FLOOR + 6.4], to: [XF, 0, CON1.z1 - 6], t: 0.7, stage: "raw" },
  { hidden: "con1", t: 0.3, stage: "raw" },
  { lane: "soc", plane: "x", from: [SOC.x1, 0, CON1.z1 + 5.4], to: [SOC.x1, 0, 115], t: 0.5, stage: "raw" },
  { lane: "soc", plane: "x", from: [SOC.x1, 0, 115], to: [SOC.x1, NPU.y0 + 2, 115], t: 0.5, stage: "raw" },
  { hidden: "npu", t: 1.6, stage: "npu" },
  { lane: "soc", plane: "x", from: [SOC.x1, NPU.y0 + 2, 115], to: [SOC.x1, SOCCHIP.y1 - 2, 115], t: 0.9, stage: "masked" },
  { hidden: "soc", t: 0.9, stage: "encoder" },
  { lane: "soc", plane: "x", from: [SOC.x1, -20, SOCCHIP.z0 - 0.4], to: [SOC.x1, -20, CON2.z1 + 4.6], t: 0.5, stage: "encoder" },
  { hidden: "con2", t: 0.3, stage: "out" },
  { lane: "riser2", plane: "x", from: [XF, -20, CON2.z1 - 6], to: [XF, -20, FLOOR + 6.4], t: 0.6, stage: "out" },
  { lane: "fpc2", plane: "z", from: [XF - 6.2, -20, FLOOR + 0.6], to: [IO.x1 + 2.6 + 6.2, -20, FLOOR + 0.6], t: 1.6, stage: "out" },
  { lane: "riser3", plane: "x", from: [IO.x1 + 2.6, -20, FLOOR + 6.4], to: [IO.x1 + 2.6, -20, CON2.z1 - 6], t: 0.6, stage: "out" },
  { hidden: "con3", t: 0.3, stage: "out" },
  { lane: "io", plane: "x", from: [IO.x1, -20, CON2.z1 + 5.4], to: [IO.x1, -20, 114], t: 0.5, stage: "out" },
  { lane: "io", plane: "x", from: [IO.x1, -20, 114], to: [IO.x1, PHY.y0 + 2, 114], t: 0.4, stage: "out" },
  { hidden: "phy", t: 1.4, stage: "sent" },
  { hidden: "rest", t: 0.8, stage: "sent" },
];

// ---------- Settle the painter's order from the recorded shapes ----------
A.settle(R, P);
R.items.sort((a, b) => a.key - b.key);
const groups = R.items.map((item, index) => {
  const [mx, my, mz] = item.move;
  const tag = item.name === "soc.npu" ? ' data-npu=""' : item.name.startsWith("io.card") ? ' data-card=""' : "";
  return `<g class="it" data-piece="${item.piece}" data-k="${item.key.toFixed(2)}" data-m="${mx.toFixed(1)},${my.toFixed(1)},${mz.toFixed(1)}" data-i="${index}"${tag}>${item.svg}</g>`;
});

const LABEL_TEXT =
  "A box-style AI CCTV camera on a wall-mount bracket clamped to a bench, with its cover lifted: inside, front to back, a window, an IR LED ring, an M12 lens, the image sensor board, the main board with its SoC under a heatsink and a separate NPU, and the I/O board with PoE magnetics and a microSD slot. Flat flex cables join the boards; a pigtail runs from the rear cap to an RJ45 coupler on the bench.";
const svg = `<svg xmlns="http://www.w3.org/2000/svg" class="iso-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${LABEL_TEXT}"><g class="cc-back">${back.join("")}</g><g class="cc-parts">${groups.join("")}</g></svg>`;

const unit = (vector) => {
  const o = at([0, 0, 0]);
  const p = at(vector);
  return [p[0] - o[0], p[1] - o[1]];
};
const DATA = {
  P,
  V,
  EX: unit([1, 0, 0]),
  EY: unit([0, 1, 0]),
  EZ: unit([0, 0, 1]),
  pieces: Object.fromEntries(Object.entries(PIECES).map(([id, piece]) => [id, { name: piece.name, say: piece.say }])),
  stagger: STAGGER,
  spring: SPRING,
  order: ORDER,
  path: PATH,
  masked: MASKED,
  model: MODEL,
};

const LIVE = `
const D = ${JSON.stringify(DATA)};
const P = D.P;
const root = document.querySelector(".cc");
const stage = root.querySelector(".cc-stage");
const partsGroup = stage.querySelector(".cc-parts");
const readout = root.querySelector("[data-readout]");
const live = root.querySelector("[data-live]");
const partButtons = [...root.querySelectorAll(".cc-part")];
const still = matchMedia("(prefers-reduced-motion: reduce)");
const nodes = [...partsGroup.querySelectorAll(".it")].map((el) => ({ el, piece: el.dataset.piece, key: Number(el.dataset.k), m: el.dataset.m.split(",").map(Number), shown: "" }));
for (const node of nodes) node.lift = node.m[0] * D.V[0] + node.m[1] * D.V[1] + node.m[2] * D.V[2];
let order = nodes.map((_, index) => index);
const lanes = Object.fromEntries([...partsGroup.querySelectorAll("[data-lane]")].map((el) => [el.dataset.lane, el]));
const npuEl = partsGroup.querySelector("[data-npu]");
const coreEl = npuEl ? npuEl.querySelector("[data-core]") : null;
const cardEl = partsGroup.querySelector("[data-card]");
const IDS = Object.keys(D.pieces);
const piece = Object.fromEntries(IDS.map((id) => [id, { x: 0, v: 0, target: 0, pending: null }]));
const legs = [];
{ let t = 0; for (const leg of D.path) { legs.push({ ...leg, start: t }); t += leg.t; } }
const CYCLE = legs.reduce((sum, leg) => sum + leg.t, 0);

let clock = 0, mode = "closed", selected = null, inside = false, leaveTimer = 0;
let touring = !still.matches, tourClock = 0, idleTimer = 0, visible = false, frame = 0, last = 0;
let tileClock = 0, frameNo = 1, stageNow = "", busy = { npu: 0, card: 0 }, tileShown = "";

const pad = (n) => String(n).padStart(4, "0");
function say(text) { if (readout.textContent !== text) readout.textContent = text; }
function announce(text) { if (live.textContent !== text) live.textContent = text; }
function goalOf(id, next) {
  if (id === "stand") return 0;
  if (next === "closed") return 0;
  if (next === "open") return id === "cover" ? 1 : 0;
  return 1;
}
function setMode(next) {
  if (mode === next) return;
  const opening = (next === "apart") || (next === "open" && mode === "closed");
  mode = next;
  let index = 0;
  for (const id of opening ? D.stagger : [...D.stagger].reverse()) {
    const goal = goalOf(id, next);
    const state = piece[id];
    if (state.target === goal && !state.pending) continue;
    if (state.target === goal) { state.pending = null; continue; }
    state.pending = { value: goal, at: clock + (still.matches ? 0 : index++ * D.spring.stagger) };
  }
  stage.setAttribute("aria-valuetext", next === "apart" ? "Apart, 8 assemblies" : next === "open" ? "Cover lifted, assembled" : "Closed");
  announce(next === "apart" ? "taken apart" : next === "open" ? "cover lifted" : "closed");
  run();
}
function select(id) {
  if (id === "stand") id = null;
  if (selected === id) return;
  selected = id;
  for (const node of nodes) node.piece === id && id ? node.el.setAttribute("data-on", "") : node.el.removeAttribute("data-on");
  id ? stage.setAttribute("data-focus", "") : stage.removeAttribute("data-focus");
  partButtons.forEach((button) => (button.dataset.part === id ? button.setAttribute("data-on", "") : button.removeAttribute("data-on")));
}
function steer(state, dt, calm) {
  if (state.pending && clock >= state.pending.at) { state.target = state.pending.value; state.pending = null; }
  if (calm) { state.x = state.target; state.v = 0; return false; }
  for (let i = 0; i < D.spring.substeps; i++) {
    const h = dt / D.spring.substeps;
    state.v += ((state.target - state.x) * D.spring.stiffness - state.v * D.spring.damping) * h;
    state.x += state.v * h;
  }
  if (Math.abs(state.target - state.x) < 2e-4 && Math.abs(state.v) < 2e-4) { state.x = state.target; state.v = 0; return state.pending !== null; }
  return true;
}
function place() {
  for (const node of nodes) {
    const share = piece[node.piece].x;
    const tx = (D.EX[0] * node.m[0] + D.EY[0] * node.m[1] + D.EZ[0] * node.m[2]) * share;
    const ty = (D.EX[1] * node.m[0] + D.EY[1] * node.m[1] + D.EZ[1] * node.m[2]) * share;
    const text = share === 0 ? "" : "translate(" + tx.toFixed(2) + " " + ty.toFixed(2) + ")";
    if (text !== node.shown) { node.shown = text; text ? node.el.setAttribute("transform", text) : node.el.removeAttribute("transform"); }
  }
  const next = nodes.map((_, i) => i).sort((a, b) => nodes[a].key + nodes[a].lift * piece[nodes[a].piece].x - (nodes[b].key + nodes[b].lift * piece[nodes[b].piece].x));
  if (next.some((index, i) => index !== order[i])) {
    const fragment = document.createDocumentFragment();
    for (const index of next) fragment.appendChild(nodes[index].el);
    partsGroup.appendChild(fragment);
    order = next;
  }
}

// The frame tile, drawn in the plane of the surface it crosses.
const f1 = (n) => n.toFixed(1);
const poly = (points) => "M" + points.map((p) => { const s = iso(p, P); return f1(s[0]) + " " + f1(s[1]); }).join("L") + "Z";
const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
function tileSvg(c, plane, masked, appear) {
  const U = plane === "z" ? [1, 0, 0] : [0, 1, 0];
  const Vv = plane === "z" ? [0, 1, 0] : [0, 0, 1];
  const lift = plane === "z" ? [0, 0, 0.25] : [0.25, 0, 0];
  const h = 5.8 * appear;
  const pt = (u, v) => [c[0] + U[0] * u + Vv[0] * v + lift[0], c[1] + U[1] * u + Vv[1] * v + lift[1], c[2] + U[2] * u + Vv[2] * v + lift[2]];
  const cell = (i, j, inset) => { const s = (2 * h) / 3; const u0 = -h + i * s + inset, v0 = -h + j * s + inset, u1 = -h + (i + 1) * s - inset, v1 = -h + (j + 1) * s - inset; return poly([pt(u0, v0), pt(u1, v0), pt(u1, v1), pt(u0, v1)]); };
  const grid = [1, 2].map((i) => { const s = -h + (i * 2 * h) / 3; return "M" + iso(pt(s, -h), P).map(f1).join(" ") + "L" + iso(pt(s, h), P).map(f1).join(" ") + "M" + iso(pt(-h, s), P).map(f1).join(" ") + "L" + iso(pt(h, s), P).map(f1).join(" "); }).join("");
  const marks = D.masked.map(([i, j]) => cell(i, j, masked ? 0.25 : (2 * h) / 3 / 2 - 0.55)).join("");
  return '<path class="cc-tile" d="' + poly([pt(-h, -h), pt(h, -h), pt(h, h), pt(-h, h)]) + '"/><path class="cc-grid" d="' + grid + '"/><path class="' + (masked ? "cc-mask" : "cc-seen") + '" d="' + marks + '"/>';
}
function drawTile(t, show) {
  let lane = null, html = "", stageId = "rest";
  busy.npu = 0; busy.card = 0;
  if (show) {
    const leg = legs.find((l) => t >= l.start && t < l.start + l.t) || legs[legs.length - 1];
    stageId = leg.stage;
    const u = Math.min(1, Math.max(0, (t - leg.start) / leg.t));
    if (leg.hidden === "npu") busy.npu = Math.sin(Math.PI * u);
    if (leg.hidden === "phy") busy.card = Math.sin(Math.PI * u);
    if (!leg.hidden) {
      const masked = leg.stage !== "raw";
      const appear = Math.min(1, (t - legs[0].start) / 0.25);
      lane = leg.lane;
      html = tileSvg(lerp(leg.from, leg.to, u * u * (3 - 2 * u)), leg.plane, masked, appear);
    }
  }
  const key = (lane || "") + html;
  if (key !== tileShown) {
    tileShown = key;
    for (const [id, el] of Object.entries(lanes)) el.innerHTML = id === lane ? html : "";
  }
  if (coreEl) coreEl.style.opacity = (0.25 + 0.75 * busy.npu).toFixed(2);
  if (cardEl) busy.card > 0.2 ? cardEl.setAttribute("data-blink", "") : cardEl.removeAttribute("data-blink");
  stageNow = show ? stageId : "";
}
function stageText() {
  const f = "frame " + pad(frameNo) + " · ";
  if (stageNow === "raw") return f + "sensor → NPU · raw, never stored";
  if (stageNow === "npu") return f + "NPU · " + D.model.npuLine;
  if (stageNow === "masked") return f + "masked · " + D.model.masked + " regions · raw buffer cleared";
  if (stageNow === "encoder") return f + "H.265 encoder · sees masked pixels only";
  if (stageNow === "out") return f + "to PoE and microSD · masked only";
  if (stageNow === "sent") return f + "sent and stored · " + D.model.fps + " fps";
  return "";
}
function readoutText() {
  if (selected) return D.pieces[selected].say;
  const moving = IDS.some((id) => Math.abs(piece[id].x - piece[id].target) > 0.002 || piece[id].pending);
  if (moving) return mode === "apart" ? "taking apart" : mode === "open" ? (piece.cover.x < 0.98 ? "lifting the cover" : "closing up") : "closing up";
  if (mode === "apart") return "apart · 8 assemblies";
  if (mode === "open" && stageNow) return stageText();
  return "closed · masks on-chip · " + D.model.frameMs + " ms/frame · " + D.model.fps + " fps";
}

const TOUR = { closed: 1.6, lift: 1.4, step: 1.25 };
function tourAt(time) {
  const openAt = TOUR.closed;
  const apartAt = openAt + TOUR.lift + CYCLE;
  const listAt = apartAt + 1.1;
  const listEnd = listAt + D.order.length * TOUR.step;
  if (time < openAt) return { mode: "closed", part: null };
  if (time < apartAt) return { mode: "open", part: null };
  if (time < listAt) return { mode: "apart", part: null };
  if (time < listEnd) return { mode: "apart", part: D.order[Math.floor((time - listAt) / TOUR.step)] };
  if (time < listEnd + 1.6) return { mode: "open", part: null };
  return null;
}
function quietTour() {
  touring = false;
  clearTimeout(idleTimer);
  idleTimer = setTimeout(() => { if (inside) return quietTour(); touring = !still.matches; tourClock = 0; run(); }, 3600);
}
function tick(now) {
  const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
  last = now;
  clock += dt;
  const calm = still.matches;
  if (touring) {
    tourClock += dt;
    let step = tourAt(tourClock);
    if (!step) { tourClock = 0; step = tourAt(0); }
    setMode(step.mode);
    select(step.part);
  }
  let moving = false;
  for (const id of IDS) if (steer(piece[id], dt, calm)) moving = true;
  const ready = mode === "open" && piece.cover.x > 0.985 && IDS.every((id) => id === "cover" || piece[id].x < 0.002);
  if (ready) {
    tileClock += calm ? 0 : dt;
    if (tileClock >= CYCLE) { tileClock -= CYCLE; frameNo++; }
  } else if (mode !== "open") { tileClock = 0; }
  drawTile(calm && ready ? D.path.slice(0, 7).reduce((s, l) => s + l.t, 0) - 0.3 : tileClock, ready);
  place();
  say(readoutText());
  frame = visible && (moving || touring || ready) ? requestAnimationFrame(tick) : 0;
}
function run() { if (!frame && visible) { last = 0; frame = requestAnimationFrame(tick); } }
new IntersectionObserver((entries) => { visible = entries[entries.length - 1].isIntersecting; if (visible) run(); }, { rootMargin: "120px 0px" }).observe(stage);

function pieceAt(event) {
  const hit = document.elementFromPoint(event.clientX, event.clientY);
  const item = hit && hit.closest ? hit.closest(".it") : null;
  return item ? item.dataset.piece : null;
}
stage.addEventListener("pointermove", (event) => {
  if (event.pointerType === "touch") return;
  inside = true;
  clearTimeout(leaveTimer);
  quietTour();
  setMode("apart");
  select(pieceAt(event));
  run();
});
stage.addEventListener("pointerleave", () => {
  inside = false;
  clearTimeout(leaveTimer);
  leaveTimer = setTimeout(() => { if (inside) return; select(null); setMode("open"); }, 260);
});
stage.addEventListener("pointerdown", (event) => {
  if (event.pointerType !== "touch") return;
  quietTour();
  const id = pieceAt(event);
  if (!id || id === "stand") { select(null); setMode(mode === "apart" ? "open" : "apart"); }
  else { setMode("apart"); select(id); }
  run();
});
partButtons.forEach((button) => {
  const show = () => { quietTour(); clearTimeout(leaveTimer); setMode("apart"); select(button.dataset.part); run(); };
  const hide = () => {
    clearTimeout(leaveTimer);
    leaveTimer = setTimeout(() => { if (inside || partButtons.some((other) => other.matches(":hover, :focus-visible"))) return; select(null); setMode("open"); }, 260);
  };
  button.addEventListener("pointerenter", show);
  button.addEventListener("focus", show);
  button.addEventListener("pointerleave", hide);
  button.addEventListener("blur", hide);
  button.addEventListener("click", show);
});
stage.addEventListener("keydown", (event) => {
  const index = selected ? D.order.indexOf(selected) : -1;
  let next = null;
  if (event.key === "ArrowRight" || event.key === "ArrowDown") next = Math.min(D.order.length - 1, index + 1);
  else if (event.key === "ArrowLeft" || event.key === "ArrowUp") next = Math.max(0, index - 1);
  else if (event.key === "Home") next = 0;
  else if (event.key === "End") next = D.order.length - 1;
  else if (event.key === "Escape") { event.preventDefault(); quietTour(); select(null); setMode("closed"); return; }
  else if (event.key === "o" || event.key === "O" || event.key === "Enter") { event.preventDefault(); quietTour(); select(null); setMode(mode === "closed" ? "open" : "closed"); return; }
  else if (event.key === " ") { event.preventDefault(); quietTour(); select(null); setMode(mode === "apart" ? "open" : "apart"); return; }
  if (next === null) return;
  event.preventDefault();
  quietTour();
  setMode("apart");
  select(D.order[next]);
  stage.setAttribute("aria-valuenow", String(next + 1));
});
if (still.matches) touring = false;
const narrow = matchMedia("(hover: none), (max-width: 480px)");
const hint = root.querySelector("[data-corner='hint']");
const wording = () => hint && (hint.textContent = narrow.matches ? "Tap to take apart" : "Hover to take apart · O opens · Space");
wording();
narrow.addEventListener("change", wording);
if (still.matches) setMode("open");
place();
say(readoutText());
`;

const CSS = `
.cc-stage{position:relative;outline:none;touch-action:pan-y}
.cc-stage:focus-visible{outline:1px solid var(--anatomy-muted);outline-offset:6px}
.cc-stage[data-focus] .it:not([data-on]):not([data-piece="stand"]){--anatomy-hi:var(--anatomy-mid);--anatomy-mid:var(--anatomy-lo)}
.cc-stage[data-focus] .it[data-piece="stand"]{--anatomy-mid:var(--anatomy-lo)}
.it[data-on]{--anatomy-hi:var(--anatomy-lit);--anatomy-mid:var(--anatomy-hi)}
.it[data-on] .iso-top,.it[data-on] .tb-shine{fill:var(--anatomy-lit-top)}
.it[data-on] .iso-shade,.it[data-on] .tb-body{fill:var(--anatomy-lit-shade)}
.it[data-on] [data-seamless]>.iso-shade{stroke:var(--anatomy-lit-shade)}
.it[data-on] [data-seamless]>.iso-top{stroke:var(--anatomy-lit-top)}
.cc-tile{fill:var(--anatomy-face);stroke:var(--anatomy-hi);stroke-width:.8px;vector-effect:non-scaling-stroke;stroke-linejoin:round}
.cc-grid{fill:none;stroke:var(--anatomy-mid);stroke-width:.45px;vector-effect:non-scaling-stroke}
.cc-seen{fill:var(--anatomy-mid);stroke:none}
.cc-mask{fill:var(--anatomy-lit);stroke:none}
.cc-core{transition:none}
.cc-glass{fill:var(--anatomy-shade-1);stroke:none}
.it[data-card][data-blink] .iso-edge{stroke:var(--anatomy-lit)}
.it[data-card][data-blink] .iso-top{fill:var(--anatomy-lit-top)}
.cc-parts-list{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px}
.cc-part{display:inline-flex;align-items:baseline;gap:7px;padding:5px 10px;border:0;border-radius:999px;background:transparent;color:var(--anatomy-muted);font:12.5px/1.3 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;cursor:pointer;box-shadow:inset 0 0 0 1px var(--cc-ring);transition:color 160ms var(--anatomy-ease),background 160ms var(--anatomy-ease)}
.cc-part:hover,.cc-part:focus-visible,.cc-part[data-on]{color:var(--anatomy-strong);background:var(--cc-hover);outline:none}
.cc-num{font-family:var(--anatomy-mono);font-size:10px;color:var(--anatomy-muted)}
.cc{--cc-ring:#ffffff14;--cc-hover:#ffffff0a}
.iso[data-theme="light"] .cc{--cc-ring:#00000017;--cc-hover:#0000000a}
.cc-sr{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%);white-space:nowrap}
@media (max-width:640px){.cc .iso-plate{padding:48px 8px 70px}.cc .iso-plate-corner[data-corner="readout"]{bottom:18px;max-width:100%;overflow:hidden;text-overflow:ellipsis}.cc .iso-plate-corner[data-corner="hint"]{bottom:0}}
`;

const PARTS = ORDER.map((id, index) => `<button type="button" class="cc-part" data-part="${id}"><span class="cc-num">${String(index + 1).padStart(2, "0")}</span>${PIECES[id].name}</button>`).join("");
const stageHtml = `<div class="cc-stage" tabindex="0" role="slider" aria-label="AI CCTV camera. Press O to lift the cover and watch a frame travel from the sensor through the NPU, which masks it, to the encoder and the PoE and microSD outputs. Hover, Space or the arrow keys take it apart; Escape closes it." aria-valuemin="0" aria-valuemax="${ORDER.length}" aria-valuenow="0" aria-valuetext="Closed">${svg}<span class="cc-sr" role="status" aria-live="polite" data-live></span></div>`;
const CAPTION =
  `In this design the image sensor has one way out: a single flat flex cable that ends on the main board. On that board every frame goes into the NPU first, where pose estimation, a body-part detector, hand tracking and a phone/POS detector find the private regions and paint them over (${MEASURED ? "measured on the laptop prototype's CPU with the detector branches in parallel, ms per frame: " : ""}${MODEL.npuLine}, about ${MODEL.fps} frames a second${MEASURED ? "; a camera NPU runs the same networks faster" : ""}); only then does the SoC's H.265 encoder see it. The microSD slot and the PoE jack hang off the I/O board, downstream of both, so the camera has no path that can store or send an unmasked frame. The tile on the cables is one frame; the black cells are the masked regions.`;
const plate = k
  .plateHtml({
    fig: "Fig 1",
    title: "Privacy-by-design AI camera",
    hint: "Hover to take apart · O opens · Space",
    readout: "",
    keys: [
      { mark: "lit", label: "The NPU: every frame is masked here" },
      { mark: "raised", label: "Boards, chips and the lens" },
      { mark: "edge", label: "Flat flex cables and the PoE pigtail" },
    ],
    caption: CAPTION,
    body: stageHtml,
  })
  .replace('<figure class="iso-figure">', '<figure class="iso-figure cc">')
  .replace('<figcaption class="iso-legend">', `<figcaption class="iso-legend"><span class="cc-parts-list">${PARTS}</span>`);
const body = `<style>${CSS}</style>${plate}`;
const page = k.pageHtml({ title: "Privacy-by-design AI camera", theme: THEME, body, script: kitScript() + LIVE, width: 1040 });
const name = THEME === "light" ? "cctv-camera-light.html" : "cctv-camera.html";
writeFileSync(join(HERE, name), page);
console.log("wrote", name, `${(page.length / 1024).toFixed(0)} KB`, `${R.items.length} items`, `${R.solids.length} solids`);

A.auditOrExit(R, P);
