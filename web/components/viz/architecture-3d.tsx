"use client";

import * as React from "react";
import { useFrame } from "@react-three/fiber";
import { Html, Text } from "@react-three/drei";
import { Flame, Play, RotateCcw, Snowflake, Sparkles } from "lucide-react";
import * as THREE from "three";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { SceneCanvas } from "@/components/viz/scene-canvas";
import { cn } from "@/lib/utils";

// ──────────────────────────────────────────────────────────────────
// EfficientNet-B0 blocks (10 cells: features[0..8] + classifier)
// ──────────────────────────────────────────────────────────────────

const BLOCKS = [
  { name: "features[0]", desc: "Conv 3→32", channels: 32 },
  { name: "features[1]", desc: "MBConv 16", channels: 16 },
  { name: "features[2]", desc: "MBConv 24", channels: 24 },
  { name: "features[3]", desc: "MBConv 40", channels: 40 },
  { name: "features[4]", desc: "MBConv 80", channels: 80 },
  { name: "features[5]", desc: "MBConv 112", channels: 112 },
  { name: "features[6]", desc: "MBConv 192", channels: 192 },
  { name: "features[7]", desc: "MBConv 320", channels: 320 },
  { name: "features[8]", desc: "Conv 320→1280", channels: 1280 },
  { name: "classifier", desc: "Dropout → Linear(1280, 39)", channels: 39 },
] as const;

const BLOCK_SPACING = 1.6;
const ANIMATION_DURATION = 6.0;

// ──────────────────────────────────────────────────────────────────
// Phase configurations — the four chapters of the training story
// ──────────────────────────────────────────────────────────────────

type PhaseConfig = {
  id: string;
  label: string;
  shortLabel: string;
  title: string;
  description: string;
  epochs: number;
  /** Per-block trainable state. 10 entries: features[0..8] + classifier. */
  unfrozen: boolean[];
  headLR: string;
  backboneLR: string;
  weightDecay: string;
  augmentation: string;
  augmentationDetail: string;
  valAccLabel: string;
  valAccValue: string;
  registeredAs?: string;
  isV4?: boolean;
  unfreezeCall: string;
  reasoning: string;
};

const PHASES: PhaseConfig[] = [
  {
    id: "phase1",
    label: "Phase 1 · Head warm-up",
    shortLabel: "Phase 1",
    title: "Head warm-up — backbone fully frozen",
    description:
      "Phase 1 trains only the new 39-class classifier head. The entire EfficientNet-B0 backbone is frozen with requires_grad=False; BatchNorm running statistics are locked in eval mode so they don't drift while the head is random.",
    epochs: 3,
    unfrozen: [false, false, false, false, false, false, false, false, false, true],
    headLR: "1e-3",
    backboneLR: "frozen",
    weightDecay: "1e-4 (head only)",
    augmentation: "train_tfms",
    augmentationDetail:
      "RandomResizedCrop(0.7, 1.0) · HorizontalFlip · VerticalFlip · Rotation(15°) · ColorJitter(0.2)",
    valAccLabel: "Val accuracy by end of Phase 1",
    valAccValue: "~93%",
    unfreezeCall: "freeze_all_backbone(model)",
    reasoning:
      "The randomly-initialized head would produce huge initial gradients that, with full unfreezing, would back-propagate through the network and trash the pretrained ImageNet features (catastrophic forgetting). Phase 1 protects the backbone absolutely while the head settles into a sane region of parameter space.",
  },
  {
    id: "phase2",
    label: "Phase 2 · Deepest blocks unfreeze",
    shortLabel: "Phase 2",
    title: "Deepest blocks unfreeze — features[6..8] + classifier",
    description:
      "Phase 2a in the original schedule. The three deepest feature blocks become trainable. These layers learned the most ImageNet-specific representations and need the most adaptation to PlantVillage; everything earlier stays frozen.",
    epochs: 4,
    unfrozen: [false, false, false, false, false, false, true, true, true, true],
    headLR: "1e-3",
    backboneLR: "1e-4 (10× lower)",
    weightDecay: "1e-4 (matrix weights only)",
    augmentation: "train_tfms",
    augmentationDetail: "Same as Phase 1 — standard PlantVillage augmentation",
    valAccLabel: "Val accuracy by end of Phase 2",
    valAccValue: "~99%",
    unfreezeCall: "unfreeze_from_block(model, from_idx=6)",
    reasoning:
      "Discriminative learning rates: the head trains 10× faster than the backbone. Unfreezing deepest-first because the deepest layers are the most ImageNet-specific — they need the largest change to specialize on leaf diseases. Earlier layers learn universal features (edges, color) that need much less adjustment.",
  },
  {
    id: "phase3",
    label: "Phase 3 · Full backbone unfreezes",
    shortLabel: "Phase 3",
    title: "Full backbone unfreezes — features[0..8] + classifier",
    description:
      "All feature blocks become trainable. The entire network fine-tunes together with the 10× LR ratio still protecting the backbone. Phase 3 runs until early stopping fires. This is where v3 reaches its registered val accuracy.",
    epochs: 17,
    unfrozen: [true, true, true, true, true, true, true, true, true, true],
    headLR: "1e-3",
    backboneLR: "1e-4 (10× lower)",
    weightDecay: "1e-4 (matrix weights only)",
    augmentation: "train_tfms",
    augmentationDetail: "Same — standard PlantVillage augmentation",
    valAccLabel: "Best val accuracy reached",
    valAccValue: "99.73%",
    registeredAs: "GreenVision v3 → Production",
    unfreezeCall: "unfreeze_from_block(model, from_idx=0)",
    reasoning:
      "By Phase 3 the head and deepest blocks have adapted. Unfreezing the universal-feature layers lets the whole network co-adapt at a slow, controlled rate. The 10× LR ratio means the backbone makes small adjustments while the head can still respond quickly. ReduceLROnPlateau halves the LR if val loss stalls; early stopping fires after 5 epochs without improvement.",
  },
  {
    id: "tuning",
    label: "Tuning · v4 OOD fix",
    shortLabel: "Tuning (v4)",
    title: "OOD-robust fine-tune — Decision 15",
    description:
      "v4 loaded v3 from the MLflow Registry and continued training with a much heavier augmentation pipeline. Goal: force the model away from background reliance so it generalizes to outside-internet field photos.",
    epochs: 12,
    unfrozen: [true, true, true, true, true, true, true, true, true, true],
    headLR: "5e-4 (conservative)",
    backboneLR: "5e-5 (10× ratio preserved)",
    weightDecay: "1e-4 (matrix weights only — same)",
    augmentation: "train_tfms_robust",
    augmentationDetail:
      "+ RandomGrayscale(0.10) + GaussianBlur(0.25) + RandomErasing(0.40, scale up to 25%) · wider crop · stronger color jitter",
    valAccLabel: "Best val accuracy reached",
    valAccValue: "99.78%",
    registeredAs: "GreenVision v4 → Production",
    isV4: true,
    unfreezeCall: "unfreeze_from_block(model, from_idx=0)  # all from epoch 0",
    reasoning:
      "v4 unfreezes everything from epoch 0 because v3's head is already trained — no catastrophic-forgetting risk. The 10× LR ratio still protects backbone features. The augmentation pipeline is the key change: RandomGrayscale forces texture learning when color is removed, RandomErasing destroys image patches (including background regions), and stronger color jitter simulates outdoor lighting. The model learns leaf shape and texture rather than the implicit 'leaves on neutral backgrounds' cue.",
  },
];

// ──────────────────────────────────────────────────────────────────
// 3D block — color-aware of phase unfreeze state
// ──────────────────────────────────────────────────────────────────

interface BlockProps {
  index: number;
  name: string;
  desc: string;
  channels: number;
  isActive: boolean;
  hovered: boolean;
  isUnfrozen: boolean;
  isHead: boolean;
  isV4: boolean;
  onHover: (i: number | null) => void;
}

function Block({
  index,
  name,
  desc,
  channels,
  isActive,
  hovered,
  isUnfrozen,
  isHead,
  isV4,
  onHover,
}: BlockProps) {
  const x = (index - (BLOCKS.length - 1) / 2) * BLOCK_SPACING;
  const scale = 0.45 + Math.sqrt(channels) / 12;

  // Color priority: active > hover > unfrozen > frozen
  let color: string;
  let emissive: string;
  let emissiveIntensity: number;

  if (isActive) {
    color = "#fb923c"; // orange — particle is in this block
    emissive = "#f97316";
    emissiveIntensity = 0.7;
  } else if (hovered) {
    color = "#60a5fa";
    emissive = "#000";
    emissiveIntensity = 0;
  } else if (isUnfrozen) {
    if (isHead) {
      color = "#10b981"; // emerald — classifier head
      emissive = "#10b981";
      emissiveIntensity = 0.18;
    } else if (isV4) {
      color = "#8b5cf6"; // violet — v4 trainable backbone block
      emissive = "#8b5cf6";
      emissiveIntensity = 0.12;
    } else {
      color = "#3b82f6"; // blue — v3 trainable backbone block
      emissive = "#3b82f6";
      emissiveIntensity = 0.12;
    }
  } else {
    color = "#475569"; // slate — frozen
    emissive = "#000";
    emissiveIntensity = 0;
  }

  return (
    <group>
      <mesh
        position={[x, 0, 0]}
        onPointerOver={(e) => {
          e.stopPropagation();
          onHover(index);
          document.body.style.cursor = "pointer";
        }}
        onPointerOut={() => {
          onHover(null);
          document.body.style.cursor = "auto";
        }}
        castShadow
        receiveShadow
      >
        <boxGeometry args={[scale * 0.7, scale, scale]} />
        <meshStandardMaterial
          color={color}
          emissive={emissive}
          emissiveIntensity={emissiveIntensity}
          roughness={0.45}
          metalness={0.15}
        />
      </mesh>
      <Text
        position={[x, -scale / 2 - 0.35, 0]}
        fontSize={0.16}
        color={isUnfrozen ? "#444" : "#888"}
        anchorX="center"
        anchorY="top"
      >
        {name}
      </Text>
      {hovered && (
        <Html position={[x, scale / 2 + 0.5, 0]} center distanceFactor={10}>
          <div className="pointer-events-none whitespace-nowrap rounded-md border bg-popover px-3 py-2 text-xs text-popover-foreground shadow-lg">
            <div className="font-medium">{name}</div>
            <div className="text-muted-foreground">{desc}</div>
            <div className="tabular-nums text-muted-foreground">
              channels: {channels}
            </div>
            <div className="mt-1 flex items-center gap-1 text-[10px]">
              {isUnfrozen ? (
                <>
                  <Flame className="h-3 w-3 text-blue-500" />
                  <span className="text-blue-700 dark:text-blue-400">
                    trainable in this phase
                  </span>
                </>
              ) : (
                <>
                  <Snowflake className="h-3 w-3 text-slate-500" />
                  <span className="text-slate-700 dark:text-slate-300">
                    frozen · requires_grad=False
                  </span>
                </>
              )}
            </div>
          </div>
        </Html>
      )}
    </group>
  );
}

// ──────────────────────────────────────────────────────────────────
// Animated data particle
// ──────────────────────────────────────────────────────────────────

function DataParticle({
  progress,
  visible,
}: {
  progress: number;
  visible: boolean;
}) {
  const meshRef = React.useRef<THREE.Mesh>(null);
  const xStart = -((BLOCKS.length - 1) / 2) * BLOCK_SPACING - 1;
  const xEnd = ((BLOCKS.length - 1) / 2) * BLOCK_SPACING + 1;
  const x = xStart + (xEnd - xStart) * progress;

  useFrame((_, delta) => {
    if (!meshRef.current) return;
    meshRef.current.rotation.x += delta * 2;
    meshRef.current.rotation.y += delta * 3;
  });

  if (!visible) return null;
  return (
    <mesh ref={meshRef} position={[x, 0, 0]}>
      <icosahedronGeometry args={[0.18, 1]} />
      <meshStandardMaterial
        color="#34d399"
        emissive="#10b981"
        emissiveIntensity={1.4}
        roughness={0.2}
      />
      <pointLight color="#34d399" intensity={2.5} distance={4} />
    </mesh>
  );
}

function ArchitectureScene({
  progress,
  particleVisible,
  hovered,
  setHovered,
  phase,
}: {
  progress: number;
  particleVisible: boolean;
  hovered: number | null;
  setHovered: (i: number | null) => void;
  phase: PhaseConfig;
}) {
  const activeIndex = particleVisible
    ? Math.min(BLOCKS.length - 1, Math.floor(progress * BLOCKS.length))
    : -1;

  return (
    <>
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, -1.5, 0]}
        receiveShadow
      >
        <planeGeometry args={[30, 8]} />
        <shadowMaterial opacity={0.2} />
      </mesh>

      {BLOCKS.map((block, i) => (
        <Block
          key={block.name}
          index={i}
          {...block}
          isActive={i === activeIndex}
          hovered={hovered === i}
          isUnfrozen={phase.unfrozen[i]}
          isHead={i === BLOCKS.length - 1}
          isV4={!!phase.isV4}
          onHover={setHovered}
        />
      ))}

      <DataParticle progress={progress} visible={particleVisible} />

      <Text
        position={[
          -((BLOCKS.length - 1) / 2) * BLOCK_SPACING - 1.4,
          0.05,
          0,
        ]}
        fontSize={0.22}
        color="#888"
        anchorX="right"
        anchorY="middle"
      >
        input · 3×224×224
      </Text>
      <Text
        position={[
          ((BLOCKS.length - 1) / 2) * BLOCK_SPACING + 1.3,
          0.05,
          0,
        ]}
        fontSize={0.22}
        color="#888"
        anchorX="left"
        anchorY="middle"
      >
        output · logits(39)
      </Text>
    </>
  );
}

function AnimationDriver({
  playing,
  onComplete,
  onProgress,
}: {
  playing: boolean;
  onComplete: () => void;
  onProgress: (p: number) => void;
}) {
  const start = React.useRef<number | null>(null);
  useFrame((state) => {
    if (!playing) {
      start.current = null;
      return;
    }
    if (start.current === null) start.current = state.clock.elapsedTime;
    const elapsed = state.clock.elapsedTime - start.current;
    const p = Math.min(elapsed / ANIMATION_DURATION, 1);
    onProgress(p);
    if (p >= 1) {
      onComplete();
      start.current = null;
    }
  });
  return null;
}

// ──────────────────────────────────────────────────────────────────
// Phase selector tabs
// ──────────────────────────────────────────────────────────────────

function PhaseSelector({
  selectedIndex,
  onSelect,
}: {
  selectedIndex: number;
  onSelect: (index: number) => void;
}) {
  return (
    <div className="flex flex-wrap gap-1 rounded-md border bg-muted/30 p-1">
      {PHASES.map((phase, i) => {
        const isActive = i === selectedIndex;
        const isV4 = phase.isV4;
        return (
          <button
            key={phase.id}
            onClick={() => onSelect(i)}
            className={cn(
              "flex-1 rounded px-3 py-2 text-xs font-medium transition-colors",
              "min-w-[120px]",
              isActive
                ? isV4
                  ? "bg-violet-500 text-white shadow-sm"
                  : "bg-blue-500 text-white shadow-sm"
                : "text-muted-foreground hover:bg-background hover:text-foreground",
            )}
          >
            <div className="text-[10px] uppercase tracking-wide opacity-80">
              {`step ${i + 1}`}
            </div>
            <div>{phase.shortLabel}</div>
          </button>
        );
      })}
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────
// Phase info panel — metric cards + reasoning
// ──────────────────────────────────────────────────────────────────

function PhaseInfoPanel({ phase }: { phase: PhaseConfig }) {
  return (
    <Card
      className={cn(
        "border-l-4",
        phase.isV4 ? "border-l-violet-500" : "border-l-blue-500",
      )}
    >
      <CardContent className="space-y-4 p-5">
        {/* Title and description */}
        <div>
          <div className="mb-1 flex items-center gap-2">
            <Badge
              variant="outline"
              className={cn(
                "text-[10px]",
                phase.isV4
                  ? "border-violet-500/40 text-violet-700 dark:text-violet-400"
                  : "border-blue-500/40 text-blue-700 dark:text-blue-400",
              )}
            >
              {phase.isV4 ? "Decision 15 · Tuning" : "Two-phase fine-tuning"}
            </Badge>
            {phase.registeredAs && (
              <Badge className="bg-emerald-500/15 text-[10px] text-emerald-700 dark:text-emerald-400">
                {phase.registeredAs}
              </Badge>
            )}
          </div>
          <h3 className="text-base font-semibold tracking-tight">
            {phase.title}
          </h3>
          <p className="mt-1 text-sm text-muted-foreground">
            {phase.description}
          </p>
        </div>

        <Separator />

        {/* Metric grid */}
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCell label="Epochs" value={String(phase.epochs)} />
          <MetricCell
            label={phase.valAccLabel}
            value={phase.valAccValue}
            emphasize
          />
          <MetricCell label="Head LR" value={phase.headLR} />
          <MetricCell label="Backbone LR" value={phase.backboneLR} />
        </div>

        {/* Augmentation + weight decay row */}
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="space-y-1 rounded-md border bg-muted/40 p-3">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              Train augmentation
            </p>
            <p className="font-mono text-xs">{phase.augmentation}</p>
            <p className="text-[11px] text-muted-foreground">
              {phase.augmentationDetail}
            </p>
          </div>
          <div className="space-y-1 rounded-md border bg-muted/40 p-3">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              Weight decay (AdamW)
            </p>
            <p className="font-mono text-xs">{phase.weightDecay}</p>
            <p className="text-[11px] text-muted-foreground">
              Bias + BatchNorm params: weight_decay=0
            </p>
          </div>
        </div>

        {/* What's trainable */}
        <div className="space-y-1 rounded-md border bg-muted/40 p-3">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            What&apos;s trainable
          </p>
          <code className="block font-mono text-xs text-foreground">
            {phase.unfreezeCall}
          </code>
          <div className="mt-2 flex flex-wrap gap-1">
            {BLOCKS.map((block, i) => (
              <span
                key={block.name}
                className={cn(
                  "inline-flex items-center gap-1 rounded border px-1.5 py-0.5 font-mono text-[10px]",
                  phase.unfrozen[i]
                    ? i === BLOCKS.length - 1
                      ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
                      : phase.isV4
                        ? "border-violet-500/40 bg-violet-500/10 text-violet-700 dark:text-violet-400"
                        : "border-blue-500/40 bg-blue-500/10 text-blue-700 dark:text-blue-400"
                    : "border-slate-500/30 bg-slate-500/10 text-slate-500",
                )}
              >
                {phase.unfrozen[i] ? (
                  <Flame className="h-2.5 w-2.5" />
                ) : (
                  <Snowflake className="h-2.5 w-2.5" />
                )}
                {block.name === "classifier" ? "head" : block.name}
              </span>
            ))}
          </div>
        </div>

        {/* Reasoning */}
        <div className="rounded-md border border-amber-500/20 bg-amber-500/5 p-3">
          <div className="mb-1 flex items-center gap-1.5">
            <Sparkles className="h-3.5 w-3.5 text-amber-600" />
            <p className="text-[10px] font-semibold uppercase tracking-wider text-amber-700 dark:text-amber-400">
              Why this phase
            </p>
          </div>
          <p className="text-xs">{phase.reasoning}</p>
        </div>
      </CardContent>
    </Card>
  );
}

function MetricCell({
  label,
  value,
  emphasize = false,
}: {
  label: string;
  value: string;
  emphasize?: boolean;
}) {
  return (
    <div className="rounded-md border bg-muted/40 p-3">
      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </p>
      <p
        className={cn(
          "mt-1 font-semibold tabular-nums",
          emphasize
            ? "text-lg text-emerald-700 dark:text-emerald-400"
            : "text-sm",
        )}
      >
        {value}
      </p>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────
// Public component — the four-phase training story
// ──────────────────────────────────────────────────────────────────

export function Architecture3D() {
  const [selectedPhase, setSelectedPhase] = React.useState(0);
  const [playing, setPlaying] = React.useState(false);
  const [progress, setProgress] = React.useState(0);
  const [particleVisible, setParticleVisible] = React.useState(false);
  const [hovered, setHovered] = React.useState<number | null>(null);

  const phase = PHASES[selectedPhase];

  // Reset animation whenever the phase changes
  React.useEffect(() => {
    setPlaying(false);
    setProgress(0);
    setParticleVisible(false);
  }, [selectedPhase]);

  function startAnimation() {
    setProgress(0);
    setParticleVisible(true);
    setPlaying(true);
  }

  function resetAnimation() {
    setPlaying(false);
    setProgress(0);
    setParticleVisible(false);
  }

  function onComplete() {
    setPlaying(false);
    setTimeout(() => setParticleVisible(false), 1200);
  }

  return (
    <div className="space-y-4">
      {/* Phase selector */}
      <PhaseSelector
        selectedIndex={selectedPhase}
        onSelect={setSelectedPhase}
      />

      {/* 3D scene */}
      <SceneCanvas
        cameraPosition={[0, 4, 10]}
        caption={`${phase.label} · hover a block · click Play to animate forward pass`}
        height={420}
      >
        <ArchitectureScene
          progress={progress}
          particleVisible={particleVisible}
          hovered={hovered}
          setHovered={setHovered}
          phase={phase}
        />
        <AnimationDriver
          playing={playing}
          onComplete={onComplete}
          onProgress={setProgress}
        />
      </SceneCanvas>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-2">
        <Button onClick={startAnimation} disabled={playing}>
          <Play className="mr-2 h-4 w-4" />
          {playing ? "Forward pass running…" : `Play forward pass · ${phase.shortLabel}`}
        </Button>
        <Button
          variant="outline"
          onClick={resetAnimation}
          disabled={!playing && progress === 0}
        >
          <RotateCcw className="mr-2 h-4 w-4" />
          Reset
        </Button>
        <p className="ml-auto text-xs text-muted-foreground">
          Forward pass works regardless of frozen state · only backprop
          stops at frozen blocks
        </p>
      </div>

      {/* Phase info panel */}
      <PhaseInfoPanel phase={phase} />
    </div>
  );
}
