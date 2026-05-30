"use client";

import * as React from "react";
import { useFrame } from "@react-three/fiber";
import { Html, Text } from "@react-three/drei";
import { Play, RotateCcw } from "lucide-react";
import * as THREE from "three";

import { Button } from "@/components/ui/button";
import { SceneCanvas } from "@/components/viz/scene-canvas";

// EfficientNet-B0 — features[0..8] plus the classifier head we attached.
// Channel counts taken from the torchvision EfficientNet config; widths
// are reflected approximately by block size in the scene.
const BLOCKS = [
  { name: "features[0]", desc: "Conv 3→32",         channels: 32  },
  { name: "features[1]", desc: "MBConv 16",         channels: 16  },
  { name: "features[2]", desc: "MBConv 24",         channels: 24  },
  { name: "features[3]", desc: "MBConv 40",         channels: 40  },
  { name: "features[4]", desc: "MBConv 80",         channels: 80  },
  { name: "features[5]", desc: "MBConv 112",        channels: 112 },
  { name: "features[6]", desc: "MBConv 192",        channels: 192 },
  { name: "features[7]", desc: "MBConv 320",        channels: 320 },
  { name: "features[8]", desc: "Conv 320→1280",     channels: 1280 },
  { name: "classifier",  desc: "Dropout → Linear(1280, 39)", channels: 39 },
] as const;

const BLOCK_SPACING = 1.6;
const ANIMATION_DURATION = 6.0; // seconds

interface BlockProps {
  index: number;
  name: string;
  desc: string;
  channels: number;
  isActive: boolean;
  hovered: boolean;
  onHover: (i: number | null) => void;
}

function Block({ index, name, desc, channels, isActive, hovered, onHover }: BlockProps) {
  const x = (index - (BLOCKS.length - 1) / 2) * BLOCK_SPACING;
  // Block size scales with sqrt(channels) so big channel counts read as bigger blocks
  // without dwarfing the early shallow ones.
  const scale = 0.45 + Math.sqrt(channels) / 12;
  const color = isActive ? "#fb923c" : hovered ? "#60a5fa" : "#475569";
  const emissive = isActive ? "#f97316" : "#000";
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
          emissiveIntensity={isActive ? 0.6 : 0}
          roughness={0.45}
          metalness={0.15}
        />
      </mesh>
      <Text
        position={[x, -scale / 2 - 0.35, 0]}
        fontSize={0.16}
        color="#888"
        anchorX="center"
        anchorY="top"
      >
        {name}
      </Text>
      {hovered && (
        <Html
          position={[x, scale / 2 + 0.5, 0]}
          center
          distanceFactor={10}
        >
          <div className="pointer-events-none whitespace-nowrap rounded-md border bg-popover px-3 py-2 text-xs text-popover-foreground shadow-lg">
            <div className="font-medium">{name}</div>
            <div className="text-muted-foreground">{desc}</div>
            <div className="tabular-nums text-muted-foreground">channels: {channels}</div>
          </div>
        </Html>
      )}
    </group>
  );
}

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
}: {
  progress: number;
  particleVisible: boolean;
  hovered: number | null;
  setHovered: (i: number | null) => void;
}) {
  // Active block: the one the particle is currently passing through
  const activeIndex = particleVisible
    ? Math.min(
        BLOCKS.length - 1,
        Math.floor(progress * BLOCKS.length),
      )
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

export function Architecture3D() {
  const [playing, setPlaying] = React.useState(false);
  const [progress, setProgress] = React.useState(0);
  const [particleVisible, setParticleVisible] = React.useState(false);
  const [hovered, setHovered] = React.useState<number | null>(null);

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
    // Leave the particle visible at the end so the audience sees it landed
    setTimeout(() => setParticleVisible(false), 1200);
  }

  return (
    <div className="space-y-3">
      <SceneCanvas
        cameraPosition={[0, 4, 10]}
        caption="Hover a block for details · Play Inference to animate"
        defaultAutoRotate={false}
        height={420}
      >
        <ArchitectureScene
          progress={progress}
          particleVisible={particleVisible}
          hovered={hovered}
          setHovered={setHovered}
        />
        <AnimationDriver
          playing={playing}
          onComplete={onComplete}
          onProgress={setProgress}
        />
      </SceneCanvas>

      <div className="flex flex-wrap items-center gap-2">
        <Button onClick={startAnimation} disabled={playing}>
          <Play className="mr-2 h-4 w-4" />
          {playing ? "Playing…" : "Play inference"}
        </Button>
        <Button variant="outline" onClick={resetAnimation} disabled={!playing && progress === 0}>
          <RotateCcw className="mr-2 h-4 w-4" />
          Reset
        </Button>
        <p className="ml-auto text-xs text-muted-foreground">
          A leaf image becomes a 3×224×224 tensor, flows through {BLOCKS.length - 1}{" "}
          backbone blocks, and lands at the classifier as 39 logits.
        </p>
      </div>
    </div>
  );
}
