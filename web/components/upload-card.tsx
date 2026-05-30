"use client";

import * as React from "react";
import {
  Image as ImageIcon,
  Loader2,
  Shuffle,
  Upload,
  X,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  sampleToFile,
  useTestSamples,
  type TestSample,
} from "@/lib/use-test-samples";
import { cn } from "@/lib/utils";

interface UploadCardProps {
  onPredict: (file: File) => void | Promise<void>;
  isPredicting: boolean;
  disabled: boolean;
  onInvalidFile?: (reason: string) => void;
  onPreviewURLChange?: (url: string | null) => void;
}

export function UploadCard({
  onPredict,
  isPredicting,
  disabled,
  onInvalidFile,
  onPreviewURLChange,
}: UploadCardProps) {
  const [file, setFile] = React.useState<File | null>(null);
  const [previewURL, setPreviewURL] = React.useState<string | null>(null);
  const [previewLabel, setPreviewLabel] = React.useState<string | null>(null);
  const [isDragging, setIsDragging] = React.useState(false);
  const inputRef = React.useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    return () => {
      if (previewURL && previewURL.startsWith("blob:")) {
        URL.revokeObjectURL(previewURL);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  React.useEffect(() => {
    onPreviewURLChange?.(previewURL);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewURL]);

  function setUploaded(f: File) {
    if (!f.type.startsWith("image/")) {
      onInvalidFile?.(`"${f.name}" is not an image file.`);
      return;
    }
    setFile(f);
    if (previewURL && previewURL.startsWith("blob:")) URL.revokeObjectURL(previewURL);
    const url = URL.createObjectURL(f);
    setPreviewURL(url);
    setPreviewLabel(null);
  }

  function handleClear() {
    if (previewURL && previewURL.startsWith("blob:")) URL.revokeObjectURL(previewURL);
    setPreviewURL(null);
    setPreviewLabel(null);
    setFile(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) setUploaded(dropped);
  }

  async function handleTestSample(sample: TestSample, image_url: string) {
    if (previewURL && previewURL.startsWith("blob:")) URL.revokeObjectURL(previewURL);
    setPreviewURL(image_url);
    setPreviewLabel(`Test sample · ${sample.display_name}`);

    try {
      const f = await sampleToFile(image_url, sample.class_name);
      setFile(f);
      await onPredict(f);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Could not load test sample";
      onInvalidFile?.(msg);
    }
  }

  const disabledState = disabled || isPredicting;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Choose a leaf</CardTitle>
        <CardDescription>
          Upload your own photo, or pick a leaf from the held-out test set
          (images the model has never seen).
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        <Tabs defaultValue="test" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="test">Test set</TabsTrigger>
            <TabsTrigger value="upload">Upload</TabsTrigger>
          </TabsList>

          {/* ── Test-set picker ────────────────────────────── */}
          <TabsContent value="test" className="space-y-3 pt-3">
            <TestSamplePicker
              onPick={handleTestSample}
              disabled={disabledState}
            />
          </TabsContent>

          {/* ── Upload tab ─────────────────────────────────── */}
          <TabsContent value="upload" className="space-y-3 pt-3">
            <div
              onClick={() => !disabledState && inputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                if (!disabledState) setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              className={cn(
                "relative flex min-h-[200px] cursor-pointer items-center justify-center rounded-lg border-2 border-dashed p-6 text-center transition-colors",
                isDragging
                  ? "border-emerald-500 bg-emerald-500/5"
                  : "border-muted-foreground/25 hover:border-muted-foreground/50",
                disabledState && "cursor-not-allowed opacity-50",
              )}
            >
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) =>
                  e.target.files?.[0] && setUploaded(e.target.files[0])
                }
                disabled={disabledState}
              />
              <div className="space-y-2">
                <Upload className="mx-auto h-9 w-9 text-muted-foreground" />
                <p className="text-sm font-medium">
                  Drag a photo here, or click to browse
                </p>
                <p className="text-xs text-muted-foreground">
                  JPG, PNG, or WebP · max 10 MB
                </p>
              </div>
            </div>
            <Button
              onClick={() => file && !previewLabel && onPredict(file)}
              disabled={!file || !!previewLabel || disabledState}
              className="w-full"
              size="lg"
            >
              {isPredicting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing…
                </>
              ) : (
                "Diagnose"
              )}
            </Button>
          </TabsContent>
        </Tabs>

        {/* ── Shared preview area ─────────────────────────── */}
        {previewURL && (
          <div className="rounded-lg border bg-muted/30 p-3 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-medium">
                {previewLabel ?? file?.name ?? "Selected image"}
              </span>
              <button
                type="button"
                onClick={handleClear}
                disabled={disabledState}
                className="text-muted-foreground hover:text-foreground disabled:opacity-50"
                aria-label="Clear image"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="relative overflow-hidden rounded">
              <img
                src={previewURL}
                alt="Selected leaf preview"
                className="mx-auto max-h-72 w-auto object-contain"
              />
              {isPredicting && (
                <div className="absolute inset-0 flex items-center justify-center bg-background/70 backdrop-blur-sm">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Analyzing…
                  </div>
                </div>
              )}
            </div>
            {file && !previewLabel && (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <ImageIcon className="h-3 w-3" />
                <span className="truncate">{file.name}</span>
                <span className="ml-auto tabular-nums">
                  {(file.size / 1024).toFixed(0)} KB
                </span>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ──────────────────────────────────────────────────────────────────
// Test-set picker
// ──────────────────────────────────────────────────────────────────

function TestSamplePicker({
  onPick,
  disabled,
}: {
  onPick: (sample: TestSample, image_url: string) => void;
  disabled: boolean;
}) {
  const { samples, error, loading } = useTestSamples();
  const [activeCrop, setActiveCrop] = React.useState<string | null>(null);

  // Group by crop
  const cropsByName = React.useMemo(() => {
    if (!samples) return new Map<string, TestSample[]>();
    const m = new Map<string, TestSample[]>();
    for (const s of samples) {
      const list = m.get(s.crop) ?? [];
      list.push(s);
      m.set(s.crop, list);
    }
    return m;
  }, [samples]);

  const crops = React.useMemo(
    () => Array.from(cropsByName.keys()),
    [cropsByName],
  );

  React.useEffect(() => {
    if (crops.length > 0 && !activeCrop) setActiveCrop(crops[0]);
  }, [crops, activeCrop]);

  function pickRandom() {
    if (!samples || samples.length === 0) return;
    const flat: { sample: TestSample; image_url: string }[] = [];
    for (const s of samples) {
      for (const img of s.images) flat.push({ sample: s, image_url: img });
    }
    const choice = flat[Math.floor(Math.random() * flat.length)];
    onPick(choice.sample, choice.image_url);
  }

  if (loading) {
    return (
      <div className="flex h-32 items-center justify-center rounded-lg border bg-muted/30 text-sm text-muted-foreground">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        Loading test samples…
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-2 rounded-lg border border-amber-500/40 bg-amber-500/5 p-4 text-xs">
        <p className="font-medium text-amber-700 dark:text-amber-400">
          Test samples not available
        </p>
        <p className="text-muted-foreground">{error}</p>
        <code className="block rounded bg-background p-2 font-mono">
          PYTHONPATH=src .venv/bin/python scripts/copy_test_samples.py
        </code>
      </div>
    );
  }

  if (!samples || samples.length === 0) return null;

  const activeSamples = activeCrop ? cropsByName.get(activeCrop) ?? [] : [];

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs text-muted-foreground">
          {samples.reduce((n, s) => n + s.images.length, 0)} held-out images
          across {samples.length} classes
        </p>
        <Button
          variant="outline"
          size="sm"
          onClick={pickRandom}
          disabled={disabled}
        >
          <Shuffle className="mr-1 h-3 w-3" />
          Random
        </Button>
      </div>

      {/* Crop chips */}
      <div className="flex flex-wrap gap-1">
        {crops.map((crop) => {
          const isActive = crop === activeCrop;
          return (
            <button
              key={crop}
              onClick={() => setActiveCrop(crop)}
              disabled={disabled}
              className={cn(
                "rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                isActive
                  ? "border-foreground bg-foreground text-background"
                  : "border-border bg-card text-muted-foreground hover:text-foreground",
                disabled && "cursor-not-allowed opacity-50",
              )}
            >
              {crop}
            </button>
          );
        })}
      </div>

      {/* Thumbnails for active crop */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
        {activeSamples.flatMap((sample) =>
          sample.images.map((image_url) => (
            <button
              key={image_url}
              onClick={() => onPick(sample, image_url)}
              disabled={disabled}
              className={cn(
                "group relative overflow-hidden rounded-md border bg-card transition-shadow",
                "hover:shadow-md focus:outline-none focus:ring-2 focus:ring-emerald-500/40",
                disabled && "cursor-not-allowed opacity-50",
              )}
              title={sample.display_name}
            >
              <div className="aspect-square">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={image_url}
                  alt={sample.display_name}
                  className="h-full w-full object-cover transition-transform group-hover:scale-105"
                />
              </div>
              <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent p-2">
                <p className="text-[10px] font-medium leading-tight text-white">
                  {sample.display_name}
                </p>
              </div>
              {sample.is_healthy && (
                <span className="absolute right-1 top-1 rounded-full bg-emerald-500/90 px-1.5 py-0.5 text-[9px] font-medium text-white">
                  healthy
                </span>
              )}
              {sample.is_background && (
                <span className="absolute right-1 top-1 rounded-full bg-amber-500/90 px-1.5 py-0.5 text-[9px] font-medium text-white">
                  background
                </span>
              )}
            </button>
          )),
        )}
      </div>
    </div>
  );
}
