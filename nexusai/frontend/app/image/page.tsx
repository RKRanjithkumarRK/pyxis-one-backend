"use client";

import { useState } from "react";
import Image from "next/image";
import { useImageStudio } from "@/hooks/use-image";
import type { ImageRequest } from "@/lib/api-types";

const MODELS = [
  { id: "flux-schnell", label: "Flux Schnell", desc: "Fastest" },
  { id: "flux-pro", label: "Flux Pro", desc: "Highest quality" },
  { id: "sdxl", label: "SDXL", desc: "Stable Diffusion XL" },
  { id: "dall-e-3", label: "DALL·E 3", desc: "OpenAI" },
  { id: "imagen-3", label: "Imagen 3", desc: "Google" },
];

const SIZES = [
  { label: "Square", width: 1024, height: 1024 },
  { label: "Landscape", width: 1792, height: 1024 },
  { label: "Portrait", width: 1024, height: 1792 },
];

function ImageGrid({ req }: { req: ImageRequest }) {
  if (req.status === "pending" || req.status === "processing") {
    return (
      <div className="grid grid-cols-2 gap-2">
        {Array.from({ length: req.num_images }).map((_, i) => (
          <div
            key={i}
            className="aspect-square rounded-xl bg-muted animate-pulse flex items-center justify-center"
          >
            <span className="text-xs text-muted-foreground">Generating…</span>
          </div>
        ))}
      </div>
    );
  }

  if (req.status === "error") {
    return (
      <div className="rounded-xl bg-destructive/10 text-destructive text-sm p-4">
        Generation failed: {req.error_msg}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-2">
      {(req.result_urls ?? []).map((url, i) => (
        <div key={i} className="relative aspect-square rounded-xl overflow-hidden group">
          <img src={url} alt={`Generated ${i + 1}`} className="w-full h-full object-cover" />
          <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors" />
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            download
            className="absolute bottom-2 right-2 opacity-0 group-hover:opacity-100 bg-white/90 text-xs px-2 py-1 rounded-lg font-medium transition-opacity"
          >
            Download
          </a>
        </div>
      ))}
    </div>
  );
}

function HistoryCard({ req, onClick }: { req: ImageRequest; onClick: () => void }) {
  const previewUrl = req.result_urls?.[0];
  return (
    <button
      onClick={onClick}
      className="relative group rounded-xl overflow-hidden border border-border aspect-square bg-muted text-left"
    >
      {previewUrl ? (
        <img src={previewUrl} alt={req.prompt} className="w-full h-full object-cover" />
      ) : (
        <div className="w-full h-full flex items-center justify-center text-muted-foreground text-xs">
          {req.status === "error" ? "Error" : "Generating…"}
        </div>
      )}
      <div className="absolute inset-0 bg-black/0 group-hover:bg-black/50 transition-colors flex items-end p-2 opacity-0 group-hover:opacity-100">
        <p className="text-white text-xs line-clamp-2">{req.prompt}</p>
      </div>
    </button>
  );
}

export default function ImageStudioPage() {
  const { history, generating, current, error, generate } = useImageStudio();

  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [model, setModel] = useState("flux-schnell");
  const [sizeIdx, setSizeIdx] = useState(0);
  const [numImages, setNumImages] = useState(4);
  const [selected, setSelected] = useState<ImageRequest | null>(null);

  const displayReq = selected ?? current;

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || generating) return;
    const size = SIZES[sizeIdx];
    setSelected(null);
    await generate({
      prompt: prompt.trim(),
      negative_prompt: negativePrompt || undefined,
      model,
      width: size.width,
      height: size.height,
      num_images: numImages,
    });
  };

  return (
    <div className="min-h-screen bg-background flex gap-0">
      {/* Left panel — controls */}
      <div className="w-80 shrink-0 border-r border-border flex flex-col h-screen overflow-y-auto">
        <div className="p-4 border-b border-border">
          <h1 className="text-lg font-bold">Image Studio</h1>
          <p className="text-xs text-muted-foreground mt-0.5">Generate images with top AI models</p>
        </div>

        <form onSubmit={handleGenerate} className="p-4 space-y-4 flex-1">
          <div>
            <label className="text-xs font-medium text-muted-foreground">Prompt</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="A photorealistic portrait of an astronaut on Mars…"
              rows={4}
              className="mt-1 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>

          <div>
            <label className="text-xs font-medium text-muted-foreground">Negative prompt</label>
            <input
              value={negativePrompt}
              onChange={(e) => setNegativePrompt(e.target.value)}
              placeholder="blurry, low quality…"
              className="mt-1 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>

          <div>
            <label className="text-xs font-medium text-muted-foreground">Model</label>
            <div className="mt-1 space-y-1">
              {MODELS.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => setModel(m.id)}
                  className={`w-full flex items-center justify-between px-3 py-2 rounded-lg border text-sm transition-colors ${
                    model === m.id
                      ? "border-primary bg-primary/5 text-foreground"
                      : "border-border text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <span className="font-medium">{m.label}</span>
                  <span className="text-xs text-muted-foreground">{m.desc}</span>
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="text-xs font-medium text-muted-foreground">Size</label>
            <div className="mt-1 flex gap-1">
              {SIZES.map((s, i) => (
                <button
                  key={s.label}
                  type="button"
                  onClick={() => setSizeIdx(i)}
                  className={`flex-1 py-1.5 rounded-lg border text-xs transition-colors ${
                    sizeIdx === i
                      ? "border-primary bg-primary/5 text-foreground"
                      : "border-border text-muted-foreground"
                  }`}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="text-xs font-medium text-muted-foreground">
              Batch size: {numImages}
            </label>
            <input
              type="range"
              min={1}
              max={4}
              value={numImages}
              onChange={(e) => setNumImages(Number(e.target.value))}
              className="mt-1 w-full"
            />
          </div>

          {error && (
            <p className="text-xs text-destructive">{error}</p>
          )}

          <button
            type="submit"
            disabled={generating || !prompt.trim()}
            className="w-full py-2.5 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 disabled:opacity-50 transition-colors"
          >
            {generating ? "Generating…" : "Generate"}
          </button>
        </form>
      </div>

      {/* Center — current result */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto p-6">
          {displayReq ? (
            <div className="max-w-2xl mx-auto">
              <p className="text-sm text-muted-foreground mb-3 line-clamp-2">
                {displayReq.prompt}
              </p>
              <ImageGrid req={displayReq} />
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-center">
              <div>
                <div className="text-5xl mb-3">🎨</div>
                <p className="font-medium">Describe what you want to create</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Your generated images will appear here
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Right panel — history */}
      {history.length > 0 && (
        <div className="w-48 shrink-0 border-l border-border overflow-y-auto p-2 space-y-2">
          <p className="text-xs font-medium text-muted-foreground px-1 py-1">History</p>
          {history.map((req) => (
            <HistoryCard
              key={req.id}
              req={req}
              onClick={() => setSelected(req)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
