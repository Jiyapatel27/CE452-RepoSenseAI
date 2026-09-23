"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import {
  Badge,
  Button,
  Card,
  Empty,
  ErrorNote,
  Input,
  SectionTitle,
  Spinner,
  Stat,
} from "@/components/ui";
import { analyzeRepository, getHealth, getRepositoryStatus } from "@/lib/api";

export default function ImportPage() {
  const router = useRouter();

  const [url, setUrl] = useState("");
  const [token, setToken] = useState("");
  const [showToken, setShowToken] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [known, setKnown] = useState([]);

  useEffect(() => {
    loadKnown();
  }, []);

  async function loadKnown() {
    try {
      const [status, health] = await Promise.all([
        getRepositoryStatus().catch(() => ({ repositories: [] })),
        getHealth().catch(() => null),
      ]);

      // /status only knows what the server has in memory. Qdrant is the real
      // record of what is searchable, so merge the two and prefer the richer
      // in-memory entry when both exist.
      const vectors = health?.qdrant?.repositories || {};
      const byId = new Map();
      for (const [id, chunks] of Object.entries(vectors)) {
        byId.set(id, { repository_id: id, total_chunks: chunks, indexed: true });
      }
      for (const repo of status.repositories || []) {
        byId.set(repo.repository_id, { ...byId.get(repo.repository_id), ...repo });
      }
      setKnown([...byId.values()].sort((a, b) =>
        a.repository_id.localeCompare(b.repository_id)
      ));
    } catch {
      setKnown([]);
    }
  }

  async function onAnalyze(event) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const data = await analyzeRepository({
        repositoryUrl: url,
        githubToken: token,
      });
      setResult(data);
      loadKnown();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-8 sm:px-8 sm:py-10">
      <header className="mb-9">
        <h1 className="text-4xl font-bold tracking-tight text-slate-900">
          Import a repository
        </h1>
        <p className="mt-2.5 max-w-xl text-base leading-relaxed text-slate-500">
          Clone, chunk and index a codebase so you can ask questions about it.
        </p>
      </header>

      <Card className="p-6 sm:p-7">
        <form onSubmit={onAnalyze}>
          <label className="mb-2 block text-sm font-semibold text-slate-700">
            GitHub repository URL
          </label>
          <Input
            className="py-3"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://github.com/owner/repository"
            required
          />

          <button
            type="button"
            onClick={() => setShowToken(!showToken)}
            className="mt-3.5 text-sm text-slate-500 underline decoration-slate-300 underline-offset-2 hover:text-slate-700"
          >
            {showToken ? "Hide token field" : "Private repository? Add a token"}
          </button>

          {showToken && (
            <div className="mt-2.5">
              <Input
                className="py-3"
                type="password"
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder="github_pat_..."
              />
              <p className="mt-2 text-sm text-slate-400">
                Sent with this request only — never stored or logged. Leave empty
                to use the server&apos;s configured token.
              </p>
            </div>
          )}

          <div className="mt-6 flex flex-wrap items-center gap-3">
            <Button
              type="submit"
              disabled={busy || !url.trim()}
              className="px-5 py-2.5 text-base"
            >
              {busy && <Spinner className="h-4.5 w-4.5" />}
              {busy ? "Analyzing…" : "Analyze repository"}
            </Button>
            {busy && (
              <span className="text-sm text-slate-400">
                Cloning, chunking and embedding — 10–30s, running locally on CPU
              </span>
            )}
          </div>
        </form>

        {error && <ErrorNote onDismiss={() => setError("")}>{error}</ErrorNote>}
      </Card>

      {result && (
        <div className="mt-6">
          <SectionTitle hint={result.repository_id}>Indexed</SectionTitle>
          <Card>
            <div className="grid grid-cols-4 gap-4">
              <Stat label="Files" value={result.supported_files} />
              <Stat label="Chunks" value={result.chunk_stats.total_chunks} />
              <Stat
                label="Lines"
                value={result.parse_stats.total_lines.toLocaleString()}
              />
              <Stat
                label="Indexed"
                value={result.indexed ? "Yes" : "No"}
                sub={result.indexed ? `${result.chunks_indexed} vectors` : null}
              />
            </div>

            <div className="mt-5 flex flex-wrap gap-1.5 border-t border-slate-100 pt-4">
              {Object.entries(result.language_counts).map(([lang, n]) => (
                <Badge key={lang} tone="slate">
                  {lang} {n}
                </Badge>
              ))}
            </div>

            <Button
              className="mt-5"
              onClick={() =>
                router.push(`/chat?repo=${encodeURIComponent(result.repository_id)}`)
              }
            >
              Ask questions about it →
            </Button>
          </Card>
        </div>
      )}

      <div className="mt-8">
        <SectionTitle hint={known.length ? `${known.length} available` : null}>
          Previously indexed
        </SectionTitle>
        <Card padded={false}>
          {known.length === 0 ? (
            <Empty title="Nothing indexed yet">
              Analyze a repository above to get started.
            </Empty>
          ) : (
            <ul className="divide-y divide-slate-100">
              {known.map((repo) => (
                <li
                  key={repo.repository_id}
                  className="flex items-center justify-between gap-3 px-6 py-4 transition-colors hover:bg-slate-50/70"
                >
                  <div className="min-w-0">
                    <div className="truncate text-base font-semibold text-slate-800">
                      {repo.repository || repo.repository_id}
                    </div>
                    <div className="mt-1 text-sm text-slate-400">
                      {repo.total_chunks} chunks
                      {repo.supported_files
                        ? ` · ${repo.supported_files} files`
                        : ""}
                    </div>
                  </div>
                  <Button
                    variant="secondary"
                    onClick={() =>
                      router.push(
                        `/chat?repo=${encodeURIComponent(repo.repository_id)}`
                      )
                    }
                  >
                    Chat →
                  </Button>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>
    </div>
  );
}

