"use client";

import { useEffect, useState } from "react";

import {
  AreaChart,
  BarList,
  DonutChart,
  RadialMeter,
  ScoreComparison,
} from "@/components/charts";
import {
  Badge,
  Card,
  Empty,
  ErrorNote,
  SectionTitle,
  Stat,
} from "@/components/ui";
import { getAnalytics } from "@/lib/api";

export default function DashboardPage() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getAnalytics().then(setData).catch((err) => setError(err.message));
  }, []);

  if (error) {
    return (
      <div className="p-8">
        <ErrorNote>{error}</ErrorNote>
      </div>
    );
  }

  // A skeleton rather than a centred spinner: the layout appears immediately
  // at its final size, so nothing jumps when the data lands.
  if (!data) return <DashboardSkeleton />;

  const { overview, activity, repositories, languages, top_files, retrieval,
          recent_questions, system } = data;

  return (
    <div className="mx-auto max-w-5xl px-4 py-8 sm:px-8 sm:py-10">
      <header className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight text-slate-900">
          Dashboard
        </h1>
        <p className="mt-1 text-sm text-slate-500">
          Indexing, usage and retrieval quality across every repository.
        </p>
      </header>

      {/* Headline numbers are stat tiles, not a chart: five unrelated scalars
          have no shared scale to plot against. */}
      <Card className="mb-6">
        <div className="grid grid-cols-2 gap-6 sm:grid-cols-5">
          <Stat
            label="Repositories"
            value={overview.repositories}
            sub={`${overview.repositories_indexed} indexed`}
          />
          <Stat
            label="Files"
            value={overview.total_files.toLocaleString()}
            sub={`${overview.total_chunks.toLocaleString()} chunks`}
          />
          <Stat
            label="Questions"
            value={overview.questions}
            sub={`${overview.conversations} conversations`}
          />
          <Stat
            label="Avg response"
            value={
              overview.avg_response_ms
                ? `${(overview.avg_response_ms / 1000).toFixed(1)}s`
                : "—"
            }
            sub={`${overview.avg_context_chunks} chunks used`}
          />
          <Stat
            label="Refusal rate"
            value={`${Math.round(overview.refusal_rate * 100)}%`}
            sub={`${overview.refusals} of ${overview.answers}`}
          />
        </div>
      </Card>

      <div className="mb-6 grid gap-6 lg:grid-cols-2">
        <div>
          <SectionTitle hint={`${activity.length} days`}>
            Questions per day
          </SectionTitle>
          <Card hover>
            {/* A single day is not a trend - drawing it as an area gives a flat
                line that reads as "maxed out". Show the number instead, which
                is what the reader can actually use. */}
            {activity.length >= 2 ? (
              <AreaChart data={activity} />
            ) : activity.length === 1 ? (
              <div className="flex items-baseline gap-3 py-6">
                <span className="text-4xl font-semibold text-slate-900">
                  {activity[0].questions}
                </span>
                <div className="text-xs text-slate-500">
                  <div>questions today</div>
                  <div className="mt-0.5 text-slate-400">
                    {activity[0].refusals} not found in the code
                  </div>
                  <div className="mt-1.5 text-[11px] text-slate-400">
                    A trend line appears once there is more than one day of
                    activity.
                  </div>
                </div>
              </div>
            ) : (
              <Empty title="No activity yet" />
            )}
          </Card>
        </div>

        <div>
          <SectionTitle hint="share of indexed files">
            Languages indexed
          </SectionTitle>
          <Card hover>
            {Object.keys(languages).length ? (
              // A donut is the right form here and only here: the job is
              // part-to-whole at a glance with few segments. The rankings
              // below stay as bars - a pie cannot express an ordering.
              <DonutChart
                rows={Object.entries(languages).map(([label, value]) => ({
                  label,
                  value,
                }))}
                centerValue={overview.total_files.toLocaleString()}
                centerLabel="files"
              />
            ) : (
              <Empty title="Nothing indexed yet" />
            )}
          </Card>
        </div>
      </div>

      <div className="mb-6 grid gap-6 lg:grid-cols-2">
        <div>
          <SectionTitle hint={`${retrieval.samples} chunks scored`}>
            Retrieval quality
          </SectionTitle>
          <Card hover>
            {/* A single ratio against a limit is a meter, not a two-slice
                pie. The score comparison beside it stays as bars. */}
            <div className="mb-4 border-b border-slate-100 pb-4">
              <RadialMeter
                value={overview.refusal_rate}
                label="Refusal rate"
                sublabel={`${overview.refusals} of ${overview.answers} answers declined`}
                tone={overview.refusal_rate > 0.4 ? "warn" : "series"}
              />
            </div>
            <ScoreComparison
              answered={retrieval.answered}
              refused={retrieval.refused}
            />
            <p className="mt-4 text-xs leading-relaxed text-slate-400">
              Average best-match similarity per answer. Refused questions should
              score <em>lower</em> — if they don&apos;t, retrieval isn&apos;t
              separating relevant code from irrelevant.
            </p>
          </Card>
        </div>

        <div>
          <SectionTitle hint="cited in answers">Most cited files</SectionTitle>
          <Card hover>
            {top_files.length ? (
              // A ranking - bars, not a pie. Angle cannot be read as order.
              <BarList
                rows={top_files.slice(0, 7).map((f) => ({
                  label: f.file_path.split("/").slice(-2).join("/"),
                  value: f.citations,
                }))}
                valueFormat={(v) => `${v}×`}
              />
            ) : (
              <Empty title="No answers yet" />
            )}
          </Card>
        </div>
      </div>

      <div className="mb-6">
        <SectionTitle>Repositories</SectionTitle>
        <Card padded={false}>
          {/* Six columns cannot fit a phone. Scrolling horizontally keeps every
              number visible; hiding columns would silently drop data. */}
          <div className="scroll-thin overflow-x-auto">
          <table className="w-full min-w-160 text-sm">
            <thead>
              <tr className="border-b border-slate-100 text-left text-xs text-slate-400">
                <th className="px-5 py-2.5 font-medium">Repository</th>
                <th className="px-3 py-2.5 text-right font-medium">Files</th>
                <th className="px-3 py-2.5 text-right font-medium">Chunks</th>
                <th className="px-3 py-2.5 text-right font-medium">Questions</th>
                <th className="px-3 py-2.5 text-right font-medium">Avg time</th>
                <th className="px-5 py-2.5 font-medium">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {repositories.map((repo) => (
                <tr key={repo.repository_id}>
                  <td className="px-5 py-2.5">
                    <div className="truncate font-medium text-slate-700">
                      {repo.repository}
                    </div>
                  </td>
                  <td className="px-3 py-2.5 text-right tabular-nums text-slate-600">
                    {repo.supported_files || "—"}
                  </td>
                  <td className="px-3 py-2.5 text-right tabular-nums text-slate-600">
                    {repo.total_chunks}
                  </td>
                  <td className="px-3 py-2.5 text-right tabular-nums text-slate-600">
                    {repo.questions}
                  </td>
                  <td className="px-3 py-2.5 text-right tabular-nums text-slate-600">
                    {repo.avg_elapsed_ms
                      ? `${(repo.avg_elapsed_ms / 1000).toFixed(1)}s`
                      : "—"}
                  </td>
                  <td className="px-5 py-2.5">
                    <RepoStatus
                      repo={repo}
                      // Qdrant is the authority on whether a repository can
                      // actually be searched. The stored `indexed` flag only
                      // records what the *last* analyze did - a run with
                      // auto_index off set it false even though the vectors
                      // from an earlier run were still there, which showed a
                      // searchable repository as "not indexed".
                      searchable={Boolean(
                        system.qdrant?.repositories?.[repo.repository_id]
                      )}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          </div>
        </Card>
      </div>

      <div className="mb-6 grid gap-6 lg:grid-cols-2">
        <div>
          <SectionTitle>Recent questions</SectionTitle>
          <Card padded={false}>
            {recent_questions.length ? (
              <ul className="divide-y divide-slate-100">
                {recent_questions.slice(0, 8).map((q) => (
                  <li key={q.id} className="px-5 py-2.5">
                    <div className="flex items-start justify-between gap-2">
                      <span className="line-clamp-1 text-xs text-slate-700">
                        {q.question}
                      </span>
                      {q.refused ? (
                        <Badge tone="amber">not found</Badge>
                      ) : (
                        <Badge tone="green">answered</Badge>
                      )}
                    </div>
                    <div className="mt-0.5 text-[11px] text-slate-400">
                      {q.repository_id.split("__").slice(1).join("__")}
                      {q.elapsed_ms
                        ? ` · ${(q.elapsed_ms / 1000).toFixed(1)}s`
                        : ""}
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <Empty title="No questions yet" />
            )}
          </Card>
        </div>

        <div>
          <SectionTitle>System</SectionTitle>
          <Card>
            <dl className="space-y-2.5 text-xs">
              <Row label="Embedding model" value={system.embedding_model} />
              <Row label="LLM" value={system.llm_model} />
              <Row
                label="Chunk size"
                value={`${system.chunk_size_tokens} tokens · ${system.chunk_overlap_tokens} overlap`}
              />
              <Row
                label="Qdrant"
                value={
                  system.qdrant.reachable
                    ? `${system.qdrant.points} vectors · ${system.qdrant.dimension}d ${system.qdrant.distance}`
                    : "offline"
                }
                tone={system.qdrant.reachable ? "green" : "red"}
              />
              <Row
                label="Database"
                value={`${system.database.messages} messages · ${(
                  system.database.size_bytes / 1024
                ).toFixed(0)} KB`}
              />
            </dl>
          </Card>
        </div>
      </div>
    </div>
  );
}

function DashboardSkeleton() {
  const block = "animate-pulse rounded bg-slate-100";
  return (
    <div className="mx-auto max-w-5xl px-4 py-8 sm:px-8 sm:py-10">
      <div className={`${block} h-7 w-40`} />
      <div className={`${block} mt-2 h-4 w-72`} />

      <Card className="mt-8">
        <div className="grid grid-cols-2 gap-6 sm:grid-cols-5">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i}>
              <div className={`${block} h-3 w-16`} />
              <div className={`${block} mt-2 h-7 w-12`} />
            </div>
          ))}
        </div>
      </Card>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <Card key={i}>
            <div className={`${block} h-3 w-24`} />
            <div className={`${block} mt-4 h-28 w-full`} />
          </Card>
        ))}
      </div>
    </div>
  );
}

function RepoStatus({ repo, searchable }) {
  if (!searchable) return <Badge tone="slate">no vectors</Badge>;
  // Searchable, but analysed before persistence existed - its file and parse
  // counts are unknown rather than zero, so say which it is.
  if (repo.metadata_missing || !repo.supported_files) {
    return <Badge tone="amber">re-analyze for stats</Badge>;
  }
  return <Badge tone="green">indexed</Badge>;
}

function Row({ label, value, tone }) {
  return (
    <div className="flex items-baseline justify-between gap-4">
      <dt className="text-slate-400">{label}</dt>
      <dd className="truncate text-right font-medium text-slate-700">
        {tone ? <Badge tone={tone}>{value}</Badge> : value}
      </dd>
    </div>
  );
}
