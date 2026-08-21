"use client";

import { useEffect, useState } from "react";

import { analyzeRepository, askQuestion, getHealth } from "@/lib/api";
import AnswerText from "./AnswerText";

const EXAMPLE_QUESTIONS = [
  "Where is the database connection initialized?",
  "Which files handle API requests?",
  "Where is authentication implemented?",
  "How is a user created?",
];

export default function Home() {
  const [health, setHealth] = useState(null);

  const [repositoryUrl, setRepositoryUrl] = useState("");
  const [githubToken, setGithubToken] = useState("");
  const [showToken, setShowToken] = useState(false);

  const [repository, setRepository] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState("");

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [asking, setAsking] = useState(false);
  const [askError, setAskError] = useState("");

  // Check backend readiness on load, so a missing Qdrant or Groq key shows up
  // as a clear banner instead of a confusing failure on first use.
  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((error) => setHealth({ unreachable: error.message }));
  }, []);

  // Repositories already indexed in Qdrant survive a backend restart, so they
  // can be queried without re-analysing.
  const indexedRepositories = Object.entries(health?.qdrant?.repositories ?? {});

  async function handleAnalyze(event) {
    event.preventDefault();
    setAnalyzing(true);
    setAnalyzeError("");
    setAnswer(null);
    try {
      const result = await analyzeRepository({ repositoryUrl, githubToken });
      setRepository(result);
      setHealth(await getHealth());
    } catch (error) {
      setAnalyzeError(error.message);
      setRepository(null);
    } finally {
      setAnalyzing(false);
    }
  }

  async function handleAsk(event) {
    event.preventDefault();
    if (!repository) return;
    setAsking(true);
    setAskError("");
    setAnswer(null);
    try {
      setAnswer(
        await askQuestion({
          repositoryId: repository.repository_id,
          question,
        })
      );
    } catch (error) {
      setAskError(error.message);
    } finally {
      setAsking(false);
    }
  }

  function selectIndexed(repositoryId, chunkCount) {
    // Build a minimal record from what /health already told us: enough to ask
    // questions, without re-cloning the repository.
    setRepository({
      repository_id: repositoryId,
      repository: repositoryId.split("__").slice(1).join("__") || repositoryId,
      indexed: true,
      restored: true,
      chunk_stats: { total_chunks: chunkCount },
    });
    setAnswer(null);
    setAnalyzeError("");
  }

  return (
    <main className="mx-auto max-w-3xl px-6 py-10">
      <h1 className="text-3xl font-bold tracking-tight">REPOSENSE AI</h1>
      <p className="mt-1 text-sm text-slate-600">
        Ask questions about any GitHub repository, answered from its own code.
      </p>

      <HealthBanner health={health} />

      {/* ---------------- repository import ---------------- */}
      <Section title="GitHub Repository URL">
        <form onSubmit={handleAnalyze} className="space-y-3">
          <input
            type="text"
            value={repositoryUrl}
            onChange={(event) => setRepositoryUrl(event.target.value)}
            placeholder="https://github.com/owner/repository"
            className="w-full rounded border border-slate-300 bg-white px-3 py-2 text-sm outline-none focus:border-slate-500"
            required
          />

          <div className="text-xs">
            <button
              type="button"
              onClick={() => setShowToken(!showToken)}
              className="text-slate-600 underline"
            >
              {showToken ? "Hide" : "Private repository? Add a GitHub token"}
            </button>
            {showToken && (
              <div className="mt-2">
                <input
                  type="password"
                  value={githubToken}
                  onChange={(event) => setGithubToken(event.target.value)}
                  placeholder="github_pat_... (only needed for private repos)"
                  className="w-full rounded border border-slate-300 bg-white px-3 py-2 text-sm outline-none focus:border-slate-500"
                />
                <p className="mt-1 text-slate-500">
                  Sent with this request only. Never stored or logged. Leave
                  empty to use GITHUB_TOKEN from the server&apos;s .env file.
                </p>
              </div>
            )}
          </div>

          <button
            type="submit"
            disabled={analyzing || !repositoryUrl.trim()}
            className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            {analyzing ? "Analyzing..." : "Analyze Repository"}
          </button>

          {analyzing && (
            <p className="text-xs text-slate-500">
              Cloning, chunking and embedding the code. This takes 10-30 seconds
              for a small repository - the embeddings run locally on CPU.
            </p>
          )}
        </form>

        {analyzeError && <ErrorBox message={analyzeError} />}

        {indexedRepositories.length > 0 && !repository && (
          <div className="mt-4 text-xs text-slate-600">
            <p className="font-medium">Already indexed - ask without re-analyzing:</p>
            <ul className="mt-1 space-y-1">
              {indexedRepositories.map(([id, count]) => (
                <li key={id}>
                  <button
                    onClick={() => selectIndexed(id, count)}
                    className="underline"
                  >
                    {id}
                  </button>{" "}
                  <span className="text-slate-400">({count} chunks)</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </Section>

      {/* ---------------- status ---------------- */}
      {repository && (
        <Section title="Repository Status">
          <p className="text-sm">
            <span className="font-mono">{repository.repository_id}</span>
          </p>
          <p className="mt-1 text-sm text-slate-700">
            {repository.supported_files !== undefined && (
              <>Files: {repository.supported_files} | </>
            )}
            Chunks: {repository.chunk_stats?.total_chunks ?? 0} | Indexed:{" "}
            {repository.indexed ? "OK" : "no"}
          </p>
          {repository.restored && (
            <p className="mt-1 text-xs text-slate-500">
              Loaded from the vector database. Re-analyze to refresh it from
              GitHub.
            </p>
          )}
          {repository.language_counts && (
            <p className="mt-1 text-xs text-slate-500">
              {Object.entries(repository.language_counts)
                .map(([language, count]) => `${language} ${count}`)
                .join(" | ")}
            </p>
          )}
        </Section>
      )}

      {/* ---------------- ask ---------------- */}
      {repository && (
        <Section title="Ask About Your Repository">
          <form onSubmit={handleAsk} className="space-y-3">
            <input
              type="text"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="How does authentication work?"
              className="w-full rounded border border-slate-300 bg-white px-3 py-2 text-sm outline-none focus:border-slate-500"
              required
            />
            <button
              type="submit"
              disabled={asking || !question.trim()}
              className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-slate-400"
            >
              {asking ? "Thinking..." : "Ask"}
            </button>
          </form>

          <div className="mt-3 flex flex-wrap gap-2">
            {EXAMPLE_QUESTIONS.map((example) => (
              <button
                key={example}
                onClick={() => setQuestion(example)}
                className="rounded border border-slate-300 px-2 py-1 text-xs text-slate-600"
              >
                {example}
              </button>
            ))}
          </div>

          {askError && <ErrorBox message={askError} />}
        </Section>
      )}

      {/* ---------------- answer ---------------- */}
      {answer && (
        <>
          <Section title="AI Response">
            <AnswerText text={answer.answer} />

            {answer.truncated_answer && (
              <p className="mt-2 rounded bg-amber-50 px-3 py-2 text-xs text-amber-800">
                This answer hit the length limit and may be cut off. Try a more
                specific question.
              </p>
            )}

            <p className="mt-3 text-xs text-slate-500">
              {answer.model} | {answer.context_chunks} chunks of context |{" "}
              {answer.prompt_tokens} in / {answer.completion_tokens} out |{" "}
              {Math.round(answer.elapsed_ms)} ms
            </p>
          </Section>

          <Section title="Relevant Files">
            {answer.sources?.length ? (
              <ul className="space-y-1 text-sm">
                {answer.sources.map((file) => (
                  <li key={file} className="font-mono text-slate-700">
                    - {file}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-slate-500">
                No files cited - the answer was not found in this repository.
              </p>
            )}

            {answer.context?.length > 0 && (
              <details className="mt-3">
                <summary className="cursor-pointer text-xs text-slate-600">
                  Show the exact code the model was given ({answer.context.length}{" "}
                  chunks)
                </summary>
                <ul className="mt-2 space-y-1 text-xs">
                  {answer.context.map((chunk, index) => (
                    <li key={index} className="font-mono text-slate-600">
                      {chunk.file_path}:{chunk.start_line}-{chunk.end_line}{" "}
                      <span className="text-slate-400">
                        (score {chunk.score.toFixed(3)})
                      </span>
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </Section>
        </>
      )}
    </main>
  );
}

function Section({ title, children }) {
  return (
    <section className="mt-8">
      <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
        {title}
      </h2>
      <div className="rounded border border-slate-200 bg-white p-4">
        {children}
      </div>
    </section>
  );
}

function ErrorBox({ message }) {
  return (
    <p className="mt-3 rounded bg-red-50 px-3 py-2 text-sm text-red-700">
      {message}
    </p>
  );
}

function HealthBanner({ health }) {
  if (!health) return null;

  if (health.unreachable) {
    return (
      <Warning>
        <strong>Backend not reachable.</strong> {health.unreachable} Start it
        with <code>.\run_dev.ps1</code> in the <code>backend</code> folder.
      </Warning>
    );
  }

  const problems = [];
  if (!health.qdrant?.reachable) {
    problems.push(
      "Qdrant is not running - start it with `docker compose up -d` from the project root."
    );
  }
  if (!health.groq?.configured) {
    problems.push(
      "GROQ_API_KEY is not set in backend/.env - answers will fail until it is."
    );
  }
  if (!problems.length) return null;

  return (
    <Warning>
      <strong>Setup incomplete:</strong>
      <ul className="mt-1 list-inside list-disc">
        {problems.map((problem) => (
          <li key={problem}>{problem}</li>
        ))}
      </ul>
    </Warning>
  );
}

function Warning({ children }) {
  return (
    <div className="mt-4 rounded border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900">
      {children}
    </div>
  );
}
