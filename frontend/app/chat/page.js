"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef, useState } from "react";

import AnswerText from "@/components/AnswerText";
import {
  Badge,
  Button,
  Empty,
  ErrorNote,
  FilePath,
  Input,
  Spinner,
} from "@/components/ui";
import {
  deleteConversation,
  getConversation,
  getHealth,
  listConversations,
  sendMessage,
  startConversation,
} from "@/lib/api";

const SUGGESTIONS = [
  "Where is authentication implemented?",
  "Where is the database connection initialized?",
  "Which files handle API requests?",
  "How is a user created?",
];

export default function ChatPage() {
  // useSearchParams needs a Suspense boundary in the App Router.
  return (
    <Suspense fallback={<div className="p-8 text-sm text-slate-400">Loading…</div>}>
      <Chat />
    </Suspense>
  );
}

function Chat() {
  const params = useSearchParams();

  const [repositories, setRepositories] = useState([]);
  const [repositoryId, setRepositoryId] = useState(params.get("repo") || "");
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [historyOpen, setHistoryOpen] = useState(false);

  const threadRef = useRef(null);

  useEffect(() => {
    getHealth()
      .then((health) => {
        const repos = Object.keys(health?.qdrant?.repositories || {});
        setRepositories(repos);
        setRepositoryId((current) => current || repos[0] || "");
      })
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!repositoryId) return;
    listConversations(repositoryId)
      .then((data) => setConversations(data.conversations))
      .catch(() => setConversations([]));
    // Switching repository invalidates the open thread - it belongs to the
    // previous one.
    setActiveId(null);
    setMessages([]);
  }, [repositoryId]);

  // Keep the newest message in view as the thread grows.
  useEffect(() => {
    threadRef.current?.scrollTo({
      top: threadRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, busy]);

  async function openConversation(id) {
    setError("");
    setActiveId(id);
    setMessages([]);
    try {
      const data = await getConversation(id);
      setMessages(data.messages);
    } catch (err) {
      setError(err.message);
    }
  }

  async function ask(text) {
    const trimmed = (text ?? question).trim();
    if (!trimmed || busy) return;

    setBusy(true);
    setError("");
    setQuestion("");

    // Show the question straight away; the answer replaces this optimistic
    // entry when it arrives.
    const pending = {
      id: `pending-${Date.now()}`,
      role: "user",
      content: trimmed,
      sources: [],
    };
    setMessages((current) => [...current, pending]);

    try {
      const data = activeId
        ? await sendMessage({ conversationId: activeId, question: trimmed })
        : await startConversation({ repositoryId, question: trimmed });

      setActiveId(data.conversation.id);
      // Re-read the thread so the optimistic message is replaced by the stored
      // one, keeping ids and metadata consistent with the server.
      const full = await getConversation(data.conversation.id);
      setMessages(full.messages);

      const list = await listConversations(repositoryId);
      setConversations(list.conversations);
    } catch (err) {
      setError(err.message);
      setMessages((current) => current.filter((m) => m.id !== pending.id));
    } finally {
      setBusy(false);
    }
  }

  async function removeConversation(id, event) {
    event.stopPropagation();
    try {
      await deleteConversation(id);
      setConversations((current) => current.filter((c) => c.id !== id));
      if (activeId === id) {
        setActiveId(null);
        setMessages([]);
      }
    } catch (err) {
      setError(err.message);
    }
  }

  if (!repositories.length) {
    return (
      <div className="p-8">
        <Empty title="No repositories indexed">
          Import one from the Import tab first, then come back to ask questions.
        </Empty>
        {error && <ErrorNote>{error}</ErrorNote>}
      </div>
    );
  }

  return (
    // The app bar takes ~45px below lg, so a plain h-screen would push the
    // composer off the bottom on mobile.
    <div className="flex h-[calc(100vh-45px)] lg:h-screen">
      {historyOpen && (
        <button
          aria-label="Close history"
          onClick={() => setHistoryOpen(false)}
          className="fixed inset-0 z-20 bg-slate-900/20 md:hidden"
        />
      )}

      {/* ---------------- history ---------------- */}
      <div
        className={`fixed inset-y-0 left-0 z-30 flex w-64 shrink-0 flex-col border-r border-slate-200 bg-white transition-transform md:static md:z-auto md:translate-x-0 md:bg-white/60 ${
          historyOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="border-b border-slate-200/70 p-3">
          <label className="mb-1.5 block px-0.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
            Repository
          </label>
          <select
            value={repositoryId}
            onChange={(e) => setRepositoryId(e.target.value)}
            // appearance-none + a drawn caret, so the control matches the rest
            // of the UI instead of inheriting the OS widget.
            className="w-full cursor-pointer appearance-none rounded-lg bg-white bg-size-[14px] bg-position-[right_0.6rem_center] bg-no-repeat py-2 pl-2.5 pr-8 text-xs font-medium text-slate-700 shadow-sm ring-1 ring-slate-200 transition-shadow focus:outline-none focus:ring-2 focus:ring-indigo-400"
            style={{
              backgroundImage:
                "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%238994a8' stroke-width='2.5' stroke-linecap='round'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E\")",
            }}
          >
            {repositories.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>

          <Button
            size="sm"
            variant="secondary"
            className="mt-2 w-full"
            onClick={() => {
              setActiveId(null);
              setMessages([]);
              setError("");
            }}
          >
            ＋ New chat
          </Button>
        </div>

        <div className="scroll-thin flex-1 overflow-y-auto p-2">
          <div className="px-2 py-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
            History
          </div>
          {conversations.length === 0 ? (
            <p className="px-2 py-3 text-xs text-slate-400">
              No conversations yet.
            </p>
          ) : (
            conversations.map((c) => (
              <button
                key={c.id}
                onClick={() => openConversation(c.id)}
                className={`group mb-0.5 block w-full rounded-lg px-2.5 py-2 text-left transition-colors ${
                  activeId === c.id
                    ? "bg-indigo-50"
                    : "hover:bg-slate-100"
                }`}
              >
                <div className="flex items-start justify-between gap-1">
                  <span
                    className={`line-clamp-2 text-xs ${
                      activeId === c.id
                        ? "font-medium text-indigo-800"
                        : "text-slate-700"
                    }`}
                  >
                    {c.title}
                  </span>
                  <span
                    onClick={(e) => removeConversation(c.id, e)}
                    className="opacity-0 transition-opacity group-hover:opacity-100 text-slate-300 hover:text-rose-500"
                    title="Delete"
                  >
                    ✕
                  </span>
                </div>
                <div className="mt-0.5 text-[11px] text-slate-400">
                  {Math.floor(c.message_count / 2)} question
                  {c.message_count === 2 ? "" : "s"}
                </div>
              </button>
            ))
          )}
        </div>
      </div>

      {/* ---------------- thread ---------------- */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Below md the history column is a drawer, so it needs a way in. */}
        <div className="flex items-center gap-2 border-b border-slate-200 bg-white px-4 py-2 md:hidden">
          <button
            onClick={() => setHistoryOpen(true)}
            className="rounded-lg border border-slate-300 px-2.5 py-1 text-xs text-slate-600"
          >
            ☰ History
          </button>
          <span className="truncate text-xs text-slate-400">{repositoryId}</span>
        </div>

        <div ref={threadRef} className="scroll-thin flex-1 overflow-y-auto">
          <div className="mx-auto max-w-3xl px-4 py-6 sm:px-8 sm:py-8">
            {messages.length === 0 ? (
              <div className="pt-10">
                <h1 className="text-xl font-semibold text-slate-900">
                  Ask about {repositoryId.split("__").slice(1).join("__") || repositoryId}
                </h1>
                <p className="mt-1 text-sm text-slate-500">
                  Answers come only from this repository&apos;s code. If it
                  isn&apos;t there, you&apos;ll be told so.
                </p>
                <div className="mt-6 grid gap-2">
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s}
                      onClick={() => ask(s)}
                      className="rounded-lg border border-slate-200 bg-white px-3.5 py-2.5 text-left text-sm text-slate-600 transition-colors hover:border-indigo-300 hover:text-slate-900"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="space-y-5">
                {messages.map((message) => (
                  <Turn key={message.id} message={message} />
                ))}
                {busy && (
                  <div className="flex items-center gap-2 text-sm text-slate-400">
                    <Spinner />
                    Searching the codebase…
                  </div>
                )}
              </div>
            )}
            {error && <ErrorNote onDismiss={() => setError("")}>{error}</ErrorNote>}
          </div>
        </div>

        <div className="border-t border-slate-200 bg-white px-4 py-3 sm:px-8 sm:py-4">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              ask();
            }}
            className="mx-auto flex max-w-3xl gap-2"
          >
            <Input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={
                activeId ? "Ask a follow-up…" : "Ask a question about the code…"
              }
              disabled={busy}
            />
            <Button type="submit" disabled={busy || !question.trim()}>
              Ask
            </Button>
          </form>
          {activeId && (
            <p className="mx-auto mt-2 max-w-3xl text-[11px] text-slate-400">
              Follow-ups use the conversation for context — you can say
              &ldquo;it&rdquo; or &ldquo;that file&rdquo;.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function Turn({ message }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-indigo-600 px-4 py-2.5 text-sm text-white">
          {message.content}
        </div>
      </div>
    );
  }

  const cited = (message.sources || []).filter((s) => s.cited);

  return (
    <div className="animate-rise">
      <div className="rounded-2xl rounded-bl-sm border border-slate-200 bg-white px-4 py-3.5">
        <AnswerText text={message.content} />

        {message.truncated && (
          <p className="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800">
            This answer hit the length limit and may be cut off. Try a narrower
            question.
          </p>
        )}

        {cited.length > 0 && (
          <div className="mt-4 border-t border-slate-100 pt-3">
            <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
              Relevant files
            </div>
            <div className="flex flex-wrap gap-1.5">
              {cited.map((source, index) => (
                <FilePath key={index}>
                  {source.file_path}
                  {source.start_line ? `:${source.start_line}` : ""}
                </FilePath>
              ))}
            </div>
          </div>
        )}

        <div className="mt-3 flex flex-wrap items-center gap-2 text-[11px] text-slate-400">
          {message.refused && <Badge tone="amber">not found in repo</Badge>}
          {message.model && <span>{message.model}</span>}
          {message.context_chunks != null && (
            <span>· {message.context_chunks} chunks</span>
          )}
          {message.elapsed_ms != null && (
            <span>· {(message.elapsed_ms / 1000).toFixed(1)}s</span>
          )}
        </div>

        {/* The rewritten question is worth exposing: it explains why a
            follow-up retrieved what it did. */}
        {message.resolved_question && (
          <details className="mt-2">
            <summary className="cursor-pointer text-[11px] text-slate-400">
              Searched for a rewritten question
            </summary>
            <p className="mt-1 text-xs italic text-slate-500">
              “{message.resolved_question}”
            </p>
          </details>
        )}
      </div>
    </div>
  );
}
