"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { getHealth } from "@/lib/api";

const NAV = [
  { href: "/", label: "Import", icon: "＋" },
  { href: "/chat", label: "Chat", icon: "◇" },
  { href: "/dashboard", label: "Dashboard", icon: "▤" },
];

export default function Shell({ children }) {
  const pathname = usePathname();
  const [health, setHealth] = useState(null);
  const [navOpen, setNavOpen] = useState(false);

  // Fetched once per navigation rather than on a timer: the UI only needs to
  // know whether setup is broken, and a poll would keep the tab busy.
  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((error) => setHealth({ unreachable: error.message }));
  }, [pathname]);

  // A drawer left open across a navigation would cover the page you just
  // asked for.
  useEffect(() => setNavOpen(false), [pathname]);

  return (
    <div className="lg:flex lg:min-h-screen">
      {/* Mobile bar. Below lg the 224px sidebar would leave ~166px of content
          on a 390px screen, so it becomes a drawer instead. */}
      <div className="panel sticky top-0 z-30 flex items-center justify-between border-b border-white/60 px-4 py-3 lg:hidden">
        <Link
          href="/"
          className="whitespace-nowrap text-xl font-extrabold tracking-tight"
        >
          <span className="brand-gradient">RepoSense AI</span>
        </Link>
        <button
          onClick={() => setNavOpen(true)}
          aria-label="Open navigation"
          className="rounded-lg px-2 py-1 text-slate-500 hover:bg-slate-100"
        >
          ☰
        </button>
      </div>

      {navOpen && (
        <button
          aria-label="Close navigation"
          onClick={() => setNavOpen(false)}
          className="fixed inset-0 z-40 bg-slate-900/20 lg:hidden"
        />
      )}

      <aside
        // 18rem rather than 15rem: the brand is set at 28px and has to sit on
        // one line, which needs ~220px of inner width after padding.
        className={`panel fixed inset-y-0 left-0 z-50 flex w-72 shrink-0 flex-col border-r border-white/70 transition-transform lg:static lg:z-auto lg:translate-x-0 ${
          navOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="hidden px-6 py-7 lg:block">
          <Link href="/" className="block">
            {/* The product name anchors the whole layout, so it gets display
                weight. whitespace-nowrap keeps it on one line even if the
                sidebar is ever narrowed again. */}
            <div className="whitespace-nowrap text-[28px] font-extrabold leading-none tracking-tight">
              <span className="brand-gradient">RepoSense AI</span>
            </div>
            <div className="mt-2.5 text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-400">
              Repository understanding
            </div>
          </Link>
        </div>

        <div className="flex items-center justify-between px-5 py-4 lg:hidden">
          <span className="text-xs font-semibold text-slate-500">Menu</span>
          <button
            onClick={() => setNavOpen(false)}
            className="text-slate-400 hover:text-slate-600"
            aria-label="Close navigation"
          >
            ✕
          </button>
        </div>

        <nav className="flex-1 px-3">
          {NAV.map((item) => {
            const active =
              item.href === "/"
                ? pathname === "/"
                : pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`mb-1 flex items-center gap-3 rounded-xl px-3.5 py-3 text-[15px] transition-all ${
                  active
                    ? "bg-white font-semibold text-indigo-700 shadow-sm ring-1 ring-indigo-100"
                    : "text-slate-600 hover:bg-white/70"
                }`}
              >
                <span
                  className={`w-4 text-center text-xs ${
                    active ? "text-indigo-500" : "text-slate-400"
                  }`}
                >
                  {item.icon}
                </span>
                {item.label}
              </Link>
            );
          })}
        </nav>

        <HealthFooter health={health} />
      </aside>

      <main className="min-w-0 flex-1">{children}</main>
    </div>
  );
}

function HealthFooter({ health }) {
  if (!health) {
    return <div className="border-t border-slate-100 px-5 py-4" />;
  }

  const problems = [];
  if (health.unreachable) {
    problems.push("Backend offline");
  } else {
    if (!health.qdrant?.reachable) problems.push("Qdrant offline");
    if (!health.groq?.configured) problems.push("No Groq key");
  }

  const ok = problems.length === 0;

  return (
    <div className="border-t border-slate-100 px-5 py-4">
      <div className="flex items-center gap-2">
        <span
          className={`h-1.5 w-1.5 shrink-0 rounded-full ${
            ok ? "bg-emerald-500" : "bg-amber-500"
          }`}
        />
        <span className="text-[11px] font-medium text-slate-500">
          {ok ? "All systems ready" : problems.join(" · ")}
        </span>
      </div>

      {/* Actionable hints, not just a status light - each failure mode here
          has a one-line fix. */}
      {!ok && (
        <p className="mt-1.5 text-[11px] leading-relaxed text-slate-400">
          {health.unreachable
            ? "Run .\\run_dev.ps1 in backend/"
            : !health.qdrant?.reachable
              ? "Run docker compose up -d"
              : "Add GROQ_API_KEY to backend/.env"}
        </p>
      )}

      {ok && (
        <p className="mt-1.5 text-[11px] text-slate-400">
          {health.qdrant.points} vectors ·{" "}
          {Object.keys(health.qdrant.repositories || {}).length} repos
        </p>
      )}
    </div>
  );
}
