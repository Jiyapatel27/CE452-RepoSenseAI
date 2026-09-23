// Shared UI primitives. Deliberately small and unabstracted - a handful of
// styled wrappers beats a component library for a project this size.

export function Button({
  children,
  variant = "primary",
  size = "md",
  className = "",
  ...props
}) {
  const base =
    "inline-flex items-center justify-center gap-2 rounded-lg font-medium " +
    "transition-colors disabled:cursor-not-allowed disabled:opacity-50";
  const sizes = {
    sm: "px-2.5 py-1.5 text-xs",
    md: "px-4 py-2 text-sm",
  };
  const variants = {
    primary: "btn-primary text-white",
    secondary:
      "bg-white text-slate-700 ring-1 ring-slate-200 shadow-sm hover:bg-slate-50 hover:ring-slate-300",
    ghost: "text-slate-600 hover:bg-white/70",
    danger: "text-rose-600 hover:bg-rose-50",
  };
  return (
    <button
      className={`${base} ${sizes[size]} ${variants[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

export function Input({ className = "", ...props }) {
  // No text-size class in the base. A `text-base` passed by a caller and a
  // `text-sm` here have identical specificity, so which one wins depends on
  // Tailwind's generated CSS order, not the class list - the override silently
  // lost. Callers set the size; `text-base` is the default via the element.
  return (
    <input
      className={
        "w-full rounded-lg bg-white px-3.5 py-2.5 text-base shadow-sm ring-1 " +
        "ring-slate-200 transition-shadow placeholder:text-slate-400 " +
        "focus:outline-none focus:ring-2 focus:ring-indigo-400 " +
        className
      }
      {...props}
    />
  );
}

export function Card({ children, className = "", padded = true, hover = false }) {
  return (
    <div
      className={
        "surface " +
        (hover ? "surface-hover " : "") +
        (padded ? "p-5 " : "") +
        className
      }
    >
      {children}
    </div>
  );
}

export function SectionTitle({ children, hint }) {
  return (
    <div className="mb-3 flex items-baseline justify-between gap-3">
      <h2 className="text-base font-semibold text-slate-900">{children}</h2>
      {hint && (
        <span className="shrink-0 text-xs text-slate-400">{hint}</span>
      )}
    </div>
  );
}

export function Stat({ label, value, sub }) {
  return (
    <div>
      <div className="text-xs font-medium uppercase tracking-wide text-slate-400">
        {label}
      </div>
      <div className="mt-1 text-2xl font-semibold tabular-nums text-slate-900">
        {value}
      </div>
      {sub && <div className="mt-0.5 text-xs text-slate-500">{sub}</div>}
    </div>
  );
}

export function Badge({ children, tone = "slate" }) {
  const tones = {
    slate: "bg-slate-100 text-slate-600",
    green: "bg-emerald-50 text-emerald-700",
    amber: "bg-amber-50 text-amber-700",
    red: "bg-rose-50 text-rose-700",
    indigo: "bg-indigo-50 text-indigo-700",
  };
  return (
    <span
      className={`inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

export function Spinner({ className = "h-4 w-4" }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none">
      <circle
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="3"
        className="opacity-25"
      />
      <path
        d="M22 12a10 10 0 0 1-10 10"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function ErrorNote({ children, onDismiss }) {
  return (
    <div className="mt-3 flex items-start gap-2 rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-700">
      <span className="flex-1">{children}</span>
      {onDismiss && (
        <button onClick={onDismiss} className="text-rose-400 hover:text-rose-600">
          ✕
        </button>
      )}
    </div>
  );
}

export function Empty({ title, children }) {
  return (
    <div className="py-10 text-center">
      <p className="text-sm font-medium text-slate-600">{title}</p>
      {children && (
        <p className="mx-auto mt-1 max-w-sm text-xs text-slate-400">{children}</p>
      )}
    </div>
  );
}

export function FilePath({ children }) {
  return (
    <code className="rounded bg-slate-50 px-1.5 py-0.5 font-mono text-xs text-slate-700">
      {children}
    </code>
  );
}
