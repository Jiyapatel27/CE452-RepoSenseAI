"use client";

// The model answers in Markdown, and it consistently uses only three things:
// **bold**, `inline code`, and "- " bullets. Rendering those as literal
// asterisks and backticks makes the primary output look broken.
//
// A full Markdown library would be overkill for that (and the brief says avoid
// unnecessary dependencies), so this handles exactly those three cases and
// leaves everything else as plain text. It builds React elements rather than
// setting innerHTML, so model output can never inject markup.

function renderInline(text, keyPrefix) {
  // Split on **bold** and `code`, keeping the delimiters so we can style them.
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);

  return parts.filter(Boolean).map((part, index) => {
    const key = `${keyPrefix}-${index}`;

    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      // The model nests these: **`src/app/auth.ts`** - bold wrapping code.
      // Recursing handles the inner backticks instead of showing them
      // literally. Depth is capped at 1 because the bold pattern excludes
      // asterisks, so the inner text can never contain another bold span.
      return (
        <strong key={key} className="font-semibold">
          {renderInline(part.slice(2, -2), `${key}-b`)}
        </strong>
      );
    }

    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return (
        <code
          key={key}
          className="rounded bg-slate-100 px-1 py-0.5 font-mono text-[0.85em] text-slate-800"
        >
          {part.slice(1, -1)}
        </code>
      );
    }

    return <span key={key}>{part}</span>;
  });
}

export default function AnswerText({ text }) {
  const lines = (text || "").split("\n");

  return (
    <div className="space-y-1 text-sm leading-relaxed">
      {lines.map((line, index) => {
        const trimmed = line.trim();

        if (!trimmed) return <div key={index} className="h-2" />;

        // Bullets: "- item", "* item", or "1. item".
        const bullet = trimmed.match(/^(?:[-*]|\d+\.)\s+(.*)$/);
        if (bullet) {
          return (
            <div key={index} className="flex gap-2 pl-1">
              <span className="text-slate-400">•</span>
              <span>{renderInline(bullet[1], index)}</span>
            </div>
          );
        }

        return <p key={index}>{renderInline(trimmed, index)}</p>;
      })}
    </div>
  );
}
