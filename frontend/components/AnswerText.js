"use client";

// The model answers in Markdown and consistently uses three constructs:
// **bold**, `inline code`, and "- " bullets. Rendered as plain text those show
// up as literal asterisks and backticks, which makes the primary output look
// broken. A full Markdown library would be overkill for three cases.
//
// Builds React elements rather than setting innerHTML, so model output can
// never inject markup.

function renderInline(text, keyPrefix) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);

  return parts.filter(Boolean).map((part, index) => {
    const key = `${keyPrefix}-${index}`;

    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      // The model nests these: **`src/app/auth.ts`** - bold wrapping code.
      // Recursing handles the inner backticks instead of showing them
      // literally. Depth is capped at 1 because the bold pattern excludes
      // asterisks, so the inner text cannot contain another bold span.
      return (
        <strong key={key} className="font-semibold text-slate-900">
          {renderInline(part.slice(2, -2), `${key}-b`)}
        </strong>
      );
    }

    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return (
        <code
          key={key}
          className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[0.85em] text-indigo-700"
        >
          {part.slice(1, -1)}
        </code>
      );
    }

    return <span key={key}>{part}</span>;
  });
}

/**
 * Split the answer into prose lines and fenced code blocks.
 *
 * Fences matter: answers about code routinely include ```typescript blocks,
 * and without this they render as literal backticks with the code squashed
 * into unformatted prose - losing the indentation that makes it readable.
 */
function parseBlocks(text) {
  const blocks = [];
  let prose = [];
  let code = null;

  for (const line of (text || "").split("\n")) {
    const fence = line.trim().match(/^```+\s*(\w+)?\s*$/);

    if (fence) {
      if (code === null) {
        // Opening fence: flush the prose collected so far.
        if (prose.length) blocks.push({ type: "prose", lines: prose });
        prose = [];
        code = { type: "code", language: fence[1] || "", lines: [] };
      } else {
        blocks.push(code);
        code = null;
      }
      continue;
    }

    if (code) code.lines.push(line);
    else prose.push(line);
  }

  // An unterminated fence still renders as code - the model ran out of tokens
  // mid-block, and showing it as prose would mangle the indentation.
  if (code) blocks.push(code);
  if (prose.length) blocks.push({ type: "prose", lines: prose });

  return blocks;
}

function Prose({ lines, keyBase }) {
  return lines.map((line, index) => {
    const trimmed = line.trim();
    const key = `${keyBase}-${index}`;
    if (!trimmed) return <div key={key} className="h-1.5" />;

    const bullet = trimmed.match(/^(?:[-*]|\d+\.)\s+(.*)$/);
    if (bullet) {
      return (
        <div key={key} className="flex gap-2 pl-1">
          <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-slate-300" />
          <span className="min-w-0">{renderInline(bullet[1], key)}</span>
        </div>
      );
    }

    return <p key={key}>{renderInline(trimmed, key)}</p>;
  });
}

export default function AnswerText({ text }) {
  const blocks = parseBlocks(text);

  return (
    <div className="space-y-1.5 text-sm leading-relaxed text-slate-700">
      {blocks.map((block, index) =>
        block.type === "code" ? (
          <pre
            key={index}
            className="scroll-thin my-2.5 overflow-x-auto rounded-lg bg-slate-50 p-3 text-xs leading-relaxed ring-1 ring-slate-200"
          >
            <code className="font-mono text-slate-800">
              {/* Trailing blank lines are noise; leading indentation is not. */}
              {block.lines.join("\n").replace(/\s+$/, "")}
            </code>
          </pre>
        ) : (
          <Prose key={index} lines={block.lines} keyBase={index} />
        )
      )}
    </div>
  );
}
