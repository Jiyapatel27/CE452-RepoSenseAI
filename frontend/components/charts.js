"use client";

// Inline-SVG charts. No charting library: each plot here is a handful of rows.
//
// Form was chosen per job, not per preference:
//   part-to-whole (<=6 segments)  -> donut          (languages)
//   trend over time, 1 series     -> area           (activity)
//   a single ratio vs a limit     -> radial meter   (refusal rate)
//   ranking / magnitude           -> horizontal bar (top files, score compare)
//
// A donut is explicitly NOT used for the rankings: a pie cannot be read as an
// ordering, and comparing near-equal slices by angle is unreliable.
//
// Colour: validated with the dataviz palette validator against the white card
// surface, all-pairs, light mode - worst CVD dE 9.2, worst normal-vision dE
// 16.3, all checks pass. Slot 4 is violet rather than the palette default
// yellow, which measures 13.7 against orange and fails the 15 floor.

const SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"];
const OTHER = "#94a3b8";
const GRID = "#e2e6ef";
const MUTED = "#8994a8";

/* ------------------------------------------------------------------ donut */

function arc(cx, cy, r, startAngle, endAngle) {
  const p = (angle) => {
    const rad = ((angle - 90) * Math.PI) / 180;
    return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)];
  };
  const [x1, y1] = p(startAngle);
  const [x2, y2] = p(endAngle);
  const large = endAngle - startAngle > 180 ? 1 : 0;
  return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
}

/**
 * Part-to-whole donut. Legitimate here because the job is share-at-a-glance
 * with few segments - not comparison of close values, and not a ranking.
 *
 * Segments past the 4th fold into a neutral "Other" rather than getting a
 * generated hue, which would be indistinguishable under CVD.
 */
export function DonutChart({ rows, total, centerLabel, centerValue }) {
  const size = 168;
  const stroke = 22;
  const radius = (size - stroke) / 2;
  const cx = size / 2;
  const cy = size / 2;
  const sum = total || rows.reduce((acc, r) => acc + r.value, 0) || 1;

  const top = rows.slice(0, 4);
  const rest = rows.slice(4);
  const segments = [...top];
  if (rest.length) {
    segments.push({
      label: `Other (${rest.length})`,
      value: rest.reduce((acc, r) => acc + r.value, 0),
      other: true,
    });
  }

  let angle = 0;
  const drawn = segments.map((segment, index) => {
    const sweep = (segment.value / sum) * 360;
    // 2px surface gap between fills, per the mark spec - never a border.
    const gap = sweep > 8 ? 2 : 0;
    const item = {
      ...segment,
      color: segment.other ? OTHER : SERIES[index % SERIES.length],
      start: angle,
      end: angle + sweep - gap,
      percent: (segment.value / sum) * 100,
    };
    angle += sweep;
    return item;
  });

  return (
    <div className="flex flex-col items-center gap-5 sm:flex-row sm:items-center">
      <svg
        viewBox={`0 0 ${size} ${size}`}
        className="h-40 w-40 shrink-0"
        role="img"
        aria-label={`${centerLabel}: ${segments
          .map((s) => `${s.label} ${s.value}`)
          .join(", ")}`}
      >
        <circle cx={cx} cy={cy} r={radius} fill="none" stroke={GRID} strokeWidth={stroke} />
        {drawn.map((segment) => (
          <path
            key={segment.label}
            d={arc(cx, cy, radius, segment.start, segment.end)}
            fill="none"
            stroke={segment.color}
            strokeWidth={stroke}
            strokeLinecap="butt"
          >
            <title>{`${segment.label}: ${segment.value} (${segment.percent.toFixed(0)}%)`}</title>
          </path>
        ))}
        {/* The hero number sits in the hole - proportional figures, system
            sans, no tabular-nums at display size. */}
        <text
          x={cx}
          y={cy - 2}
          textAnchor="middle"
          className="fill-slate-900"
          style={{ fontSize: 26, fontWeight: 600 }}
        >
          {centerValue}
        </text>
        <text
          x={cx}
          y={cy + 16}
          textAnchor="middle"
          fill={MUTED}
          style={{ fontSize: 10 }}
        >
          {centerLabel}
        </text>
      </svg>

      {/* Legend with values: identity is never colour-alone, and it supplies
          the "relief" the validator asks for where a slot is under 3:1. */}
      <ul className="w-full space-y-1.5">
        {drawn.map((segment) => (
          <li
            key={segment.label}
            className="flex items-center gap-2 text-xs"
          >
            <span
              className="h-2.5 w-2.5 shrink-0 rounded-sm"
              style={{ background: segment.color }}
            />
            <span className="flex-1 truncate text-slate-600">{segment.label}</span>
            <span className="tabular-nums text-slate-400">
              {segment.percent.toFixed(0)}%
            </span>
            <span className="w-8 text-right font-medium tabular-nums text-slate-700">
              {segment.value}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/* ------------------------------------------------------------- area chart */

/**
 * Trend over time, single series - so no legend, the title names it.
 * Area rather than bars: it reads as a continuous rate of activity, and the
 * gradient fill gives the card visual weight without a loud block.
 */
export function AreaChart({ data, height = 150 }) {
  if (!data.length) return null;

  const width = 620;
  const pad = { top: 14, right: 10, bottom: 24, left: 30 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const max = Math.max(...data.map((d) => d.questions), 1);
  const top = max <= 4 ? max : Math.ceil(max / 5) * 5;

  // A single point has no line to draw, so give it a flat two-point series.
  const points = (data.length === 1 ? [data[0], data[0]] : data).map((d, i, arr) => ({
    x: pad.left + (plotW / Math.max(arr.length - 1, 1)) * i,
    y: pad.top + plotH - (d.questions / top) * plotH,
    d,
  }));

  const line = points.map((p) => `${p.x} ${p.y}`).join(" L ");
  const area =
    `M ${points[0].x} ${pad.top + plotH} L ${line} ` +
    `L ${points[points.length - 1].x} ${pad.top + plotH} Z`;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full"
      role="img"
      aria-label="Questions asked per day"
    >
      <defs>
        <linearGradient id="areaFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={SERIES[0]} stopOpacity="0.22" />
          <stop offset="100%" stopColor={SERIES[0]} stopOpacity="0.02" />
        </linearGradient>
      </defs>

      {[0, top / 2, top]
        .filter((t, i, a) => a.indexOf(t) === i)
        .map((tick) => {
          const y = pad.top + plotH - (tick / top) * plotH;
          return (
            <g key={tick}>
              {/* Solid hairlines, one shade off the surface - never dashed. */}
              <line
                x1={pad.left}
                x2={width - pad.right}
                y1={y}
                y2={y}
                stroke={GRID}
                strokeWidth="1"
              />
              <text x={pad.left - 7} y={y + 3} textAnchor="end" fontSize="9" fill={MUTED}>
                {Math.round(tick)}
              </text>
            </g>
          );
        })}

      <path d={area} fill="url(#areaFill)" />
      <path
        d={`M ${line}`}
        fill="none"
        stroke={SERIES[0]}
        strokeWidth="2"
        strokeLinejoin="round"
        strokeLinecap="round"
      />

      {points.map((p, i) => (
        <g key={i}>
          {/* >=8px marker with a 2px surface ring. */}
          <circle cx={p.x} cy={p.y} r="4" fill="#ffffff" stroke={SERIES[0]} strokeWidth="2" />
          {/* Oversized invisible hit area - the visible dot is too small to aim at. */}
          <circle cx={p.x} cy={p.y} r="12" fill="transparent">
            <title>{`${p.d.day}: ${p.d.questions} question(s), ${p.d.refusals} refused`}</title>
          </circle>
          {(i === 0 || i === points.length - 1) && (
            <text x={p.x} y={height - 8} textAnchor="middle" fontSize="9" fill={MUTED}>
              {p.d.day.slice(5)}
            </text>
          )}
        </g>
      ))}
    </svg>
  );
}

/* ----------------------------------------------------------------- meter */

/**
 * A single ratio against a limit -> a meter, not a 2-slice pie.
 * Track and fill are the same ramp so the fill reads as "part of a whole".
 */
export function RadialMeter({ value, label, sublabel, tone = "series" }) {
  const size = 132;
  const stroke = 12;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.max(0, Math.min(value, 1));
  // 270-degree sweep leaves an open base, which reads as a gauge.
  const sweep = 0.75;
  const dash = circumference * sweep * clamped;
  const rest = circumference - dash;

  const color = tone === "warn" ? "#eb6834" : SERIES[0];

  return (
    <div className="flex items-center gap-4">
      <svg
        viewBox={`0 0 ${size} ${size}`}
        className="h-28 w-28 shrink-0 rotate-[-225deg]"
        role="img"
        aria-label={`${label}: ${Math.round(clamped * 100)} percent`}
      >
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={GRID}
          strokeWidth={stroke}
          strokeDasharray={`${circumference * sweep} ${circumference}`}
          strokeLinecap="round"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeDasharray={`${dash} ${rest}`}
          strokeLinecap="round"
        />
      </svg>

      <div>
        <div className="text-2xl font-semibold text-slate-900">
          {Math.round(clamped * 100)}%
        </div>
        <div className="text-xs font-medium text-slate-600">{label}</div>
        {sublabel && (
          <div className="mt-0.5 text-[11px] text-slate-400">{sublabel}</div>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------- bars */

/** Horizontal bars for a labelled ranking. One hue: the labels carry identity. */
export function BarList({ rows, valueFormat = (v) => v, color = SERIES[0] }) {
  const max = Math.max(...rows.map((r) => r.value), 1);

  return (
    <div className="space-y-2.5">
      {rows.map((row) => (
        <div key={row.label}>
          <div className="mb-1 flex items-baseline justify-between gap-3">
            <span className="truncate text-xs text-slate-600" title={row.label}>
              {row.label}
            </span>
            {/* Direct value label, so bar length never has to be decoded. */}
            <span className="shrink-0 text-xs font-medium tabular-nums text-slate-500">
              {valueFormat(row.value)}
            </span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${Math.max((row.value / max) * 100, 2)}%`,
                background: `linear-gradient(90deg, ${color}cc, ${color})`,
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * Answered vs refused best-match score. Two series, both directly labelled
 * with name and value, so colour is a redundant channel rather than the only one.
 */
export function ScoreComparison({ answered, refused }) {
  const rows = [
    { label: "Answered", ...answered, color: SERIES[0] },
    { label: "Refused", ...refused, color: SERIES[1] },
  ];

  return (
    <div className="space-y-3.5">
      {rows.map((row) => (
        <div key={row.label}>
          <div className="mb-1 flex items-baseline justify-between">
            <span className="flex items-center gap-1.5 text-xs text-slate-600">
              <span className="h-2.5 w-2.5 rounded-sm" style={{ background: row.color }} />
              {row.label}
              <span className="text-slate-400">({row.answers})</span>
            </span>
            <span className="text-xs font-medium tabular-nums text-slate-500">
              {row.avg_best_score ? row.avg_best_score.toFixed(3) : "—"}
            </span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                // Scores are 0-1, so the bar maps the score directly.
                width: `${Math.max((row.avg_best_score || 0) * 100, row.answers ? 2 : 0)}%`,
                background: `linear-gradient(90deg, ${row.color}cc, ${row.color})`,
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
