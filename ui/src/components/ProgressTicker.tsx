import { useEffect, useRef, useState } from "react";
import { progressBarClass, progressTrackClass } from "../lib/styles";

type ProgressTickerProps = {
  message: string;
  progress: number;
};

export function ProgressTicker({ message, progress }: ProgressTickerProps) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const textRef = useRef<HTMLSpanElement>(null);
  const [scroll, setScroll] = useState(false);

  const pct = Math.round(Math.min(Math.max(progress, 0), 1) * 100);
  const line = [message.trim() || "Обработка…", `${pct}%`].join(" · ");

  useEffect(() => {
    const viewport = viewportRef.current;
    const text = textRef.current;
    if (!viewport || !text) return;

    const measure = () => {
      setScroll(text.scrollWidth > viewport.clientWidth + 2);
    };

    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(viewport);
    observer.observe(text);
    return () => observer.disconnect();
  }, [line]);

  const tickerText = scroll ? `${line}    ${line}` : line;

  return (
    <div className="grid w-full max-w-lg gap-2" role="status" aria-live="polite" aria-atomic="true">
      <div
        ref={viewportRef}
        className="relative h-9 overflow-hidden rounded-md border border-[#e3e4e8] bg-white/95 px-3 shadow-sm"
      >
        <div
          className={`flex h-full items-center text-sm text-[#3b3e47] ${
            scroll ? "progress-ticker-marquee" : "justify-center"
          }`}
        >
          <span
            ref={textRef}
            className={scroll ? "progress-ticker-marquee-inner shrink-0 whitespace-nowrap" : "truncate"}
          >
            {tickerText}
          </span>
        </div>
      </div>
      <div className={progressTrackClass}>
        <div className={progressBarClass} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
