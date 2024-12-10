"use client";

interface ChunksDisplayProps {
  _text: string;
  chunk_spans: [number, number][];
}

const ChunksDisplay: React.FC<ChunksDisplayProps> = ({
  _text,
  chunk_spans,
}) => {
  return (
    <div className="w-full flex flex-col gap-2 items-center justify-center">
      {chunk_spans.map(([start, end], index) => {
        // Calculate context boundaries with limits
        const character_window = 100;
        const contextStart = Math.max(0, start - character_window);
        const contextEnd = Math.min(_text.length, end + character_window);

        return (
          <div key={index} className="w-full p-2 flex flex-col gap-2">
            <p className="text-sm text-primary">
              <span className="text-secondary text-xs">
                {_text.slice(contextStart, start)}
              </span>
              <span className="text-primary text-sm italic font-bold">
                "{_text.slice(start, end)}"
              </span>
              <span className="text-secondary text-xs">
                {_text.slice(end, contextEnd)}
              </span>
            </p>
            <p className="text-xs text-secondary">
              {start} - {end}
            </p>
          </div>
        );
      })}
    </div>
  );
};

export default ChunksDisplay;
